# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Utilities to process and compute parameters. Module for Algebra calculus.
"""
import copy 
import inspect 
import warnings 
import itertools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import scipy.stats as spstats
from scipy._lib._util import float_factorial
from scipy.cluster.hierarchy import linkage 
from scipy.linalg import lstsq
from scipy.ndimage import convolve1d
from scipy.stats import rankdata
from scipy.signal import argrelextrema 
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize, LabelEncoder 

from .._gofastlog import gofastlog
from ..api.box import KeyBox
from ..api.docstring import refglossary
from ..api.types import _T, _F,_SP, List, Tuple, Union
from ..api.types import ArrayLike, NDArray, Optional
from ..api.types import Series, DataFrame,  Dict
from ..api.summary import ResultSummary 
from ..decorators import AppendDocReferences
from ..exceptions import SiteError

from ._arraytools import axis_slice
from .coreutils import _assert_all_types, _validate_name_in
from .coreutils import concat_array_from_list
from .coreutils import find_close_position, normalize_string 
from .coreutils import to_numeric_dtypes, ellipsis2false
from .coreutils import smart_format, type_of_target, is_iterable 
from .coreutils import reshape, assert_ratio
from .validator import _is_arraylike_1d, _is_numeric_dtype, validate_multioutput 
from .validator import check_consistent_length 
from .validator import check_classification_targets, check_y
from .validator import assert_xy_in, _ensure_y_is_valid, ensure_non_negative
from .validator import check_epsilon, parameter_validator, is_binary_class 
from .validator import validate_sample_weights, validate_multiclass_target
from .validator import validate_length_range, validate_scores, ensure_2d
from .validator import is_frame 

_logger =gofastlog.get_gofast_logger(__name__)

mu0 = 4 * np.pi * 1e-7 

__all__=[
    'adaptive_moving_average', 
    'adjust_for_control_vars',
    'calculate_adjusted_lr', 
    'calculate_binary_iv', 
    'calculate_optimal_bins', 
    'calculate_residuals', 
    'compute_balance_accuracy',
    'compute_effort_yield',
    'compute_errors', 
    'compute_sunburst_data',
    'compute_cost_based_threshold', 
    'compute_sensitivity_specificity', 
    'compute_youdens_index',
    'cubic_regression', 
    'exponential_regression', 
    'get_bearing', 
    'get_distance',
    'infer_sankey_columns', 
    'label_importance',
    'linear_regression',
    'linkage_matrix', 
    'logarithmic_regression',
    'make_mxs', 
    'minmax_scaler',
    'normalize',
    'optimized_spearmanr', 
    'quadratic_regression', 
    'rank_data', 
    'savgol_filter', 
    'sinusoidal_regression',
    'standard_scaler', 
    'step_regression',
    'weighted_spearman_rank', 
    'compute_p_values'
    ]

def compute_p_values(
    data, depvar,
    method: str='pearson', 
    significance_threshold: float=0.05, 
    ignore: Optional [Union [str, list]]=None
    ):
    """
    Compute p-values for the correlation between each independent variable
    and a dependent variable.

    Parameters
    ----------
    data : pandas DataFrame
        The DataFrame containing the dataset.
    depvar : str or pandas Series
        The name of the dependent variable as a string or a pandas Series.
    method : {'pearson', 'spearman', 'kendall'}, optional
        The correlation method to use. Default is 'pearson'.
    significance_threshold : float, optional
        The significance threshold for p-values. Default is 0.05.
    ignore : str or list, optional
        Columns to ignore during computation.

    Returns
    -------
    p_values : :class:`gofast.api.summary.ResultSummary` object 
        A dictionary containing independent variables as keys and their 
        corresponding p-values.

    Raises
    ------
    ValueError
        If depvar is not found in DataFrame columns or if an invalid 
        correlation method is specified.

    Notes
    -----
    - If depvar is a string, it checks whether it exists in the DataFrame's 
      columns. If it doesn't exist, it raises a ValueError.
    - The function computes p-values for the specified correlation method 
      between each independent variable
      and the dependent variable. It excludes the depvar column from the 
      computation if it's in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.tools.mathex import compute_p_values
    >>> np.random.seed(0)
    >>> data = pd.DataFrame({
    ...     'x1': np.random.randn(100),
    ...     'x2': np.random.randn(100),
    ...     'y': np.random.randn(100)
    ... })
    >>> p_values = compute_p_values(data, 'y', method='pearson') 
    >>> print(p_values ) 
    P-values(
      {

           x1 : None
           x2 : None

      }
    )

    [ 2 entries ]
    >>> p_values = compute_p_values(data, 'y', method='pearson',
    ...                             significance_threshold=None)
    >>> p_values
    Out[76]: <P-values with 2 entries. Use print() to see detailed contents.>

    >>> print(p_values)
    P-values(
      {

           x1 : 0.4516405974318084
           x2 : 0.5797578201347333

      }
    )

    [ 2 entries ]
    """
    if isinstance (data, pd.Series): 
        data = data.to_frame() 
        
    if not isinstance (data, pd.DataFrame): 
        raise TypeError(f"'data' should be a frame, not {type(data).__name__!r}")
        
    if isinstance (depvar, str):
        if depvar not in data.columns:
            raise ValueError(f"'{depvar}' not found in DataFrame columns.")
        depvar = data[depvar]
        data = data.drop(columns=depvar.name)
        
    elif hasattr (depvar, '__array__'): 
        depvar = depvar.squeeze () 
        if depvar.ndim ==2: 
            raise TypeError ("Dependent variable 'depvar' should be Series or"
                             " one-dimensional array, not a two-dimensional.")
        if not isinstance (depvar, pd.Series): 
            depvar = pd.Series ( depvar, name = 'depvar')
            
    if ignore is not None: 
        if isinstance ( ignore, str): 
            ignore =[ignore]
        # check whether column is in data 
        column2ignore = [ col for col in ignore if col in data.columns]
        data = data.drop (columns= column2ignore)
        
    check_consistent_length(data, depvar)
    
    corr_methods = {
        'pearson': pearsonr,
        'spearman': spearmanr,
        'kendall': kendalltau
    }
     
    # Select only numeric columns
    data = data.select_dtypes([np.number])
    if data.empty:
        raise ValueError("P-value calculations expect numeric data, but"
                         " the DataFrame contains no numeric data.")
    
    if method not in corr_methods:
        raise ValueError("Invalid correlation method. Supported methods:"
                         " 'pearson', 'spearman', 'kendall'.")

    p_values = {}
    for column in data.columns:
        if column != depvar.name and (ignore is None or column not in ignore):
            corr_func = corr_methods[method]
            corr, p_value = corr_func(data[column], depvar)
            if significance_threshold: 
                p_values[column] = ( 
                    p_value if p_value <= significance_threshold else "reject"
                    )
            else: 
                p_values[column] = p_value 
            
    p_values = ResultSummary("P-values").add_results(p_values)
    
    return p_values

def compute_balance_accuracy(
    y_true, y_pred, 
    epsilon:float=1e-15,
    zero_division: int=0, 
    strategy: str="ovr",
    normalize: bool=False, 
    sample_weight:Optional[ArrayLike]=None):
    """
    Compute the balanced accuracy for binary or multiclass classification 
    problems, determining the appropriate calculation method based on the 
    label type and specified strategy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels for the classification task.
    y_pred : array-like of shape (n_samples,)
        Predicted labels from the classifier.
    epsilon : float, default=1e-15
        A small constant added to the denominator in calculations to prevent
        division by zero.
    zero_division : int, default=0
        The value to return when a zero division occurs during calculation, 
        typically encountered when a class is missing in the predictions.
    strategy : {'ovr', 'ovo'}, default='ovr'
        Strategy for handling multiclass classification:
        - 'ovr': One-vs-Rest, calculates balanced accuracy for each class
          against all others.
        - 'ovo': One-vs-One, calculates balanced accuracy for each pair 
          of classes.
    normalize : bool, default=False
        If True, normalizes the confusion matrix before calculating the 
        balanced accuracy.
    sample_weight : array-like of shape (n_samples,), optional
        Weights for each sample. If provided, the calculation will take 
        these into account.

    Returns
    -------
    float or np.ndarray
        The balanced accuracy score. Returns a single float for binary 
        classification or an array of scores for multiclass classification, 
        depending on the strategy.

    Notes
    -----
    Balanced accuracy is particularly useful for evaluating classifiers 
    on imbalanced datasets. It calculates the average of recall obtained on 
    each class, effectively handling classes without bias.

    Examples
    --------
    >>> from gofast.tools.mathex import compute_balance_accuracy
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> compute_balance_accuracy(y_true, y_pred)
    0.833333333333333

    For multiclass using One-vs-Rest strategy:
    >>> y_true = [0, 1, 2, 0, 1]
    >>> y_pred = [0, 2, 1, 0, 1]
    >>> compute_balance_accuracy(y_true, y_pred, strategy='ovr')
    array([1.        , 0.58333333, 0.375     ])

    For multiclass using One-vs-One strategy:
    >>> compute_balance_accuracy(y_true, y_pred, strategy='ovo')
    array([1.        , 0.16666667, 0.16666667])
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)

    # Check whether y_true is binary and compute balanced accuracy accordingly
    if is_binary_class(y_true):
        return _compute_balanced_accuracy_binary(
            y_true=y_true, y_pred=y_pred, 
            zero_division=zero_division, 
            normalize=normalize, 
            sample_weight=sample_weight)

    # Validate the strategy parameter
    strategy = parameter_validator(
        "strategy", target_strs={"ovr", "ovo"})(strategy)
    
    labels = unique_labels(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Compute balanced accuracy based on the specified strategy
    if strategy == 'ovr':
        scores = _balanced_accuracy_ovr(
            y_true=y_true, y_pred=y_pred, 
            labels=labels, 
            epsilon=epsilon, 
            zero_division=zero_division, 
            normalize=normalize, 
            sample_weight=sample_weight)
    elif strategy == 'ovo':
        labels = np.unique(np.concatenate([y_true, y_pred]))
        scores = _balanced_accuracy_ovo(
            y_true=y_true, y_pred=y_pred, 
            labels=labels, sample_weight=sample_weight)

    return scores

def _compute_balanced_accuracy_binary(
    y_true, y_pred, epsilon=1e-15,
    zero_division=0, normalize =False, 
    sample_weight=None
    ):
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight )
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0] + epsilon)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + epsilon)
    return (sensitivity + specificity) / 2  if not np.isnan(
        sensitivity + specificity) else zero_division

def _balanced_accuracy_ovr(
    y_true, y_pred, labels, epsilon=1e-15,
    zero_division=0, normalize=False, 
    sample_weight =None
  ):
    bal_acc_scores = []
    for label in labels:
        binary_y_true = (y_true == label).astype(int)
        binary_y_pred = (y_pred == label).astype(int)
        score = _compute_balanced_accuracy_binary(
            binary_y_true, binary_y_pred, epsilon, zero_division,
            normalize=normalize, sample_weight= sample_weight )
        bal_acc_scores.append(score)
    return np.array(bal_acc_scores)

def _balanced_accuracy_ovo(y_true, y_pred, labels, sample_weight=None):
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)
    y_bin = label_binarize(y_true_encoded, classes=range(len(labels)))
    
    bal_acc_scores = []
    for i, label_i in enumerate(labels[:-1]):
        for j, label_j in enumerate(labels[i+1:]):
            specific_y_true = y_bin[:, i]
            specific_y_pred = y_bin[:, j]
            auc_score = roc_auc_score(
                specific_y_true, specific_y_pred, sample_weight=sample_weight)
            bal_acc_scores.append(auc_score)
            
    return np.array(bal_acc_scores)

def compute_cost_based_threshold(
    y_true, y_scores, costs, *, 
    sample_weight=None
    ):
    """
    Compute threshold using a cost-based approach.

    This method computes the total cost for each potential threshold and 
    selects the one that minimizes this cost.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_scores : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by decision_function on some classifiers).
    costs : tuple (C_FP, C_FN)
        Costs associated with false positives (C_FP) and false negatives (C_FN).
    sample_weight : array-like of shape (n_samples,), optional (default=None)
        Sample weights.

    Returns
    --------
    optimal_threshold : float
        The threshold that minimizes the total cost.
    total_cost : float
        The total cost at the optimal threshold.

    Examples
    ---------
    >>> from gofast.tools.mathex import compute_cost_based_threshold
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_scores = [0.1, 0.8, 0.3, 0.6, 0.9]
    >>> costs = (1, 2)  # Cost of FP = 1, Cost of FN = 2
    >>> optimal_threshold, total_cost = compute_cost_based_threshold(y_true, y_scores, costs)
    >>> print("Optimal Threshold:", optimal_threshold)
    Optimal Threshold: 0.8
    >>> print("Total Cost:", total_cost)
    Total Cost: 4.0
    """
    y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric=True)
    # Get unique thresholds from the scores
    thresholds = np.unique(y_scores)
    
    # Initialize variables to store the best threshold and minimum cost
    best_threshold = None
    C_FP, C_FN = validate_length_range(costs, param_name="costs")
    min_cost = float('inf')

    # If sample weights are not provided, set them to ones
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
        
    # Validate sample weights 
    sample_weight = validate_sample_weights(
        sample_weight, y_true, normalize=True)
    # Iterate over each unique threshold
    for threshold in thresholds:
        # Predict labels based on the current threshold
        predicted_labels = (y_scores >= threshold).astype(int)
        
        # Calculate false positives (FP) and false negatives (FN) using sample weights
        FP = np.sum((predicted_labels == 1) & (y_true == 0) & (sample_weight > 0))
        FN = np.sum((predicted_labels == 0) & (y_true == 1) & (sample_weight > 0))
        
        # Calculate total cost using the provided costs and weighted FP/FN counts
        total_cost = C_FP * FP + C_FN * FN
        
        # Update the minimum cost and best threshold if the current cost is lower
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = threshold

    # Return the best threshold and its corresponding minimum cost
    return best_threshold, min_cost

def compute_youdens_index(
    y_true, y_scores, *, 
    sample_weight=None, 
    pos_label=None, 
    drop_intermediate=True
    ):
    """
    Compute Youden's Index and optimal threshold.

    Youden's Index is a single statistic that captures the performance of a
    binary classification test. 
    It's defined as the maximum vertical distance between the ROC curve and 
    the diagonal line.
    Mathematically, it is calculated as:
    
    .. math::
        J = \text{True Positive Rate} - \text{False Positive Rate}

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_scores : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by decision_function on some classifiers).
    sample_weight : array-like of shape (n_samples,), optional (default=None)
        Sample weights.
    pos_label : int or str, optional (default=None)
        The label of the positive class.
    drop_intermediate : bool, optional (default=True)
        Whether to drop some suboptimal thresholds that do not appear
        on a ROC curve with vertical drops.

    Returns:
    --------
    optimal_threshold : float
        The optimal threshold that maximizes Youden's Index.
    optimal_youden : float
        The value of Youden's Index at the optimal threshold.
    
    Examples:
    ---------
    >>> from gofast.tools.mathex import compute_youdens_index
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_scores = [0.1, 0.8, 0.3, 0.6, 0.9]
    >>> optimal_threshold, optimal_youden = compute_youdens_index(y_true, y_scores)
    >>> print("Optimal Threshold:", optimal_threshold)
    Optimal Threshold: 0.8
    >>> print("Youden's Index:", optimal_youden)
    Youden's Index: 0.6
    
    """
    # Ensure input arrays y_true and y_scores are valid
    y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric=True)
    
    # Check if the class labels are binary
    if not is_binary_class(y_true):
        raise ValueError("Youden's index calculation requires binary class"
                         "labels. Provided labels are not binary.")

    # Validate scores as proper probability distributions
    y_scores = validate_scores(y_scores, y_true, mode="passthrough")

    # Calculate the ROC curve and the corresponding AUC
    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, 
        sample_weight=sample_weight, 
        drop_intermediate=drop_intermediate
    )
    
    # Calculate Youden's Index for each threshold
    youdens_index = tpr - fpr
    
    # Find the optimal threshold (maximizes Youden's Index)
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youdens_index[optimal_idx]
    
    return optimal_threshold, optimal_youden

def calculate_multiclass_avg_lr(
    y_true, y_pred, *, 
    strategy='ovr',
    consensus="positive", 
    sample_weight=None, 
    multi_output='uniform_average', 
    epsilon=1e-10
    ):
    """
    Calculate the average likelihood ratio for multiclass classification 
    based on the average sensitivity and specificity across all classes or 
    class pairs.

    This function supports one-versus-rest (OvR) and one-versus-one (OvO) 
    strategies for multiclass data and computes the likelihood ratio either 
    positively or negatively based on the specified consensus.

    Parameters
    ----------
    y_true : array-like
        True class labels as integers.
    y_pred : array-like
        Predicted class labels as integers.
    strategy : str, optional
        Specifies the computation strategy: 'ovr' for one-versus-rest or 'ovo' 
        for one-versus-one.
    consensus : str, optional
        'positive' for positive likelihood ratio or 'negative' for negative 
        likelihood ratio.
    sample_weight : array-like, optional
        Weights applied to classes in averaging sensitivity and specificity. 
        If None, equal weighting is assumed.
    epsilon : float, optional
        A small value to prevent division by zero in calculations. 
        Default is 1e-10.

    Returns
    -------
    float
        The computed average likelihood ratio.
    float
        The average sensitivity computed across classes or class pairs.
    float
        The average specificity computed across classes or class pairs.

    Examples
    --------
    >>> from gofast.tools.mathex import calculate_multiclass_avg_lr
    >>> y_true = [0, 1, 2, 2, 1, 0]
    >>> y_pred = [0, 2, 1, 2, 1, 0]
    >>> lr, avg_sens, avg_spec = calculate_multiclass_avg_lr(
    ...    y_true, y_pred, strategy='ovr', consensus='positive')
    >>> print(f"Likelihood Ratio: {lr:.2f}, 
    ...          Average Sensitivity: {avg_sens:.2f},
    ...          Average Specificity: {avg_spec:.2f}")
    """
    _ensure_y_is_valid( y_true, y_pred, y_numeric =True)
    ensure_non_negative(
        y_true, y_pred,
        err_msg="y_true and y_pred must contain non-negative values."
    )
    epsilon = check_epsilon(epsilon, y_true, y_pred )

    consensus = parameter_validator( 
        "consensus", target_strs={'negative', 'positive'})(consensus)
    strategy = parameter_validator( 
        "strategy", target_strs={'ovr', 'ovo'})( strategy)
    
    classes = np.unique(y_true)
    sensitivities, specificities = [], []
    # validate multitarget and samples weigths if exists 
    y_true = validate_multiclass_target( y_true )
    
    if sample_weight is not None: 
        sample_weight =validate_sample_weights( sample_weight, y_true)
        
    if strategy == 'ovr':
        y_true_binarized = label_binarize(y_true, classes=classes)
        for i, cls in enumerate(classes):
            sensitivity, specificity = _compute_sensitivity_specificity(
                y_true_binarized[:, i], (y_pred == cls).astype(int),
                sample_weight=sample_weight,  
                epsilon=epsilon
            )
            sensitivities.append(sensitivity)
            specificities.append(specificity)

    elif strategy == 'ovo':
        for cls1, cls2 in itertools.combinations(classes, 2):
            relevant_mask = (y_true == cls1) | (y_true == cls2)
            y_true_binary = (y_true[relevant_mask] == cls1).astype(int)
            y_pred_binary = (y_pred[relevant_mask] == cls1).astype(int)
            sensitivity, specificity = _compute_sensitivity_specificity(
                y_true_binary, y_pred_binary,
                sample_weight=sample_weight,  
                epsilon=epsilon
                )
            sensitivities.append(sensitivity)
            specificities.append(specificity)

    # Weighted averages if sample_weight is provided
    if  multi_output=='uniform_average': 
        avg_sensitivity = np.average(sensitivities)
        avg_specificity = np.average(specificities)
    else: 
        avg_sensitivity = np.asarray(sensitivities)
        avg_specificity = np.asarray(specificities)
        
    # Compute LR based on average values
    lr = calculate_adjusted_lr(
        avg_sensitivity, avg_specificity, 
        consensus = consensus,max_lr = 100.
        )
    return lr, avg_sensitivity, avg_specificity

def calculate_multiclass_lr(
    y_true, y_pred, *, 
    consensus='positive',
    sample_weight=None, 
    strategy='ovr', 
    epsilon=1e-10, 
    multi_output='uniform_average', 
    apply_log_scale=False,
    include_metrics=False
    ):
    """
    Calculate the multiclass likelihood ratio for classification using either 
    one-versus-rest (OvR) or one-versus-one (OvO) strategies. Optionally applies 
    logarithmic scaling to the likelihood ratios and can return sensitivity and 
    specificity values.

    Parameters
    ----------
    y_true : array-like
        True class labels as integers.
    y_pred : array-like
        Predicted class labels as integers.
    consensus : str, optional
        Specifies the type of likelihood ratio to compute: 'positive' 
        (default) or 'negative'.
    strategy : str, optional
        Specifies the computation strategy: 'ovr' (one-versus-rest, default) 
        or 'ovo' (one-versus-one).
    epsilon : float, optional
        A small value to prevent division by zero in calculations. 
        Default is 1e-10. 
    multi_output : str, optional
        If 'uniform_average', returns the average of the computed likelihood 
        ratios. If 'raw_values', returns the likelihood ratios for each class 
        comparison.
    apply_log_scale : bool, optional
        If True, applies the natural logarithm to the likelihood ratios, 
        returning the log-likelihood ratios.
    include_metrics : bool, optional
        If True, returns a tuple containing the likelihood ratios and arrays 
        of sensitivities and specificities.

    Returns
    -------
    float or tuple
        Depending on 'multi_output' and 'include_metrics', returns either the 
        average likelihood ratio, an array of likelihood ratios, or a tuple 
        containing the likelihood ratios and metrics arrays.

    Examples
    --------
    >>> from gofast.tools.mathex import calculate_multiclass_lr
    >>> y_true = [0, 1, 2, 2, 1, 0]
    >>> y_pred = [0, 2, 1, 2, 1, 0]
    >>> calculate_multiclass_lr(y_true, y_pred, consensus='positive', strategy='ovr')
    1.6765
    
    >>> from gofast.tools.mathex import calculate_multiclass_lr
    >>> y_true = [0, 1, 2, 2, 1, 0]
    >>> y_pred = [0, 2, 1, 2, 1, 0]
    >>> calculate_multiclass_lr(y_true, y_pred, consensus='positive', strategy='ovr')
    1.6765
    
    >>> calculate_multiclass_lr(y_true, y_pred, consensus='negative', strategy='ovo')
    0.9890
    
    Notes
    -----
    The likelihood ratio (LR) for a given class or class pair is calculated as:
    
    .. math::
        LR_+ = \\frac{\\text{sensitivity}}{1 - \\text{specificity}}
    
    or
    
    .. math::
        LR_- = \\frac{1 - \\text{sensitivity}}{\\text{specificity}}
    
    If `apply_log_scale` is True, the log-likelihood ratio (LLR) is computed, 
    which transforms the LR using the natural logarithm:
    
    .. math::
        LLR = \\log(LR)
    
    This transformation helps manage extreme values and improves the interpretability
    of the results, especially when dealing with very high or very low likelihood ratios.
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    ensure_non_negative(
        y_true, y_pred,
        err_msg="y_true and y_pred must contain non-negative values."
    )
    epsilon = check_epsilon(epsilon, y_true, y_pred)

    consensus = parameter_validator(
        "consensus", target_strs={'negative', 'positive'})(consensus)
    strategy = parameter_validator(
        "strategy", target_strs={'ovr', 'ovo'})(strategy)
    
    classes = np.unique(y_true)
    results = []
    sensitivities = []
    specificities = []
    y_true = validate_multiclass_target( y_true )
    if strategy == 'ovr':
        y_true_binarized = label_binarize(y_true, classes=classes)
        for i, label in enumerate(classes):
            # Isolate the class against all others
            sensitivity, specificity = _compute_sensitivity_specificity(
                y_true_binarized[:, i], (y_pred == label).astype(int), 
                sample_weight=sample_weight,  
                epsilon=epsilon
            )
            lr = calculate_adjusted_lr (
                sensitivity, specificity,
                consensus = consensus, 
                max_lr=1e1 
            )
            results.append(np.log(lr) if apply_log_scale else lr)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

    elif strategy == 'ovo':
        for cls1, cls2 in itertools.combinations(classes, 2):
            relevant_mask = (y_true == cls1) | (y_true == cls2)
            y_true_binary = (y_true[relevant_mask] == cls1).astype(int)
            y_pred_binary = (y_pred[relevant_mask] == cls1).astype(int)
            sensitivity, specificity = _compute_sensitivity_specificity(
                y_true_binary, y_pred_binary, 
                sample_weight=sample_weight,  
                epsilon= epsilon,
                )
            lr = calculate_adjusted_lr (
                sensitivity, specificity,
                consensus = consensus, max_lr=1e1 )
            results.append(np.log(lr) if apply_log_scale else lr)
            sensitivities.append(sensitivity)
            specificities.append(specificity)

    if multi_output == 'uniform_average':
        # Return raw values for each class comparison
        return (np.mean(results), np.asarray(sensitivities),
                np.asarray(specificities)) if include_metrics else np.mean(results)
    else:
        return (np.array(results), np.asarray(sensitivities),
                np.asarray(specificities)) if include_metrics else np.array(results)
     
def calculate_adjusted_lr(
    sensitivity, 
    specificity, 
    consensus="positive",
    max_lr=100,
    buffer=1e-2
    ):
    """
    Calculate the likelihood ratio with modifications to avoid extremely high 
    values, particularly when specificity is close to 1.

    Parameters
    ----------
    sensitivity : float or numpy.ndarray
        The probability of correctly identifying a true positive, expressed as a 
        scalar for single measurement or an array for multiple measurements. Each
        value represents the sensitivity for a given test or condition.
    
    specificity : float or numpy.ndarray
        The probability of correctly identifying a true negative, similarly
        expressed as either a scalar for a single measurement or an array for
        multiple measurements. Each value corresponds to the specificity of 
        a test or condition and must match the dimensions of `sensitivity` if
        provided as an array.

    consensus : str, optional
        Specifies the type of likelihood ratio to compute:
        - 'positive': Computes the positive likelihood ratio (LR+), which
          is sensitivity divided by (1 - specificity). This ratio indicates
          how much the odds of the disease increase when a test is positive.
        - 'negative': Computes the negative likelihood ratio (LR-), which
          is (1 - sensitivity) divided by specificity. This ratio indicates
          how much the odds of the disease decrease when a test is negative.
          Default is 'positive'.
    
    max_lr : float, optional
        The maximum allowed value for the likelihood ratio to prevent
        extreme values that could be misleading or difficult to interpret.
        Default is 100, which means likelihood ratios are capped at this value 
        to avoid disproportionately high results that could result from very
        small denominators.

    buffer : float, optional
        A small value added to the denominator in the likelihood ratio
        calculation to prevent division by near-zero, which can lead to
        extremely high values. This buffer ensures stability in the calculations
        by avoiding infinite or very large ratios. Default is 1e-2.

    Returns
    -------
    float or numpy.ndarray
        The adjusted likelihood ratio, either as a single value or an array of
        values, depending on the input format. Each ratio is capped at `max_lr`
        and adjusted for low denominators using `buffer`.

    Examples
    --------
    >>> from gofast.tools.mathex import calculate_adjusted_lr
    >>> calculate_adjusted_lr(0.95, 0.89)
    8.636363636363637

    >>> calculate_adjusted_lr(np.array([0.95, 0.80]), np.array([0.89, 0.90]), 
                              consensus='negative', max_lr=10)
    array([1.04545455, 2.        ])
  
    >>> calculate_adjusted_lr(0.99, 0.999, consensus='positive')
    99.0
    >>> calculate_adjusted_lr(0.80, 0.95, consensus='negative', max_lr=100,
    ... buffer=0.005)
    0.21052631578947364

    Notes
    -----
    The likelihood ratio (LR) is calculated based on the specified `consensus`:
    
    For a 'positive' likelihood ratio:
        
    .. math::
        LR_+ = \\frac{\\text{sensitivity}}{\\max(1 - \\text{specificity}, 
        \\text{buffer})}

    For a 'negative' likelihood ratio:
        
    .. math::
        LR_- = \\frac{1 - \\text{sensitivity}}{\\max(\\text{specificity}, 
        \\text{buffer})}

    This approach helps to manage situations where specificity is very close to 1,
    which would normally result in a very high LR due to a small denominator.
    The `max_lr` parameter caps the LR to a maximum value, preventing extremely
    high ratios that might be misleading or difficult to interpret in practical
    scenarios.
    """
    # Ensure input compatibility if sensitivity and specificity are 
    # provided as lists or tuples
    sensitivity = np.array(sensitivity)
    specificity = np.array(specificity)

    # Compute the likelihood ratio based on the given consensus
    if consensus == "positive":
        lr = sensitivity / np.maximum(1 - specificity, buffer)
    elif consensus == "negative":
        lr = (1 - sensitivity) / np.maximum(specificity, buffer)
    else:
        raise ValueError("Consensus must be either 'positive' or 'negative'")
    
    # Cap the likelihood ratios at the maximum allowed value
    lr = np.minimum(lr, max_lr)

    return lr

def _compute_sensitivity_specificity(
        y_true, y_pred, sample_weight=None, epsilon=1e-10):
    """
    Calculate sensitivity and specificity for binary classification results,
    optionally using sample weights to adjust the importance of each sample.

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (1 for positive class, 0 for negative class).
    y_pred : array-like
        Binary predicted labels (1 for positive class, 0 for negative class).
    sample_weight : array-like, optional
        Weights applied to each sample when calculating sensitivity and specificity.
        If None, all samples are assumed to have equal weight. Sample weights are
        typically used in datasets where some samples are more important than
        others or in the presence of class imbalance.
    epsilon : float, optional
        Small value added to denominators to avoid division by zero. 
        Default is 1e-10.

    Returns
    -------
    tuple
        A tuple containing the sensitivity and specificity values.

    Notes
    -----
    Sensitivity and specificity are calculated using the following formulas:
    
    .. math::
        \text{sensitivity} = \frac{\sum (w_i \cdot [y_{true}^i = 1 \cap y_{pred}^i = 1])}
        {\sum (w_i \cdot [y_{true}^i = 1]) + \epsilon}

    .. math::
        \text{specificity} = \frac{\sum (w_i \cdot [y_{true}^i = 0 \cap y_{pred}^i = 0])}
        {\sum (w_i \cdot [y_{true}^i = 0]) + \epsilon}

    Where \( w_i \) are the sample weights. If `sample_weight` is None,
    all weights are considered equal to 1.

    Examples
    --------
    >>> from gofast.tools.mathex import calculate_binary_metrics
    >>> y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    >>> y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
    >>> sensitivity, specificity = calculate_binary_metrics(y_true, y_pred)
    >>> print(f"Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
    Sensitivity: 0.75, Specificity: 0.75

    >>> weights = [1, 1, 2, 2, 1, 2, 1, 1]
    >>> sensitivity, specificity = calculate_binary_metrics(y_true, y_pred, sample_weight=weights)
    >>> print(f"Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}")
    Sensitivity: 0.80, Specificity: 0.83
    """
    y_true, y_pred = _ensure_y_is_valid( y_true, y_pred, y_numeric =True)
    ensure_non_negative(
        y_true, y_pred,
        err_msg="y_true and y_pred must contain non-negative values."
    )
    epsilon = check_epsilon(epsilon, y_true, y_pred )

    if sample_weight is not None:
        sample_weight = validate_sample_weights(sample_weight, y_true)
    
    # Calculate true positives, false positives, false negatives, and true negatives
    true_positives = np.sum(((y_true == 1) & (y_pred == 1)) * (
        sample_weight if sample_weight is not None else 1))
    false_positives = np.sum(((y_true == 0) & (y_pred == 1)) * (
        sample_weight if sample_weight is not None else 1))
    false_negatives = np.sum(((y_true == 1) & (y_pred == 0)) * (
        sample_weight if sample_weight is not None else 1))
    true_negatives = np.sum(((y_true == 0) & (y_pred == 0)) * (
        sample_weight if sample_weight is not None else 1))

    # Calculate sensitivity and specificity with weighted counts
    sensitivity = true_positives / (true_positives + false_negatives + epsilon)
    specificity = true_negatives / (true_negatives + false_positives + epsilon)

    return sensitivity, specificity

def compute_sensitivity_specificity(
    y_true, y_pred, *, 
    sample_weight=None, 
    strategy='ovr', 
    average='macro',
    epsilon="auto" 
    ):
    """
    Calculate sensitivity and specificity for binary or multiclass classification
    results, with support for different strategies and averaging methods. 

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels. Each element in this array should be an integer
        representing the actual class of the corresponding sample.
    y_pred : array-like of shape (n_samples,)
        Predicted class labels. Each element should correspond to the predicted
        class label for the sample.
    sample_weight : array-like of shape (n_samples,), optional
        Weights applied to each sample when calculating metrics. If not provided,
        all samples are assumed to have equal weight. Useful in cases of class
        imbalance or when some samples are more critical than others.
    strategy : {'ovr', 'ovo'}, default 'ovr'
        The strategy to calculate metrics:
        - 'ovr' (One-vs-Rest): Metrics are calculated by treating each class as
          the positive class against all other classes combined.
        - 'ovo' (One-vs-One): Metrics are calculated for each pair of classes.
          This can be more computationally intensive.
    average : {'macro', 'micro', 'weighted'}, default 'macro'
        Specifies the method to average sensitivity and specificity:
        - 'macro': Calculate metrics for each label, and find their unweighted mean.
        - 'micro': Calculate metrics globally across all classes.
        - 'weighted': Calculate metrics for each label, and find their average,
          weighted by the number of true instances for each label.
    epsilon : float or "auto", optional
        A small constant added to the denominator to prevent division by zero. If
        'auto', the machine epsilon for float64 is used. Defaults to "auto".

    Returns
    -------
    tuple
        A tuple containing the averaged or non-averaged sensitivity and specificity.

    Examples
    --------
    >>> from gofast.tools.mathex import compute_sensitivity_specificity
    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0, 1, 0, 1]
    >>> compute_sensitivity_specificity(y_true, y_pred)
    (0.5, 0.5)

    Notes
    -----
    Sensitivity (also known as recall or true positive rate) and specificity
    (true negative rate) are calculated from the confusion matrix as:

    .. math::
        \text{sensitivity} = \frac{TP}{TP + FN}

    .. math::
        \text{specificity} = \frac{TN}{TN + FP}

    Here, TP, TN, FN, and FP denote the counts of true positives, true negatives,
    false negatives, and false positives, respectively.

    - Micro averaging will sum the counts of true positives, false negatives, and
      false positives across all classes and then compute the metrics.
    - Macro averaging will compute the metrics independently for each class but
      then take the unweighted mean. This does not take label imbalance into
      account.
    - Weighted averaging will compute the metrics for each class, and find their
      average weighted by the number of true instances for each label, which
      helps in addressing label imbalance.

    The choice of averaging method depends on the balance of classes and the
    importance of each class in the dataset.
    """

    if is_binary_class(y_true, accept_multioutput= False) :
        return _compute_sensitivity_specificity (
            y_true, y_pred, sample_weight=sample_weight, 
            epsilon=epsilon # already defined. 
            ) 

    y_true, y_pred = _ensure_y_is_valid( y_true, y_pred, y_numeric =True)
    strategy = parameter_validator(
        "strategy", target_strs={'ovr', 'ovo'})(strategy)
    average = parameter_validator(
        "average", target_strs={'macro', 'micro', 'weighted'})(average)
 
    epsilon = check_epsilon(epsilon, y_true, y_pred , scale_factor =1e-10)
    # Encode labels to integer indices
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_true_indices = np.vectorize(label_to_index.get)(y_true)
    y_pred_indices = np.vectorize(label_to_index.get)(y_pred)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(labels)))

    # Sensitivity and specificity calculation
    sensitivity = np.zeros(len(labels))
    specificity = np.zeros(len(labels))

    if strategy == 'ovr':  # One-vs-Rest
        for i in range(len(labels)):
            TP = cm[i, i]
            FN = np.sum(cm[i, :]) - TP
            FP = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - (TP + FP + FN)

            sensitivity[i] = TP / (TP + FN + epsilon)
            specificity[i] = TN / (TN + FP + epsilon)

    elif strategy == 'ovo':
        # Retrieve unique class labels
        labels = np.unique(y_true)
        return _compute_ovo_sens_spec(y_true, y_pred, labels, 
                                      return_array=True
                 )
    # Handle averaging of results
    if average == 'macro':
        return np.mean(sensitivity), np.mean(specificity)
    elif average == 'weighted':
        support = np.sum(cm, axis=1)
        return (np.average(sensitivity, weights=support),
                np.average(specificity, weights=support))
    elif average == 'micro':
        total_TP = np.sum(np.diag(cm))
        total_FN = np.sum(np.sum(cm, axis=1) - np.diag(cm))
        total_FP = np.sum(np.sum(cm, axis=0) - np.diag(cm))
        total_TN = np.sum(cm) - (total_TP + total_FP + total_FN)
        micro_sensitivity = total_TP / (total_TP + total_FN + epsilon)
        micro_specificity = total_TN / (total_TN + total_FP + epsilon)
        return micro_sensitivity, micro_specificity

    return sensitivity, specificity 

def _compute_ovo_sens_spec(y_true, y_pred, labels, return_array=True):
    """
    Compute sensitivity and specificity for each pair of classes 
    (One-vs-One strategy).
    Optionally returns sensitivity and specificity as arrays, averaging results 
    over all relevant pairs for each class.

    Parameters:
    y_true : array-like
        Ground truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    labels : array-like
        Array of unique class labels.
    return_array : bool, default True
        If True, returns sensitivity and specificity as numpy arrays averaged 
        over all pairs involving each class. If False, returns dictionaries
        with pairs as keys.

    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray) or (dict, dict)
        Sensitivity and specificity either as arrays or dictionaries depending 
        on `return_array`.
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.tools.mathex import _compute_ovo_sens_spec
    >>> labels = np.array([0, 1, 2])
    >>> y_true = np.array([0, 1, 1, 2, 2, 0, 1, 2, 0])
    >>> y_pred = np.array([0, 1, 2, 2, 0, 0, 1, 0, 2])
    >>> sensitivity, specificity = _compute_ovo_sens_spec(y_true, y_pred, labels, return_array=True)
    >>> print("Sensitivity:", sensitivity)
    >>> print("Specificity:", specificity)
    Sensitivity: [0.66666667 0.66666667 0.66666667]
    Specificity: [0.66666667 1.         0.66666667]
    """
    sensitivity = {}
    specificity = {}
    
    # Iterate over all pairs of labels
    for i, j in itertools.combinations(labels, 2):
        # Filter data for the pair (i, j)
        pair_indices = (y_true == i) | (y_true == j)
        y_true_pair = y_true[pair_indices]
        y_pred_pair = y_pred[pair_indices]

        # Map labels i and j to 0 and 1 for binary classification in this pair
        y_true_binary = np.where(y_true_pair == i, 1, 0)
        y_pred_binary = np.where(y_pred_pair == i, 1, 0)

        # Calculate confusion matrix for the binary case
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[1, 0])
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        # Store sensitivity and specificity for this pair
        sensitivity[(i, j)] = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity[(i, j)] = TN / (TN + FP) if (TN + FP) > 0 else 0

    if return_array:
        # Convert to arrays, averaging results for each class across all pairs involving it
        sens_array = np.zeros(len(labels))
        spec_array = np.zeros(len(labels))

        for idx, label in enumerate(labels):
            # Averaging sensitivity and specificity for each label
            involved_pairs = [key for key in sensitivity if label in key]
            sens_array[idx] = np.mean([sensitivity[pair] for pair in involved_pairs])
            spec_array[idx] = np.mean([specificity[pair] for pair in involved_pairs])

        return sens_array, spec_array

    return sensitivity, specificity

def calculate_histogram_bins(
        data:ArrayLike,  bins:Union [int, str, list]='auto', range:Tuple[float, float]=None, normalize: bool=False):
    """
    Calculates histogram bin edges from data with optional normalization.

    Parameters
    ----------
    data : array_like
        The input data to calculate histogram bins for.
    bins : int, sequence of scalars, or str, optional
        The criteria to bin the data. If an integer, it defines the number 
        of equal-width bins in the given range. If a sequence, it defines the 
        bin edges directly. If a string, it defines the method used to calculate 
        the optimal bin width, as defined by numpy.histogram_bin_edges().
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is 
        simply (data.min(), data.max()).
        Values outside the range are ignored.
    normalize : bool, default False
        If True, scales the data to range [0, 1] before calculating bins.

    Returns
    -------
    bin_edges : ndarray
        The computed or specified bin edges.

    Examples
    --------
    >>> from gofast.tools.mathex import calculate_histogram_bins
    >>> data = np.random.randn(1000)
    >>> bins = calculate_histogram_bins(data, bins=30)
    >>> print(bins)

    Notes
    -----
    This function is particularly useful in data preprocessing for histogram plotting.
    Normalization before binning can be useful when dealing with data with outliers
    or very skewed distributions.
    """
    data = np.asarray (data)
    if normalize:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    bin_edges = np.histogram_bin_edges(data, bins=bins, range=range)
    return bin_edges

def rank_data(data, method='average'):
    """
    Assigns ranks to data, handling ties according to the specified method.
    This function supports several strategies for tie-breaking, making it
    versatile for ranking tasks in statistical analyses and machine learning.

    Parameters
    ----------
    data : array-like
        The input data to rank. This can be any sequence that can be converted
        to a numpy array.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to assign ranks to tied elements. The options are:
        - 'average': Assign the average of the ranks to the tied elements.
        - 'min': Assign the minimum of the ranks to the tied elements.
        - 'max': Assign the maximum of the ranks to the tied elements.
        - 'dense': Like 'min', but the next rank is always one greater than
          the previous rank (i.e., no gaps in rank values).
        - 'ordinal': Assign a unique rank to each element, with ties broken
          by their order in the data.

    Returns
    -------
    ranks : ndarray
        The ranks of the input data.

    Examples
    --------
    >>> from gofast.tools.mathex import rank_data
    >>> data = [40, 20, 30, 20]
    >>> rank_data(data, method='average')
    array([4. , 1.5, 3. , 1.5])

    >>> rank_data(data, method='min')
    array([4, 1, 3, 1])

    Notes
    -----
    The ranking methods provided offer flexibility for different ranking
    scenarios. 'average', 'min', and 'max' are particularly useful in
    statistical contexts where ties need to be accounted for explicitly,
    while 'dense' and 'ordinal' provide strategies for more ordinal or
    categorical data ranking tasks.

    References
    ----------
    - Freund, J.E., & Wilson, W.J. (1993). Statistical Methods, 2nd ed.
    - Gibbons, J.D., & Chakraborti, S. (2011). Nonparametric Statistical Inference.
    
    See Also
    --------
    scipy.stats.rankdata : Rank the data in an array.
    numpy.argsort : Returns the indices that would sort an array.
    """
    sorter = np.argsort(data)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(data))
    ranks = np.empty_like(data, dtype=float)
    valid_methods = ['average', 'min', 'max', 'dense', 'ordinal']
    method = normalize_string(
        method, target_strs=valid_methods, raise_exception=True, 
        return_target_only=True, error_msg= (
            f"Invalid method '{method}'. Expect {smart_format(valid_methods, 'or')} ")
        )
    if method == 'average':
        # Average ranks of tied groups
        ranks[sorter] = np.mean([np.arange(len(data))], axis=0)
    elif method == 'min':
        # Minimum rank for all tied entries
        ranks[sorter] = np.min([np.arange(len(data))], axis=0)
    elif method == 'max':
        # Maximum rank for all tied entries
        ranks[sorter] = np.max([np.arange(len(data))], axis=0)
    elif method == 'dense':
        # Like 'min', but rank always increases by 1 between groups
        dense_rank = 0
        prev_val = np.nan
        for i in sorter:
            if data[i] != prev_val:
                dense_rank += 1
                prev_val = data[i]
            ranks[i] = dense_rank
    elif method == 'ordinal':
        # Distinct rank for every entry, resolving ties arbitrarily
        ranks[sorter] = np.arange(len(data))
    
    return ranks

def optimized_spearmanr(
    y_true, y_pred, *, 
    sample_weight:Optional[ArrayLike]=None, 
    tie_method:str='average', 
    nan_policy:str='propagate', 
    control_vars:Optional[ArrayLike]=None,
    multioutput:str='uniform_average'
    ):
    """
    Compute Spearman's rank correlation coefficient with support for 
    sample weights, custom tie handling, and NaN policies. This function 
    extends the standard Spearman's rank correlation to offer more 
    flexibility and utility in statistical and machine learning applications.

    Parameters
    ----------
    y_true : array-like
        True values for calculating the correlation. Must be 1D.
    y_pred : array-like
        Predicted values, corresponding to y_true.
    sample_weight : array-like, optional
        Weights for each pair of values. Default is None, which gives
        equal weight to all values.
    tie_method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        Method to handle ranking ties. Default is 'average'.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains NaN. 'propagate' returns NaN,
        'raise' throws an error, 'omit' ignores pairs with NaN.
    control_vars : array-like, optional
        Control variables for partial correlation. Default is None.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Strategy for aggregating errors across multiple output dimensions:
        - 'raw_values' : Returns an array of RMSLE values for each output.
        - 'uniform_average' : Averages errors across all outputs.
    Returns
    -------
    float
        Spearman's rank correlation coefficient.

    Examples
    --------
    >>> from gofast.tools.mathex import optimized_spearmanr
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [5, 6, 7, 8, 7]
    >>> optimized_spearmanr(y_true, y_pred)
    0.8208

    Notes
    -----
    Spearman's rank correlation assesses monotonic relationships by using the 
    ranked values for each variable. It is a non-parametric measure of 
    statistical dependence between two variables [1]_.

    .. math::
        \\rho = 1 - \\frac{6 \\sum d_i^2}{n(n^2 - 1)}

    where \\(d_i\\) is the difference between the two ranks of each observation, 
    and \\(n\\) is the number of observations [2]_.

    This extended implementation allows for weighted correlation calculation, 
    handling of NaN values according to a specified policy, and consideration 
    of control variables for partial correlation analysis.

    References
    ----------
    .. [1] Spearman, C. (1904). "The proof and measurement of association 
           between two things".
    .. [2] Myer, K., & Waller, N. (2009). Applied Spearman's rank correlation. 
           Statistics in Medicine.

    See Also
    --------
    scipy.stats.spearmanr : Spearman correlation calculation in SciPy.
    """
    # Handle the multioutput scenario
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    check_consistent_length(y_true, y_pred) 
    results = []
    for i in range(y_true.shape[1]):
        corr = _compute_spearmanr(y_true[:, i], y_pred[:, i], sample_weight,
                                  tie_method, nan_policy, control_vars)
        results.append(corr)

    multioutput = validate_multioutput(multioutput )

    return np.array(results) if multioutput == 'raw_values' else np.mean(results)

def _compute_spearmanr(
        y_true, y_pred, sample_weight, tie_method, nan_policy, control_vars):
    # The key addition is the handling of multioutput by reshaping inputs if 
    # necessary and iterating over columns (or outputs) to compute Spearman's
    # correlation for each, aggregating the results according to the 
    # multioutput strategy.
    def _weighted_spearman_corr(ranks_true, ranks_pred, weights):
        """
        Computes Spearman's rank correlation with support for sample weights.
        """
        # Weighted mean rank
        mean_rank_true = np.average(ranks_true, weights=weights)
        mean_rank_pred = np.average(ranks_pred, weights=weights)

        # Weighted covariance and variances
        cov = np.average((ranks_true - mean_rank_true) * (
            ranks_pred - mean_rank_pred), weights=weights)
        var_true = np.average((ranks_true - mean_rank_true)**2, weights=weights)
        var_pred = np.average((ranks_pred - mean_rank_pred)**2, weights=weights)

        # Weighted Spearman's rank correlation
        spearman_corr = cov / np.sqrt(var_true * var_pred)
        return spearman_corr
    
    # Validate and clean data based on `nan_policy`
    valid_policies = ['propagate', 'raise', 'omit']
    nan_policy= normalize_string(
        nan_policy, target_strs= valid_policies, 
        raise_exception=True, deep=True, 
        return_target_only=True, 
        error_msg=(f"Invalid nan_policy: {nan_policy}")
    )
    if nan_policy == 'omit':
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true, y_pred = y_true[valid_mask], y_pred[valid_mask]
        if sample_weight is not None:
            sample_weight = sample_weight[valid_mask]
    elif nan_policy == 'raise':
        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            raise ValueError("Input values contain NaNs.")
    
    # Implement tie handling
    valid_methods = ['average', 'min', 'max', 'dense', 'ordinal']
    tie_method = normalize_string(
        tie_method, target_strs=valid_methods, raise_exception=True, 
        return_target_only=True, error_msg= (
            f"Invalid method '{tie_method}'. Expect {smart_format(valid_methods, 'or')} ")
        )
    # Rank data with specified tie handling method
    ranks_true = rankdata(y_true, method=tie_method)
    ranks_pred = rankdata(y_pred, method=tie_method)

    if control_vars is not None:
        ranks_true, ranks_pred = adjust_for_control_vars (
            ranks_true, ranks_pred, control_vars )
    
    # Compute weighted Spearman's rank correlation 
    # if sample_weight is provided
    if sample_weight is not None:
        corr = _weighted_spearman_corr(ranks_true, ranks_pred, sample_weight)
    else:
        corr = np.corrcoef(ranks_true, ranks_pred)[0, 1]
    return corr

def adjust_for_control_vars(y_true:ArrayLike, y_pred:ArrayLike, control_vars:Optional[Union[ArrayLike, List[ArrayLike]]]=None):
    """
    Adjusts y_true and y_pred for either regression or classification tasks by 
    removing the influence of control variables. 
    
    The function serves as a wrapper that decides the adjustment strategy 
    based on the type of task (regression or classification) inferred from y_true.

    Parameters
    ----------
    y_true : array-like
        True target values. The nature of these values (continuous for regression or 
        categorical for classification) determines the adjustment strategy.
    y_pred : array-like
        Predicted target values. Must have the same shape as `y_true`.
    control_vars : array-like or list of array-likes, optional
        Control variables to adjust for. Can be a single array or a list of arrays. 
        If None, no adjustment is performed.

    Returns
    -------
    adjusted_y_true : ndarray
        Adjusted true target values, with the influence of control variables 
        removed.
    adjusted_y_pred : ndarray
        Adjusted predicted target values, with the influence of control 
        variables removed.

    Notes
    -----
    The function dynamically determines whether the targets suggest a 
    regression or classification task and applies the appropriate adjustment
    method. For regression, the adjustment involves residualizing the targets 
    against the control variables. 
    For classification, the approach might involve stratification or other 
    methods to control for the variables' influence.

    In practice, this adjustment is crucial when control variables might 
    confound or otherwise influence the relationship between the predictors 
    and the target variable, potentially biasing the correlation measure.

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4])
    >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    >>> control_vars = np.array([1, 1, 2, 2])
    >>> adjusted_y_true, adjusted_y_pred = adjust_for_control_vars(
    ... y_true, y_pred, control_vars)
    # Adjusted values depend on the specific implementation for regression
    or classification.

    See Also
    --------
    adjust_for_control_vars_regression : 
        Function to adjust targets in a regression task.
    adjust_for_control_vars_classification : 
        Function to adjust targets in a classification task.

    References
    ----------
    .. [1] K. Pearson, "On the theory of contingency and its relation to 
           association and normal correlation," Drapers' Company Research 
           Memoirs (Biometric Series I), London, 1904.
    .. [2] D. C. Montgomery, E. A. Peck, and G. G. Vining, "Introduction to
           Linear Regression Analysis," 5th ed., Wiley, 2012.
    """
    if control_vars is None:
        return y_true, y_pred 
    # Convert control_vars to numpy array if not already
    control_vars = np.asarray(control_vars)
    
    # statistical method suitable for the specific use case.
    if type_of_target(y_true) =='continuous': 
        adjusted_y_true, adjusted_y_true = adjust_for_control_vars_regression(
            y_true, y_pred, control_vars)
    else: 
        # is classification 
        adjusted_y_true, adjusted_y_true = adjust_for_control_vars_classification(
            y_true, y_pred, control_vars)
   
    return adjusted_y_true, adjusted_y_true

def adjust_for_control_vars_regression(y_true:ArrayLike, y_pred: ArrayLike, control_vars: Union[ArrayLike, List[ArrayLike]]):
    """
    Adjust y_true and y_pred for regression tasks by accounting for the influence
    of specified control variables through residualization.

    This approach fits a linear model to predict y_true and y_pred solely based on
    control variables, and then computes the residuals. These residuals represent
    the portion of y_true and y_pred that cannot be explained by the control
    variables, effectively isolating the effect of the predictors of interest.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values for regression.
    y_pred : array-like of shape (n_samples,)
        Predicted target values for regression.
    control_vars : array-like or list of array-likes
        Control variables to adjust for. Can be a single array or a list of arrays.

    Returns
    -------
    adjusted_y_true : ndarray of shape (n_samples,)
        Adjusted true target values, with the influence of control variables removed.
    adjusted_y_pred : ndarray of shape (n_samples,)
        Adjusted predicted target values, with the influence of control variables removed.

    Raises
    ------
    ValueError
        If y_true or y_pred are not 1-dimensional arrays.

    Notes
    -----
    This function uses LinearRegression from sklearn.linear_model to fit models
    predicting y_true and y_pred from the control variables. The residuals from
    these models (the differences between the observed and predicted values) are
    the adjusted targets.

    The mathematical concept behind this adjustment is as follows:
    
    .. math::
        \text{adjusted\_y} = y - \hat{y}_{\text{control}}
        
    where :math:`\hat{y}_{\text{control}}` is the prediction from a linear model
    trained only on the control variables.

    Examples
    --------
    >>> from gofast.tools.mathex import adjust_for_control_vars_regression
    >>> y_true = np.array([3, 5, 7, 9])
    >>> y_pred = np.array([4, 6, 8, 10])
    >>> control_vars = np.array([1, 2, 3, 4])
    >>> adjusted_y_true, adjusted_y_pred = adjust_for_control_vars_regression(
    ... y_true, y_pred, control_vars)
    >>> print(adjusted_y_true)
    >>> print(adjusted_y_pred)

    References
    ----------
    .. [1] Freedman, D. A. (2009). Statistical Models: Theory and Practice. 
          Cambridge University Press.
    
    See Also
    --------
    sklearn.linear_model.LinearRegression
    """
    from sklearn.linear_model import LinearRegression
    # Convert inputs to numpy arrays for consistency
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    control_vars = np.asarray(control_vars)

    # Ensure the task is appropriate for the data
    if y_true.ndim > 1 or y_pred.ndim > 1:
        raise ValueError(
            "y_true and y_pred should be 1-dimensional arrays for regression tasks.")

    if control_vars is None or len(control_vars) == 0:
        # No adjustment needed if there are no control variables
        return y_true, y_pred

    # Check if control_vars is a single array; if so,
    # reshape for sklearn compatibility
    if control_vars.ndim == 1:
        control_vars = control_vars.reshape(-1, 1)

    # Adjust y_true based on control variables
    model_true = LinearRegression().fit(control_vars, y_true)
    residuals_true = y_true - model_true.predict(control_vars)

    # Adjust y_pred based on control variables
    model_pred = LinearRegression().fit(control_vars, y_pred)
    residuals_pred = y_pred - model_pred.predict(control_vars)

    return residuals_true, residuals_pred

def adjust_for_control_vars_classification(y_true:ArrayLike, y_pred:ArrayLike, control_vars:DataFrame):
    """
    Adjusts `y_true` and `y_pred` in a classification task by stratifying the
    data based on control variables. It optionally applies logistic regression
    within each stratum  for adjustment, aiming to refine predictions based 
    on the influence of control variables.

    Parameters
    ----------
    y_true : array-like
        True class labels. Must be a 1D array of classification targets.
    y_pred : array-like
        Predicted class labels, corresponding to `y_true`. Must be of the 
        same shape as `y_true`.
    control_vars : pandas.DataFrame
        DataFrame containing one or more columns that represent control 
        variables. These variables are used to stratify the data before 
        applying any adjustment logic.

    Returns
    -------
    adjusted_y_true : numpy.ndarray
        Adjusted array of true class labels, same as input `y_true` 
        (adjustment process does not alter true labels).
    adjusted_y_pred : numpy.ndarray
        Adjusted array of predicted class labels after considering the 
        stratification by control variables.

    Notes
    -----
    This function aims to account for potential confounders or additional
    information represented by control variables.
    Logistic regression is utilized within each stratum defined by unique 
    combinations of control variables to adjust predictions.
    The essence is to mitigate the influence of control variables on the 
    prediction outcomes, thereby potentially enhancing the prediction accuracy
    or fairness across different groups.

    The adjustment is particularly useful in scenarios where control variables
    significantly influence the target variable, and their effects need to be
    isolated from the primary predictive modeling process.

    Examples
    --------
    >>> from gofast.tools.mathex import adjust_for_control_vars_classification
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 0, 1, 1]
    >>> control_vars = pd.DataFrame({'age': [25, 30, 35, 40], 'gender': [0, 1, 0, 1]})
    >>> adjusted_y_true, adjusted_y_pred = adjust_for_control_vars_classification(
    ... y_true, y_pred, control_vars)
    >>> print(adjusted_y_pred)
    [0 0 1 1]

    The function does not modify `y_true` but adjusts `y_pred` based on 
    logistic regression adjustments within each stratum defined by 
    `control_vars`.

    See Also
    --------
    sklearn.metrics.classification_report : Compute precision, recall,
        F-measure and support for each class.
    sklearn.preprocessing.LabelEncoder : 
        Encode target labels with value between 0 and n_classes-1.
    sklearn.linear_model.LogisticRegression : 
        Logistic Regression (aka logit, MaxEnt) classifier.

    References
    ----------
    .. [2] J. D. Hunter. "Matplotlib: A 2D graphics environment", 
           Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
    .. [1] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python",
           Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder 
     
    # first check whether y_true and y_pred are classification data 
    y_true, y_pred = check_classification_targets(
        y_true, y_pred, strategy='custom logic')
    
    # Ensure input is a DataFrame for easier manipulation
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    for col in control_vars.columns:
        data[col] = control_vars[col]
    
    # Encode y_true and y_pred if they are not numerical
    le_true = LabelEncoder().fit(y_true)
    data['y_true'] = le_true.transform(data['y_true'])
    
    if not np.issubdtype(data['y_pred'].dtype, np.number):
        le_pred = LabelEncoder().fit(y_pred)
        data['y_pred'] = le_pred.transform(data['y_pred'])
    
    # Iterate over each unique combination of control variables (each stratum)
    adjusted_preds = []
    for _, group in data.groupby(list(control_vars.columns)):
        if len(group) > 1:  # Enough data for logistic regression
            # Apply logistic regression within each stratum
            lr = LogisticRegression().fit(group[control_vars.columns], group['y_true'])
            adjusted_pred = lr.predict(group[control_vars.columns])
            adjusted_preds.extend(adjusted_pred)
        else:
            # Not enough data for logistic regression, use original predictions
            adjusted_preds.extend(group['y_pred'])

    # Convert adjusted predictions back to original class labels
    adjusted_y_pred = le_true.inverse_transform(adjusted_preds)
    return np.array(y_true), adjusted_y_pred

def weighted_spearman_rank(
    y_true, y_pred, sample_weight,
    return_weighted_rank=False, 
    epsilon=1e-10 
    ):
    """
    Compute Spearman's rank correlation coefficient with sample weights,
    offering an extension to the standard Spearman's correlation by incorporating
    sample weights into the rank calculation. This method is particularly useful
    for datasets where some observations are more important than others.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values. Both `y_true` and `y_pred` must have the same length.
    sample_weight : array-like
        Weights for each sample, indicating the importance of each observation
        in `y_true` and `y_pred`. Must be the same length as `y_true` and `y_pred`.
    return_weighted_rank : bool, optional
        If True, returns the weighted ranks of `y_true` and `y_pred` instead of
        Spearman's rho. Default is False.
    epsilon : float, optional
        A small value added to the denominator to avoid division by zero in the
        computation of Spearman's rho. Default is 1e-10.

    Returns
    -------
    float or tuple of ndarray
        If `return_weighted_rank` is False (default), returns Spearman's rho,
        considering sample weights. If `return_weighted_rank` is True, returns
        a tuple containing the weighted ranks of `y_true` and `y_pred`.

    Notes
    -----
    The weighted Spearman's rank correlation coefficient is computed as:

    .. math::
        \\rho = 1 - \\frac{6 \\sum d_i^2 w_i}{\\sum w_i(n^3 - n)}

    where :math:`d_i` is the difference between the weighted ranks of each observation,
    :math:`w_i` is the weight of each observation, and :math:`n` is the number of observations.

    This function calculates weighted ranks based on the sample weights, adjusting
    the influence of each data point in the final correlation measure. It is useful
    in scenarios where certain observations are deemed more critical than others.

    Examples
    --------
    >>> from gofast.tools.mathex import weighted_spearman_corr
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [5, 6, 7, 8, 7]
    >>> sample_weight = [1, 1, 1, 1, 2]
    >>> weighted_spearman_corr(y_true, y_pred, sample_weight)
    0.8208

    References
    ----------
    .. [1] Myatt, G.J. (2007). Making Sense of Data, A Practical Guide to 
           Exploratory Data Analysis and Data Mining. John Wiley & Sons.

    See Also
    --------
    scipy.stats.spearmanr : Spearman rank-order correlation coefficient.
    numpy.cov : Covariance matrix.
    numpy.var : Variance.

    """
    # Check and convert inputs to numpy arrays
    y_true, y_pred, sample_weight = map(np.asarray, [y_true, y_pred, sample_weight])

    if str(epsilon).lower() =='auto': 
        epsilon = determine_epsilon(y_pred, scale_factor= 1e-10)
        
    # Compute weighted ranks
    def weighted_rank(data, weights):
        order = np.argsort(data)
        ranks = np.empty_like(order, dtype=float)
        cum_weights = np.cumsum(weights[order])
        total_weight = cum_weights[-1]
        ranks[order] = cum_weights / total_weight * len(data)
        return ranks
    
    ranks_true = weighted_rank(y_true, sample_weight)
    ranks_pred = weighted_rank(y_pred, sample_weight)
    
    if return_weighted_rank: 
        return ranks_true, ranks_pred
    # Compute covariance between the weighted ranks
    cov = np.cov(ranks_true, ranks_pred, aweights=sample_weight)[0, 1]
    
    # Compute standard deviations of the weighted ranks
    std_true = np.sqrt(np.var(ranks_true, ddof=1, aweights=sample_weight))
    std_pred = np.sqrt(np.var(ranks_pred, ddof=1, aweights=sample_weight))
    
    # Compute Spearman's rho
    rho = cov / ( (std_true * std_pred) + epsilon) 
    return rho

def calculate_optimal_bins(y_pred, method='freedman_diaconis', data_range=None):
    """
    Calculate the optimal number of bins for histogramming a given set of 
    predictions, utilizing various heuristics. This function supports the 
    Freedman-Diaconis rule, Sturges' formula, and the Square-root choice, 
    allowing users to select the most appropriate method based on their data 
    distribution and size.

    Parameters
    ----------
    y_pred : array-like
        Predicted probabilities for the positive class. This array should be 
        one-dimensional.
    method : str, optional
        The binning method to use. Options include:
        - 'freedman_diaconis': Uses the Freedman-Diaconis rule, which is 
          particularly useful for data with skewed distributions.
          .. math:: \text{bin width} = 2 \cdot \frac{IQR}{\sqrt[3]{n}}
        - 'sturges': Uses Sturges' formula, ideal for normal distributions 
          but may be suboptimal for large datasets or non-normal distributions.
          .. math:: \text{bins} = \lceil \log_2(n) + 1 \rceil
        - 'sqrt': Employs the Square-root choice, a simple rule that works well 
          for small datasets.
          .. math:: \text{bins} = \lceil \sqrt{n} \rceil
        Default is 'freedman_diaconis'.
    data_range : tuple, optional
        A tuple specifying the range of the data as (min, max). If None, the 
        minimum and maximum values in `y_pred` are used. Default is None.

    Returns
    -------
    int
        The calculated optimal number of bins.

    Raises
    ------
    ValueError
        If an invalid `method` is specified.

    Examples
    --------
    Calculate the optimal number of bins for a dataset of random predictions 
    using different methods:
    
    >>> from gofast.tools.mathex import calculate_optimal_bins
    >>> y_pred = np.random.rand(100)
    >>> print(calculate_optimal_bins(y_pred, method='freedman_diaconis'))
    9
    >>> print(calculate_optimal_bins(y_pred, method='sturges'))
    7
    >>> print(calculate_optimal_bins(y_pred, method='sqrt'))
    10

    References
    ----------
    - Freedman, D. and Diaconis, P. (1981). On the histogram as a density estimator:
      L2 theory. Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete, 
      57(4), 453-476.
    - Sturges, H. A. (1926). The choice of a class interval. Journal of the American 
      Statistical Association, 21(153), 65-66.
    """
    y_pred = np.asarray(y_pred)
    
    if data_range is not None:
        if not isinstance(data_range, tuple) or len(data_range) != 2:
            raise ValueError(
                "data_range must be a tuple of two numeric values (min, max).")
        if any(not np.isscalar(v) or not np.isreal(v) for v in data_range):
            raise ValueError("data_range must contain numeric values.")
        data_min, data_max = data_range
    else:
        data_min, data_max = np.min(y_pred), np.max(y_pred)
    # Handle case where data is uniform
    if data_min == data_max:
        return 1

    n = len(y_pred)
    
    method = normalize_string(
        method, target_strs=["freedman_diaconis","sturges","sqrt"], 
        return_target_only= True, match_method="contains", raise_exception=True,
        error_msg=("Invalid method specified. Choose among"
                   " 'freedman_diaconis', 'sturges', 'sqrt'.")
        )
    if method == 'freedman_diaconis':
        iqr = np.subtract(*np.percentile(y_pred, [75, 25]))  # Interquartile range
        if iqr == 0:  # Handle case where IQR is 0
            return max(1, n // 2)  # Fallback to avoid division by zero
        
        bin_width = 2 * iqr * (n ** (-1/3))
        optimal_bins = int(np.ceil((data_max - data_min) / bin_width))
    elif method == 'sturges':
        optimal_bins = int(np.ceil(np.log2(n) + 1))
    elif method == 'sqrt':
        optimal_bins = int(np.ceil(np.sqrt(n)))
 
    return max(1, optimal_bins)  # Ensure at least one bin

def calculate_binary_iv(
    y_true, 
    y_pred, 
    epsilon=1e-15, 
    method='base', 
    bins='auto',
    bins_method='freedman_diaconis', 
    data_range=None):
    """
    Calculate the Information Value (IV) for binary classification problems
    using a base or binning approach. This function provides flexibility in
    IV calculation by allowing for simple percentage-based calculations or
    detailed binning techniques to understand the predictive power across
    the distribution of predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities for the positive class.
    epsilon : float or 'auto', optional
        A small epsilon value added to probabilities to prevent division
        by zero in logarithmic calculations. If 'auto', dynamically determines
        an appropriate epsilon based on `y_pred`. Default is 1e-15.
    method : str, optional
        The method for calculating IV. Options are 'base' for a direct approach
        using the overall percentage of events, and 'binning' for a detailed
        analysis using bins of predicted probabilities. Default is 'base'.
    bins : int or 'auto', optional
        The number of bins to use for the 'binning' method. If 'auto', the
        optimal number of bins is calculated based on `bins_method`.
        Default is 'auto'.
    bins_method : str, optional
        Method to use for calculating the optimal number of bins when
        `bins` is 'auto'. Options include 'freedman_diaconis', 'sturges', 
        and 'sqrt'. Default is 'freedman_diaconis'.
    data_range : tuple, optional
        A tuple specifying the range of the data as (min, max) for bin
        calculation. If None, the range is derived from `y_pred`. Default is None.

    Returns
    -------
    float
        The calculated Information Value (IV).

    Raises
    ------
    ValueError
        If an invalid method is specified.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.mathex import calculate_binary_iv
    >>> y_true = np.array([0, 1, 0, 1, 1])
    >>> y_pred = np.array([0.1, 0.8, 0.2, 0.7, 0.9])
    >>> print(calculate_binary_iv(y_true, y_pred, method='base'))
    1.6094379124341003

    >>> print(calculate_binary_iv(y_true, y_pred, method='binning', bins=3,
    ...                           bins_method='sturges'))
    0.6931471805599453

    Notes
    -----
    The Information Value (IV) quantifies the predictive power of a feature or
    model in binary classification, illustrating its ability to distinguish
    between classes.

    - The 'base' method calculates IV using the overall percentage of events
      and non-events:

      .. math::
        IV = \sum ((\% \text{{ of non-events}} - \% \text{{ of events}}) \times
        \ln\left(\frac{\% \text{{ of non-events}} + \epsilon}
        {\% \text{{ of events}} + \epsilon}\right))

    - The 'binning' method divides `y_pred` into bins and calculates IV for
      each bin, summing up the contributions from all bins.

    References
    ----------
    - Freedman, D. and Diaconis, P. (1981). "On the histogram as a density estimator:
      L2 theory." Zeitschrift für Wahrscheinlichkeitstheorie und verwandte Gebiete.
    - Sturges, H. A. (1926). "The choice of a class interval." Journal of the American
      Statistical Association.
      
    """
    # Ensure y_true and y_pred are numpy arrays for efficient computation
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    
    # Validate and process epsilon
    if isinstance(epsilon, str):
        if epsilon == 'auto':
            epsilon = determine_epsilon(y_pred)
        else:
            raise ValueError(
                "Epsilon value 'auto' is acceptable or should be a numeric value.")
            
    elif not isinstance(epsilon, (int, float)):
        raise ValueError("Epsilon must be a numeric value or 'auto'.")
    else:
        epsilon = float(epsilon)

    # Validate method parameter
    msg_meth=copy.deepcopy(method)
    method = normalize_string(
        method, target_strs= ['base', 'binning'], match_method='contains', 
        return_target_only=True, raise_exception=True , error_msg= (
            f"Invalid method '{msg_meth}'. Use 'base' or 'binning'.")
        )
    # Base method for IV calculation
    if method == 'base':
        percent_events = y_true.mean()
        percent_non_events = 1 - percent_events
        return np.sum((percent_non_events - percent_events) * np.log(
            (percent_non_events + epsilon) / (percent_events + epsilon)))
    
    # Binning method for IV calculation
    elif method == 'binning':
        if isinstance (bins, str):
            if bins=='auto' and bins_method is None: 
                warnings.warn(
                    "The 'bins' parameter is set to 'auto', but no 'bins_method'"
                    " has been specified. Defaulting to the 'freedman_diaconis'"
                    " method for determining the optimal number of bins.")
                bins_method="freedman_diaconis" 
            bins = calculate_optimal_bins(
                y_pred, method=bins_method, data_range=data_range)
            
        elif not isinstance(bins, (int, float)) or bins < 1:
            raise ValueError("Bins must be 'auto' or a positive integer.")
            
        if isinstance ( bins, float): 
            bins =int (bins )
    
        bins_array = np.linspace(0, 1, bins + 1)
        digitized = np.digitize(y_pred, bins_array) - 1
        iv = 0
        
        for bin_index in range(len(bins_array) - 1):
            in_bin = digitized == bin_index
            if not in_bin.any(): # if np.sum(indices) == 0:
                continue # # Skip empty bins
            
            bin_true = y_true[in_bin]
            percent_events_bin = bin_true.mean()
            percent_non_events_bin = 1 - percent_events_bin
            bin_contribution = (percent_non_events_bin - percent_events_bin) * np.log(
                (percent_non_events_bin + epsilon) / (percent_events_bin + epsilon))
            iv += bin_contribution if not np.isnan(bin_contribution) else 0
        
        return iv

def determine_epsilon(y_pred, base_epsilon=1e-15, scale_factor=1e-5):
    """
    Determine an appropriate epsilon value based on the predictions.

    If any predicted value is greater than 0, epsilon is set as a fraction 
    of the smallest non-zero prediction to avoid division by zero in 
    logarithmic calculations. Otherwise, a default small epsilon value is used.

    Parameters
    ----------
    y_pred : array-like
        Predicted probabilities or values.
    base_epsilon : float, optional
        The minimum allowed epsilon value to ensure it's not too small, 
        by default 1e-15.
    scale_factor : float, optional
        The factor to scale the minimum non-zero prediction by to determine 
        epsilon, by default 1e-5.

    Returns
    -------
    float
        The determined epsilon value.
    """
    if not isinstance (y_pred, np.ndarray): 
        y_pred = np.asarray(y_pred )
    # if np.any(y_pred > 0):
    #     # Find the minimum non-zero predicted probability/value
    #     min_non_zero_pred = np.min(y_pred[y_pred > 0])
    #     # Use a fraction of the smallest non-zero prediction
    #     epsilon = min_non_zero_pred * scale_factor  
    #     # Ensure epsilon is not too small, applying a lower bound
    #     epsilon = max(epsilon, base_epsilon)
    # else:
    #     # Use the base epsilon if no predictions are greater than 0
    #     epsilon = base_epsilon

    # return epsilon
    positive_preds = y_pred[y_pred > 0]
    if positive_preds.size > 0:
        min_non_zero_pred = np.min(positive_preds)
        epsilon = max(min_non_zero_pred * scale_factor, base_epsilon)
    else:
        epsilon = base_epsilon
        
    return epsilon

def calculate_residuals(
    actual: ArrayLike, 
    predicted: Union[np.ndarray, List[ArrayLike]], 
    task_type: str = 'regression',
    predict_proba: Optional[ArrayLike] = None
) -> ArrayLike:
    """
    Calculate the residuals for regression, binary, or multiclass 
    classification tasks.

    Parameters
    ----------
    actual : np.ndarray
        The actual observed values or class labels.
    predicted : Union[np.ndarray, List[np.ndarray]]
        The predicted values for regression or class labels for classification.
        Can be a list of predicted probabilities for each class from predict_proba.
    task_type : str, default='regression'
        The type of task: 'regression', 'binary', or 'multiclass'.
    predict_proba : np.ndarray, optional
        Predicted probabilities for each class from predict_proba 
        (for classification tasks).

    Returns
    -------
    residuals : np.ndarray
        The residuals of the model.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.tools.mathex import calculate_residuals
    >>> # For regression
    >>> actual = np.array([3, -0.5, 2, 7])
    >>> predicted = np.array([2.5, 0.0, 2, 8])
    >>> residuals = calculate_residuals(actual, predicted, task_type='regression')
    >>> print(residuals)

    >>> # For binary classification
    >>> actual = np.array([0, 1, 0, 1])
    >>> predicted = np.array([0, 1, 0, 1])  # predicted class labels
    >>> residuals = calculate_residuals(actual, predicted, task_type='binary')
    >>> print(residuals)

    >>> # For multiclass classification with predict_proba
    >>> actual = np.array([0, 1, 2, 1])
    >>> predict_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], 
                                  [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]])
    >>> residuals = calculate_residuals(actual, None, task_type='multiclass',
                                        predict_proba=predict_proba)
    >>> print(residuals)
    """
    if task_type == 'regression':
        if predicted is None:
            raise ValueError("Predicted values must be provided for regression tasks.")
        return actual - predicted
    elif task_type in ['binary', 'multiclass']:
        if predict_proba is not None:
            if predict_proba.shape[0] != actual.shape[0]:
                raise ValueError("The length of predict_proba does not match "
                                 "the number of actual values.")
            # For each sample, find the predicted probability of the true class
            prob_true_class = predict_proba[np.arange(len(actual)), actual]
            residuals = 1 - prob_true_class  # Residuals are 1 - P(true class)
        elif predicted is not None:
            # For binary classification without probabilities, residuals 
            # are 0 for correct predictions and 1 for incorrect
            residuals = np.where(actual == predicted, 0, 1)
        else:
            raise ValueError("Either predicted class labels or predict_proba "
                             "must be provided for classification tasks.")
    else:
        raise ValueError("The task_type must be 'regression', 'binary', or"
                         " 'multiclass'.")

    return residuals

def infer_sankey_columns(data: DataFrame,
  ) -> Tuple[List[str], List[str], List[int]]:
    """
    Infers source, target, and value columns for a Sankey diagram 
    from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame from which to infer the source, target, and value columns.

    Returns
    -------
    Tuple[List[str], List[str], List[int]]
        Three lists containing the names of the source nodes, target nodes,
        and the values of the flows between them, respectively.

    Raises
    ------
    ValueError
        If the DataFrame does not contain at least two columns for source and target,
        and an additional column for value.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.mathex import infer_sankey_columns
    >>> df = pd.DataFrame({
    ...     'from': ['A', 'A', 'B', 'B'],
    ...     'to': ['X', 'Y', 'X', 'Y'],
    ...     'amount': [10, 20, 30, 40]
    ... })
    >>> sources, targets, values = infer_sankey_columns(df)
    >>> print(sources, targets, values)
    ['A', 'A', 'B', 'B'] ['X', 'Y', 'X', 'Y'] [10, 20, 30, 40]
    """
    if len(data.columns) < 3:
        raise ValueError("DataFrame must have at least three columns:"
                         " source, target, and value")

    # Heuristic: The source is often the first column, the target is the second,
    # and the value is the third or the one with numeric data
    numeric_cols = data.select_dtypes([float, int]).columns

    if len(numeric_cols) == 0:
        raise ValueError(
            "DataFrame does not contain any numeric columns for values")

    # Choose the first numeric column as the value by default
    value_col = numeric_cols[0]
    source_col = data.columns[0]
    target_col = data.columns[1]

    # If there's a 'source' or 'target' column, prefer that
    for col in data.columns:
        if 'source' in col.lower():
            source_col = col
        elif 'target' in col.lower():
            target_col = col
        elif 'value' in col.lower() or 'amount' in col.lower() or 'count' in col.lower():
            value_col = col

    # Check for consistency in data
    if data[source_col].isnull().any() or data[target_col].isnull().any():
        raise ValueError("Source and Target columns must not contain null values")

    if data[value_col].isnull().any():
        raise ValueError("Value column must not contain null values")

    # Extract the columns and return
    sources = data[source_col].tolist()
    targets = data[target_col].tolist()
    values = data[value_col].tolist()

    return sources, targets, values

def compute_sunburst_data(
    data: DataFrame, 
    hierarchy: Optional[List[str]] = None, 
    value_column: Optional[str] = None
  ) -> List[Dict[str, str]]:
    """
    Computes the data structure required for generating a sunburst chart from
    a DataFrame.
    
    The function allows for automatic inference of hierarchy and values if 
    not explicitly provided. This is useful for visualizing hierarchical 
    datasets where the relationship between parent and child categories is 
    important.

    The sunburst chart provides insights into the proportion of categories at 
    multiple levels of the hierarchy through their area size. It is especially 
    useful in identifying patterns and contributions of various parts to the 
    whole in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the hierarchical data. It should have columns 
        representing levels of the hierarchy and optionally a column for values.
    hierarchy : Optional[List[str]], optional
        The list of columns that represent the hierarchy levels, ordered from 
        top to bottom. If not provided, the function assumes all columns except
        the last one are part of the hierarchy.
    value_column : Optional[str], optional
        The name of the column that contains the values for each leaf node in 
        the sunburst chart. If not provided, the function will count the 
        occurrences of the lowest hierarchy level and use this count as the 
        value for each leaf node.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries where each dictionary represents a node in the
        sunburst chart with 'name', 'value', and 'parent' keys.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'A', 'B', 'B'],
    ...     'Subcategory': ['A1', 'A2', 'B1', 'B2']
    ... })
    >>> data = compute_sunburst_data(df)
    >>> print(data)
    [
        {'name': 'A', 'value': 2, 'parent': ''},
        {'name': 'B', 'value': 2, 'parent': ''},
        {'name': 'A1', 'value': 1, 'parent': 'A'},
        {'name': 'A2', 'value': 1, 'parent': 'A'},
        {'name': 'B1', 'value': 1, 'parent': 'B'},
        {'name': 'B2', 'value': 1, 'parent': 'B'}
    ]
    """
    
    is_frame ( data, df_only =True , raise_exception=True )
    # If hierarchy is not provided, infer it from all columns except the last
    if hierarchy is None:
        hierarchy = data.columns[:-1].tolist()
    
    # If value_column is not provided, create a 'Count' column and use 
    # it as the value column
    if value_column is None:
        data = data.assign(Count=1)
        value_column = 'Count'
    
    # Compute the values for each level of the hierarchy
    df_full = data.copy()
    for i in range(1, len(hierarchy)):
        df_level = df_full[hierarchy[:i+1] + [value_column]].groupby(
            hierarchy[:i+1]).sum().reset_index()
        df_full = pd.concat([df_full, df_level], ignore_index=True)
    
    df_full = df_full.drop_duplicates(subset=hierarchy).reset_index(drop=True)

    # Generate the sunburst data structure
    sunburst_data = [
        {"name": row[hierarchy[-1]], 
         "value": row[value_column], 
         "parent": row[hierarchy[-2]] if i > 0 else ""}
        for i in range(len(hierarchy))
        for _, row in df_full[hierarchy[:i+1] + [value_column]].iterrows()
    ]

    # Remove duplicates, preserve order, and return
    seen = set()
    return [x for x in sunburst_data if not (
        tuple(x.items()) in seen or seen.add(tuple(x.items())))]

def compute_effort_yield(
        d: ArrayLike,  reverse: bool = True
        ) -> Tuple[ArrayLike, np.ndarray]:
    """
    Compute effort and yield values from importance data for use in 
    ABC analysis or similar plots.

    This function takes an array of importance measures (e.g., weights, scores) 
    and computes the cumulative effort and corresponding yield. 
    The effort is the cumulative percentage of items when sorted by importance,
    and the yield is the cumulative sum of importance
    measures, also as a percentage of the total sum.

    Parameters
    ----------
    d : np.ndarray
        1D array of importance measures for each item or feature.
    reverse : bool, optional
        If True (default), sort the data in descending order 
        (highest importance first).
        If False, sort in ascending order (lowest importance first).
    
    Returns
    -------
    effort : np.ndarray
        The cumulative percentage of items considered, sorted by importance.
    yield_ : np.ndarray
        The cumulative sum of importance measures, normalized to the total 
        sum to represent the yield as a proportion of the total importance.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.tools.mathex import compute_effort_yield
    >>> importances = np.array([0.1, 0.4, 0.3, 0.2])
    >>> effort, yield_ = compute_effort_yield(importances)
    >>> print(effort)
    >>> print(yield_)
    
    This would output:
    >>> effort
    [0.25 0.5  0.75 1.  ]
    >>> yield_
    [0.4  0.7  0.9  1.  ]

    Note that the effort is simply the proportion of total items, 
    and the yield is the
    cumulative proportion of the sum of importances.
    """
    d = np.array (d)
    # Validate input data
    if not isinstance(d, np.ndarray) or d.ndim != 1:
        raise ValueError("Input data must be a one-dimensional numpy array.")
    
    if not np.issubdtype(d.dtype, np.number):
        raise ValueError("Input data must be a numpy array of numerical type.")

    # Sort the data by importance
    sorted_indices = np.argsort(d)
    sorted_data = d[sorted_indices]
    if reverse:
        sorted_data = sorted_data[::-1]

    # Calculate cumulative sum of the sorted data
    cumulative_data = np.cumsum(sorted_data)

    # Normalize cumulative sum to get yield as a proportion of the total sum
    yield_ = cumulative_data / cumulative_data[-1]

    # Calculate the effort as the proportion of total number of items
    effort = np.arange(1, d.size + 1) / d.size

    return effort, yield_

def make_mxs(
    y,
    yt,
    threshold=0.5, 
    star_mxs=True, 
    return_ymxs=False,
    mode="strict", 
    include_nan=False, 
    trailer="*"
    ):
    """
    Compute the similarity between labels in arrays true y and predicted yt. 
    
    Function transform yt based on these similarities, and create a new 
    array `ymxs` by filling NaN values in y with corresponding labels from 
    transformed yt. Handles NaN values in `yt` based on the `mode` and
    `include_nan` parameters. See more in [1]_

    Parameters
    ----------
    y : array-like
        The target array containing valid labels and potentially NaN values.
    yt : array-like
        The array containing predicted labels from KMeans.
    threshold : float, optional
        The threshold for considering a label in `y` as similar to a label 
        in `yt` (default is 0.5).
    star_mxs : bool, optional
        If True, appends `trailer` to labels in `yt` when similarity is found 
        (default is True).
    return_ymx : bool, optional
        If True, returns the mixed array `ymx`; otherwise, returns a 
        dictionary of label similarities (default is False).
    mode : str, optional
        "strict" or "soft" handling of NaN values in `yt` (default is "strict").
    include_nan : bool, optional
        If True and `mode` is "soft", includes NaN values in `yt` during 
        similarity computation (default is False).
    trailer : str, optional
        The string to append to labels in `yt` when `star_mxs` is True
        (default is "*").

    Returns
    -------
    array or dict
        Mixed array `ymx` if `return_ymx` is True; otherwise, a 
        dictionary representing similarities of labels in `y` and `yt`.

    Raises
    ------
    ValueError
        If `yt` contains NaN values in "strict" mode or if `trailer` 
        is a number.
        
    References
    -----------
    [1] Kouadio, K.L, Liu R., Liu J., A mixture Learning Strategy for predicting 
        permeability coefficient K (2024). Computers and Geosciences, doi:XXXXX 

    Examples
    --------
    >>> y = np.array([1, 2, np.nan, 4])
    >>> yt = np.array([1, 2, 3, 4])
    >>> make_mxs(y, yt, threshold=0.5, star_mxs=True, return_ymx=True, trailer="#")
    array([1, 2, '3#', '44#'])

    >>> make_mxs(y, yt, threshold=1.5, star_mxs=False, return_ymx=False, mode="soft")
    {1: True, 2: True, np.nan: False, 4: True}
    """
    from sklearn.metrics import pairwise_distances
    
    if not isinstance(trailer, str) or trailer.isdigit():
        raise ValueError("trailer must be a non-numeric string.")

    if mode == "strict" and np.isnan(yt).any():
        raise ValueError("yt should not contain NaN values in 'strict' mode.")

    # Appending trailer to yt if star_mxs is True
    yt_transformed = np.array([f"{label}{trailer}" for label in yt]
                              ) if star_mxs else yt.copy()

    # Computing similarities and transforming yt
    similarities = {}
    for i, label_y in enumerate(y):
        include_label = not np.isnan(label_y) or (include_nan and mode == "soft")
        if include_label:
            similarity = pairwise_distances([[label_y]], [[yt[i]]])[0][0] <= threshold
            similarities[label_y] = similarity
            if similarity and star_mxs:
                # Transform similar labels in yt
                label_yt_trailer = f"{yt[i]}{trailer}"
                yt_transformed[yt_transformed == label_yt_trailer
                               ] = f"{label_y}{label_yt_trailer}"
    # Filling NaN positions in y with corresponding labels from transformed yt
    ymxs = np.where(np.isnan(y), yt_transformed, y)
    
    return ymxs if return_ymxs else similarities

def label_importance(y, include_nan=False):
    """
    Compute the importance of each label in a target array.

    This function calculates the frequency of each unique label 
    in the target array `y`. Importance is defined as the proportion of 
    occurrences of each label in the array.

    Parameters
    ----------
    y : array-like
        The target array containing labels.
    include_nan : bool, optional
        If True, includes NaN values in the calculation, otherwise 
        excludes them (default is False).

    Returns
    -------
    dict
        A dictionary with labels as keys and their corresponding 
        importance as values.

    Notes
    -----
    The mathematical formulation for the importance of a label `l` is given by:

    .. math::

        I(l) = \\frac{\\text{{count of }} l \\text{{ in }} y}{\\text{{total number of elements in }} y}

    Examples
    --------
    >>> y = np.array([1, 2, 2, 3, 3, 3, np.nan])
    >>> label_importance(y)
    {1.0: 0.16666666666666666, 2.0: 0.3333333333333333, 3.0: 0.5}

    >>> label_importance(y, include_nan=True)
    {1.0: 0.14285714285714285, 2.0: 0.2857142857142857, 3.0: 0.42857142857142855,
     nan: 0.14285714285714285}
    """
    y = np.array ( y )
    if not include_nan:
        y = y[~np.isnan(y)]
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {label: count / total for label, count in zip(labels, counts)}

def linear_regression(X, coef, bias=0., noise=0.):
    """
    linear regression.
    
    Generate output for linear regression, modeling a relationship between
    features and a response using a linear approach.

    Linear regression is one of the simplest formss of regression, useful for
    understanding relationships between variables and for making predictions.
    It's widely used in various fields like economics, biology, and engineering.

    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Returns
    -------
    y : ndarray
        The output values for linear regression with shape (n_samples,).

    Formula
    -------
    y = X \cdot coef + bias + noise
    
    Applications
    ------------
    - Trend analysis in time series data.
    - Predictive modeling in business and finance.
    - Estimating relationships in scientific experiments.
    """
    return np.dot(X, coef) + bias + noise * np.random.randn(X.shape[0])

def quadratic_regression(X, coef, bias=0., noise=0.):
    """
    Quadratic regression.

    Generate output for quadratic regression, which models a parabolic 
    relationship between the dependent variable and independent variables.

    Quadratic regression is suitable for datasets with a non-linear trend. It's 
    often used in areas where the rate of change increases or decreases rapidly.

    Applications
    ------------
    - Modeling acceleration or deceleration patterns in physics.
    - Growth rate analysis in biology and economics.
    - Prediction in financial markets with parabolic trends.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.
        
    Formula
    -------
    y = (X^2) \cdot coef + bias + noise
    """
    return np.dot(X**2, coef) + bias + noise * np.random.randn(X.shape[0])

def cubic_regression(X, coef, bias=0., noise=0.):
    """
    Cubic regression.

    Generate output for cubic regression, fitting a cubic polynomial to the data.

    Cubic regression provides a more flexible curve than quadratic models and is 
    beneficial in studying more complex relationships, especially where inflection 
    points are present.

    Applications
    ------------
    - Analyzing drug response curves in pharmacology.
    - Studying the growth patterns of organisms or populations.
    - Complex trend analysis in economic data.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = (X^3) \cdot coef + bias + noise
    """
    return np.dot(X**3, coef) + bias + noise * np.random.randn(X.shape[0])

def exponential_regression(X, coef, bias=0., noise=0.):
    """
    Exponential regression.

    Generate output for exponential regression, ideal for modeling growth or decay.

    Exponential regression is used when data grows or decays at a constant
    percentage rate. It's crucial in fields like biology for population growth 
    studies or in finance for compound interest calculations.

    Applications
    ------------
    - Modeling population growth or decline.
    - Financial modeling for compound interest.
    - Radioactive decay in physics.
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = exp(X \cdot coef) + bias + noise
    """
    return np.exp(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def logarithmic_regression(X, coef, bias=0., noise=0.):
    """
    Logarithmic regression.

    Generate output for logarithmic regression, suitable for modeling processes 
    that rapidly increase or decrease and then level off.

    Logarithmic regression is particularly useful in situations where the rate of
    change decreases over time. It's often used in scientific data analysis.

    Applications
    ------------
    - Analyzing diminishing returns in economics.
    - Growth rate analysis in biological processes.
    - Signal processing and sound intensity measurements.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = log(X) \cdot coef + bias + noise
    """
    return np.dot(np.log(X), coef) + bias + noise * np.random.randn(X.shape[0])

def sinusoidal_regression(X, coef, bias=0., noise=0.):
    """
    Sinusoidal regression.

    Generate output for sinusoidal regression, fitting a sinusoidal model to the data.

    This type of regression is useful for modeling cyclical patterns and is commonly
    used in fields like meteorology, seasonal studies, and signal processing.

    Applications
    ------------
    - Seasonal pattern analysis in climatology.
    - Modeling cyclical trends in economics.
    - Signal analysis in electrical engineering.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = sin(X \cdot coef) + bias + noise
    """
    return np.sin(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def step_regression(X, coef, bias=0., noise=0.):
    """
    Step regression.

    Step regression is valuable for modeling scenarios where the dependent variable
    changes abruptly at specific thresholds. It's used in quality control and market
    segmentation analysis.

    Applications
    ------------
    - Quality assessment in manufacturing processes.
    - Customer segmentation in marketing.
    - Modeling sudden changes in environmental data.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.


    Formula
    -------
    y = step_function(X \cdot coef) + bias + noise

    Note: step_function returns 1 if x >= 0, else 0.
    """
    step_function = np.vectorize(lambda x: 1 if x >= 0 else 0)
    return step_function(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def standard_scaler(X, y=None):
    """
    Scales features to have zero mean and unit variance.

    Standard scaling is vital in many machine learning algorithms that are 
    sensitive to the scale of input features. It's commonly used in algorithms
    like Support Vector Machines and k-Nearest Neighbors.

    Applications
    ------------
    - Data preprocessing for machine learning models.
    - Feature normalization in image processing.
    - Standardizing variables in statistical analysis.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be scaled as well.

    Returns
    -------
    X_scaled : ndarray
        Scaled version of X.
    y_scaled : ndarray, optional
        Scaled version of y, if y is provided.

    Formula
    -------
    For each feature, the Standard Scaler performs the following operation:
        z = \frac{x - \mu}{\sigma}
    where \mu is the mean and \sigma is the standard deviation of the feature.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled = standard_scaler(X)
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    if y is not None:
        y_mean = y.mean()
        y_std = y.std()
        y_scaled = (y - y_mean) / y_std
        return X_scaled, y_scaled

    return X_scaled

def minmax_scaler(X, y=None):
    """
    Scales each feature to a given range, typically [0, 1].

    MinMax scaling is often used when the algorithm requires a bounded interval. 
    It's particularly useful in neural networks and image processing where values 
    need to be normalized.

    Applications
    ------------
    - Data normalization for neural networks.
    - Preprocessing data in computer vision tasks.
    - Scaling features for optimization problems.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be scaled as well.

    Returns
    -------
    X_scaled : ndarray
        Scaled version of X.
    y_scaled : ndarray, optional
        Scaled version of y, if y is provided.

    Formula
    -------
    The MinMax Scaler performs the following operation for each feature:
        z = \frac{x - \min(x)}{\max(x) - \min(x)}

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled = minmax_scaler(X)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)

    if y is not None:
        y_min = y.min()
        y_max = y.max()
        y_scaled = (y - y_min) / (y_max - y_min)
        return X_scaled, y_scaled

    return X_scaled

def normalize(X, y=None):
    """
    Scales individual samples to have unit norm.

    Normalization is critical for distance-based algorithms like k-Nearest 
    Neighbors and clustering algorithms. It ensures that each feature 
    contributes proportionately to the final distance.

    Applications
    ------------
    - Preprocessing for clustering algorithms.
    - Normalizing data in natural language processing.
    - Feature scaling in bioinformatics.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be normalized as well.

    Returns
    -------
    X_normalized : ndarray
        Normalized version of X.
    y_normalized : ndarray, optional
        Normalized version of y, if y is provided.

    Formula
    -------
    The Normalize method scales each sample as follows:
        z = \frac{x}{||x||}
    where ||x|| is the Euclidean norm (L2 norm) of the sample.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized = normalize(X)
    """
    X = ensure_2d(X)
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / X_norm

    if y is not None:
        y_norm = np.linalg.norm(y, axis=0, keepdims=True)
        y_normalized = y / y_norm
        return X_normalized, y_normalized

    return X_normalized
def moving_average (
    y:ArrayLike,
    *, 
    window_size:int  = 3 , 
    method:str  ='sma',
    mode:str  ='same', 
    alpha: Union [int, float]  =.5 
)-> ArrayLike: 
    """ A moving average is  used with time series data to smooth out
    short-term fluctuations and highlight longer-term trends or cycles.
    
    Funtion analyzes data points by creating a series of averages of different
    subsets of the full data set. 
    
    Parameters 
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
        
    window_size : int
        the length of the window. Must be greater than 1 and preferably
        an odd integer number.Default is ``3``
        
    method: str 
        variant of moving-average. Can be ``sma``, ``cma``, ``wma`` and ``ema`` 
        for simple, cummulative, weight and exponential moving average. Default 
        is ``sma``. 
        
    mode: str
        returns the convolution at each point of overlap, with an output shape
        of (N+M-1,). At the end-points of the convolution, the signals do not 
        overlap completely, and boundary effects may be seen. Can be ``full``,
        ``same`` and ``valid``. See :doc:`~np.convole` for more details. Default 
        is ``same``. 
        
    alpha: float, 
        smoothing factor. Only uses in exponential moving-average. Default is 
        ``.5``.
    
    Returns 
    --------
    ya: array like, shape (N,) 
        Averaged time history of the signal
    
    Notes 
    -------
    The first element of the moving average is obtained by taking the average 
    of the initial fixed subset of the number series. Then the subset is
    modified by "shifting forward"; that is, excluding the first number of the
    series and including the next value in the subset.
    
    Examples
    --------- 
    >>> import numpy as np ; import matplotlib.pyplot as plt 
    >>> from gofast.tools.tools   import moving_average 
    >>> data = np.random.randn (37) 
    >>> # add gaussion noise to the data 
    >>> data = 2 * np.sin( data)  + np.random.normal (0, 1 , len(data))
    >>> window = 5  # fixed size to 5 
    >>> sma = moving_average(data, window) 
    >>> cma = moving_average(data, window, method ='cma' )
    >>> wma = moving_average(data, window, method ='wma' )
    >>> ema = moving_average(data, window, method ='ema' , alpha =0.6)
    >>> x = np.arange(len(data))
    >>> plt.plot (x, data, 'o', x, sma , 'ok--', x, cma, 'g-.', x, wma, 'b:')
    >>> plt.legend (['data', 'sma', 'cma', 'wma'])
    
    References 
    ----------
    .. * [1] https://en.wikipedia.org/wiki/Moving_average
    .. * [2] https://www.sciencedirect.com/topics/engineering/hanning-window
    .. * [3] https://stackoverflow.com/questions/12816011/weighted-moving-average-with-numpy-convolve
    
    """
    y = np.array(y)
    try:
        window_size = np.abs(_assert_all_types(int(window_size), int))
    except ValueError:
        raise ValueError("window_size has to be of type int")
    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if  window_size > len(y):
        raise TypeError("window_size is too large for averaging. Window"
                        f" must be greater than 0 and less than {len(y)}")
    
    method =str(method).lower().strip().replace ('-', ' ') 
    
    if method in ('simple moving average',
                  'simple', 'sma'): 
        method = 'sma' 
    elif method  in ('cumulative average', 
                     'cumulative', 'cma'): 
        method ='cma' 
    elif method  in ('weighted moving average',
                     'weight', 'wma'): 
        method = 'wma'
    elif method in('exponential moving average',
                   'exponential', 'ema'):
        method = 'ema'
    else : 
        raise ValueError ("Variant average methods only includes "
                          f" {smart_format(['sma', 'cma', 'wma', 'ema'], 'or')}")
    if  1. <= alpha <= 0 : 
        raise ValueError ('alpha should be less than 1. and greater than 0. ')
 
    if method =='cma': 
        y = np.cumsum (y) 
        ya = np.array([ y[ii]/ len(y[:ii +1]) for ii in range(len(y))]) 
        
    elif method =='wma': 
        w = np.cumsum(np.ones(window_size, dtype = float))
        w /= np.sum(w)
        ya = np.convolve(y, w[::-1], mode ) #/window_size
        
    elif method =='ema': 
        ya = np.array ([y[0]]) 
        for ii in range(1, len(y)): 
            v = y[ii] * alpha + ( 1- alpha ) * ya[-1]
            ya = np.append(ya, v)
    else:
        ya = np.convolve(y , np.ones (window_size), mode ) / window_size 
            
    return ya 


def count_local_minima(arr,  method='robust', order=1):
    """
    Count the number of local minima in a 1D array.
    
    Parameters
    ----------
    arr : array_like
        Input array.
    method : {'base', 'robust'}, optional
        Method to use for finding local minima. 
        'base' uses a simple comparison, while 'robust' 
        uses scipy's argrelextrema function. Default is 'robust'.
    order : int, optional
        How many points on each side to use for the comparison 
        to consider a point as a local minimum (only used with 
        'robust' method). Default is 1.
    
    Returns
    -------
    int
        Number of local minima in the array.
    
    Examples
    --------
    >>> from gofast.tools.mathex import count_local_minima
    >>> arr = [1, 3, 2, 4, 1, 0, 2, 1]
    >>> count_local_minima(arr, method='base')
    2
    >>> count_local_minima(arr, method='robust')
    3
    
    Notes
    -----
    - The 'base' method compares each element with its immediate 
      neighbors, which might be less accurate for noisy data.
    - The 'robust' method uses scipy's argrelextrema function, which 
      allows for more flexible and accurate detection of local minima 
      by considering a specified number of neighboring points.
    """
    arr= check_y (arr , y_numeric =True, estimator="Array" )
    method= parameter_validator(
        "method", target_strs= {"base", "robust"})(method)
    if method == 'base':
        local_minima_count = 0
        # Iterate through the array from the second element 
        # to the second-to-last element
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
                local_minima_count += 1
        return local_minima_count
    elif method == 'robust':
        # Find indices of local minima
        local_minima_indices = argrelextrema(np.array(arr), 
                                             np.less, 
                                             order=order)[0]
        # The number of local minima is the length of the indices array
        return len(local_minima_indices)
    
def get_azimuth (
    xlon: Union [str, ArrayLike], 
    ylat:Union[ str, ArrayLike], 
    *, 
    data: DataFrame =None, 
    utm_zone:str=None, 
    projection:str='ll', 
    isdeg:bool=True, 
    mode:str='soft', 
    extrapolate:bool =...,
    view:bool=..., 
    ): 
    """Compute azimuth from coordinate locations ( latitude,  longitude). 
    
    If `easting` and `northing` are given rather than `longitude` and  
    `latitude`, the projection should explicitely set to ``UTM`` to perform 
    the ideal conversion. However if mode is set to `soft` (default), the type
    of projection is automatically detected . Note that when UTM coordinates 
    are provided, `xlon` and `ylat` fit ``easting`` and ``northing`` 
    respectively.
    
    Parameters
    -----------
    xlon, ylat : Arraylike 1d or str, str 
       ArrayLike of easting/longitude and arraylike of nothing/latitude. They 
       should be one dimensional. In principle if data is supplied, they must 
       be series.  If `xlon` and `ylat` are given as string values, the 
       `data` must be supplied. xlon and ylat names must be included in the  
       dataframe otherwise an error raises. 
       
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       when `mode='soft'`,  the auto-detection will be triggered and find the 
       suitable coordinate system. However, it is recommended to explicitly 
       provide projection when data is in UTM coordinates. 
       Note that if `x` and `y` are composed of value greater than 180 degrees 
       for longitude and 90 degrees for latitude, and method is still in 
       the ``soft` mode, it should be considered as  longitude-latitude ``UTM``
       coordinates system. 
       
    isdeg: bool, default=True 
      By default xlon and xlat are in degree coordinates. If both arguments 
      are given in radians, set to ``False`` instead. 
      
    mode: str , ['soft'|'strict']
      ``strict`` mode does not convert any coordinates system to other at least
      it is explicitly set to `projection` whereas the `soft` does.
      
    extrapolate: bool, default=False 
      In principle, the azimuth is compute between two points. Thus, the number
      of values computed for :math:`N` stations should  be  :math:`N-1`. To fit
      values to match the number of size of the array, `extrapolate` should be 
      ``True``. In that case, the first station holds a <<fake>> azimuth as 
      the closer value computed from interpolation of all azimuths. 
      
    view: bool, default=False, 
       Quick view of the azimuth. It is usefull especially when 
       extrapolate is set to ``True``. 
       
    Return 
    --------
    azim: ArrayLike 
       Azimuth computed from locations. 
       
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.mathex  import get_azimuth 
    >>> # generate a data from ERP 
    >>> data = gf.make_erp (n_stations =7 ).frame 
    >>> get_azimuth ( data.longitude, data.latitude)
    array([54.575, 54.575, 54.575, 54.575, 54.575, 54.575])
    >>> get_azimuth ( data.longitude, data.latitude, view =True, extrapolate=True)
    array([54.57500007, 54.575     , 54.575     , 54.575     , 54.575     ,
           54.575     , 54.575     ])
    
    """
    from ..site import Location 
    
    mode = str(mode).lower() 
    projection= str(projection).lower()
    extrapolate, view = ellipsis2false (extrapolate, view)

    xlon , ylat = assert_xy_in(xlon , ylat , data = data )
    
    if ( 
            xlon.max() > 180.  and ylat.max() > 90.  
            and projection=='ll' 
            and mode=='soft'
            ): 
        warnings.warn("xlon and ylat arguments are greater than 180 degrees."
                     " we assume the coordinates are UTM. Set explicitly"
                     " projection to ``UTM`` to avoid this warning.")
        projection='utm'
        
    if projection=='utm':
        if utm_zone is None: 
            raise TypeError ("utm_zone cannot be None when projection is UTM.")
            
        ylat , xlon = Location.to_latlon_in(
            xlon, ylat, utm_zone= utm_zone)
        
    if len(xlon) ==1 or len(ylat)==1: 
        msg = "Azimuth computation expects at least two points. Got 1"
        if mode=='soft': 
            warnings.warn(msg) 
            return 0. 
        
        raise TypeError(msg )
    # convert to radian 
    if isdeg: 
        xlon = np.deg2rad (xlon ) ; ylat = np.deg2rad ( ylat)
    
    dx = map (lambda ii: np.cos ( ylat[ii]) * np.sin( ylat [ii+1 ]) - 
        np.sin(ylat[ii]) * np.cos( ylat[ii+1]) * np.cos (xlon[ii+1]- xlon[ii]), 
        range (len(xlon)-1)
        )
    dy = map( lambda ii: np.cos (ylat[ii+1])* np.sin( xlon[ii+1]- xlon[ii]), 
                   range ( len(xlon)-1)
                   )
    # to deg 
    z = np.around ( np.rad2deg ( np.arctan2(list(dx) , list(dy) ) ), 3)  
    azim = z.copy() 
    if extrapolate: 
        # use mean azimum of the total area zone and 
        # recompute the position by interpolation 
        azim = np.hstack ( ( [z.mean(), z ]))
        # reset the interpolare value at the first position
        with warnings.catch_warnings():
            #warnings.filterwarnings(action='ignore', category=OptimizeWarning)
            warnings.simplefilter("ignore")
            azim [0] = scalePosition(azim )[0][0] 
        
    if view: 
        x = np.arange ( len(azim )) 
        fig,  ax = plt.subplots (1, 1, figsize = (10, 4))
        # add Nan to the first position of z 
        z = np.hstack (([np.nan], z )) if extrapolate else z 
       
        ax.plot (x, 
                 azim, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='extra-azimuth'
                 ) 
        
        ax.plot (x, 
                z, 
                'ok-', 
                label ='raw azimuth'
                )
        ax.legend ( ) 
        ax.set_xlabel ('x')
        ax.set_ylabel ('y') 

    return azim

def linkage_matrix(
    df: DataFrame ,
    columns:List[str] =None,  
    kind:str ='design', 
    metric:str ='euclidean',   
    method:str ='complete', 
    as_frame =False,
    optimal_ordering=False, 
 )->NDArray: 
    r""" Compute the distance matrix from the hierachical clustering algorithm.
    
    Parameters 
    ------------ 
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions.
        
    optimal_ordering : bool, optional
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal. This results in a more intuitive
        tree structure when the data are visualized. defaults to False, because
        this algorithm can be slow, particularly on large datasets. See
        also :func:`scipy.cluster.hierarchy.linkage`. 
        
        
    Returns 
    --------
    row_clusters: linkage matrix 
        consist of several rows where each rw represents one merge. The first 
        and second columns denotes the most dissimilar members of each cluster 
        and the third columns reports the distance between those members 
        
        
    Linkage Methods 
    -----------------
    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

    * method='single' assigns

      .. math::
         d(u,v) = \min(dist(u[i],v[j]))

      for all points :math:`i` in cluster :math:`u` and
      :math:`j` in cluster :math:`v`. This is also known as the
      Nearest Point Algorithm.

    * method='complete' assigns

      .. math::
         d(u, v) = \max(dist(u[i],v[j]))

      for all points :math:`i` in cluster u and :math:`j` in
      cluster :math:`v`. This is also known by the Farthest Point
      Algorithm or Voor Hees Algorithm.

    * method='average' assigns

      .. math::
         d(u,v) = \sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

      for all points :math:`i` and :math:`j` where :math:`|u|`
      and :math:`|v|` are the cardinalities of clusters :math:`u`
      and :math:`v`, respectively. This is also called the UPGMA
      algorithm.

    * method='weighted' assigns

      .. math::
         d(u,v) = (dist(s,v) + dist(t,v))/2

      where cluster u was formed with cluster s and t and v
      is a remaining cluster in the forest (also called WPGMA).

    * method='centroid' assigns

      .. math::
         dist(s,t) = ||c_s-c_t||_2

      where :math:`c_s` and :math:`c_t` are the centroids of
      clusters :math:`s` and :math:`t`, respectively. When two
      clusters :math:`s` and :math:`t` are combined into a new
      cluster :math:`u`, the new centroid is computed over all the
      original objects in clusters :math:`s` and :math:`t`. The
      distance then becomes the Euclidean distance between the
      centroid of :math:`u` and the centroid of a remaining cluster
      :math:`v` in the forest. This is also known as the UPGMC
      algorithm.

    * method='median' assigns :math:`d(s,t)` like the ``centroid``
      method. When two clusters :math:`s` and :math:`t` are combined
      into a new cluster :math:`u`, the average of centroids s and t
      give the new centroid :math:`u`. This is also known as the
      WPGMC algorithm.

    * method='ward' uses the Ward variance minimization algorithm.
      The new entry :math:`d(u,v)` is computed as follows,

      .. math::

         d(u,v) = \sqrt{\frac{|v|+|s|}{_T}d(v,s)^2 \\
                      + \frac{|v|+|t|}{_T}d(v,t)^2 \\
                      - \frac{|v|}{_T}d(s,t)^2}

      where :math:`u` is the newly joined cluster consisting of
      clusters :math:`s` and :math:`t`, :math:`v` is an unused
      cluster in the forest, :math:`_T=|v|+|s|+|t|`, and
      :math:`|*|` is the cardinality of its argument. This is also
      known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    implementation may choose a different minimum than the MATLAB
    version.
    
    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
           algorithms", :arXiv:`1109.2378v1`.
    .. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal
           leaf ordering for hierarchical clustering", 2001. Bioinformatics
           :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    """
    df = _assert_all_types(df, pd.DataFrame, np.ndarray)
    
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("Number of columns must fit the shape of X."
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df.values if hasattr(df, 'columns') else df ,
                          columns = columns )
        
    kind= str(kind).lower().strip() 
    if kind not in ('squareform', 'condense', 'design'): 
        raise ValueError(f"Unknown method {method!r}. Expect 'squareform',"
                         " 'condense' or 'design'.")
        
    labels = [f'ID_{i}' for i in range(len(df))]
    if kind =='squareform': 
        row_dist = pd.DataFrame (squareform ( 
        pdist(df, metric= metric )), columns = labels  , 
        index = labels)
        row_clusters = linkage (row_dist, method =method, metric =metric
                                )
    if kind =='condens': 
        row_clusters = linkage (pdist(df, metric =metric), method =method
                                )
    if kind =='design': 
        row_clusters = linkage(df.values if hasattr (df, 'columns') else df, 
                               method = method, 
                               optimal_ordering=optimal_ordering )
        
    if as_frame: 
        row_clusters = pd.DataFrame ( row_clusters, 
                                     columns = [ 'row label 1', 
                                                'row lable 2', 
                                                'distance', 
                                                'no. of items in clust.'
                                                ], 
                                     index = ['cluster %d' % (i +1) for i in 
                                              range(row_clusters.shape[0])
                                              ]
                                     )
    return row_clusters 

@AppendDocReferences(refglossary.__doc__)
def scalePosition(
        ydata: Union[ArrayLike, _SP, Series, DataFrame],
        xdata: Union[ArrayLike, Series] = None, 
        func: Optional[_F] = None,
        c_order: Optional[Union[int, str]] = 0,
        show: bool = False, 
        **kws):
    """ Correct data location or position and return new corrected location 
    
    Parameters 
    ----------
    ydata: array_like, series or dataframe
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
        
    xdata: array_like or object
        The independent variable where the data is measured. Should usually 
        be an M-length sequence or an (k,M)-shaped array for functions with
        k predictors, but can actually be any object. If ``None``, `xdata` is 
        generated by default using the length of the given `ydata`.
        
    func: callable 
        The model function, ``f(x, ...)``. It must take the independent variable 
        as the first argument and the parameters to fit as separate remaining
        arguments. The default `func` is ``linear`` function i.e  for ``f(x)= ax +b``. 
        where `a` is slope and `b` is the intercept value. Setting your own 
        function for better fitting is recommended. 
        
    c_order: int or str
        The index or the column name if ``ydata`` is given as a dataframe to 
        select the right column for scaling.
    show: bool 
        Quick visualization of data distribution. 

    kws: dict 
        Additional keyword argument from  `scipy.optimize_curvefit` parameters. 
        Refer to `scipy.optimize.curve_fit`_.  
        
    Returns 
    --------
    - ydata - array -like - Data scaled 
    - popt - array-like Optimal values for the parameters so that the sum of 
        the squared residuals of ``f(xdata, *popt) - ydata`` is minimized.
    - pcov - array like The estimated covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard deviation 
        errors on the parameters use ``perr = np.sqrt(np.diag(pcov))``. How the
        sigma parameter affects the estimated covariance depends on absolute_sigma 
        argument, as described above. If the Jacobian matrix at the solution
        doesn’t have a full rank, then ‘lm’ method returns a matrix filled with
        np.inf, on the other hand 'trf' and 'dogbox' methods use Moore-Penrose
        pseudoinverse to compute the covariance matrix.
        
    Examples
    --------
    >>> from gofast.tools import erpSelector, scalePosition 
    >>> df = erpSelector('data/erp/l10_gbalo.xlsx') 
    >>> df.columns 
    ... Index(['station', 'resistivity', 'longitude', 'latitude', 'easting',
           'northing'],
          dtype='object')
    >>> # correcting northing coordinates from easting data 
    >>> northing_corrected, popt, pcov = scalePosition(ydata =df.northing , 
                                               xdata = df.easting, show=True)
    >>> len(df.northing.values) , len(northing_corrected)
    ... (20, 20)
    >>> popt  # by default popt =(slope:a ,intercept: b)
    ...  array([1.01151734e+00, 2.93731377e+05])
    >>> # corrected easting coordinates using the default x.
    >>> easting_corrected, *_= scalePosition(ydata =df.easting , show=True)
    >>> df.easting.values 
    ... array([790284, 790281, 790277, 790270, 790265, 790260, 790254, 790248,
    ...       790243, 790237, 790231, 790224, 790218, 790211, 790206, 790200,
    ...       790194, 790187, 790181, 790175], dtype=int64)
    >>> easting_corrected
    ... array([790288.18571705, 790282.30300999, 790276.42030293, 790270.53759587,
    ...       790264.6548888 , 790258.77218174, 790252.88947468, 790247.00676762,
    ...       790241.12406056, 790235.2413535 , 790229.35864644, 790223.47593938,
    ...       790217.59323232, 790211.71052526, 790205.8278182 , 790199.94511114,
    ...       790194.06240407, 790188.17969701, 790182.29698995, 790176.41428289])
    
    """
    def linfunc (x, a, b): 
        """ Set the simple linear function"""
        return a * x + b 
        
    if str(func).lower() in ('none' , 'linear'): 
        func = linfunc 
    elif not hasattr(func, '__call__') or not inspect.isfunction (func): 
        raise TypeError(
            f'`func` argument is a callable not {type(func).__name__!r}')
        
    ydata = _assert_all_types(ydata, list, tuple, np.ndarray,
                              pd.Series, pd.DataFrame  )
    c_order = _assert_all_types(c_order, int, float, str)
    try : c_order = int(c_order) 
    except: pass 

    if isinstance(ydata, pd.DataFrame): 
        if c_order ==0: 
            warnings.warn("The first column of the data should be considered"
                          " as the `y` target.")
        if c_order is None: 
            raise TypeError('Dataframe is given. The `c_order` argument should '
                            'be defined for column selection. Use column name'
                            ' instead')
        if isinstance(c_order, str): 
            # check whether the value is on the column name
            if c_order.lower() not in list(map( 
                    lambda x :x.lower(), ydata.columns)): 
                raise ValueError (
                    f'c_order {c_order!r} not found in {list(ydata.columns)}'
                    ' Use the index instead.')
                # if c_order exists find the index and get the 
                # right column name 
            ix_c = list(map( lambda x :x.lower(), ydata.columns)
                        ).index(c_order.lower())
            ydata = ydata.iloc [:, ix_c] # series 
        elif isinstance (c_order, (int, float)): 
            c_order =int(c_order) 
            if c_order >= len(ydata.columns): 
                raise ValueError(
                    f"`c_order`'{c_order}' should be less than the number of " 
                    f"given columns '{len(ydata.columns)}'. Use column name instead.")
            ydata= ydata.iloc[:, c_order]
                  
    ydata = check_y (np.array(ydata)  , input_name= "ydata")
    
    if xdata is None: 
        xdata = np.linspace(0, 4, len(ydata))
        
    xdata = check_y (xdata , input_name= "Xdata")
    
    if len(xdata) != len(ydata): 
        raise ValueError(" `x` and `y` arrays must have the same length."
                        "'{len(xdata)}' and '{len(ydata)}' are given.")
        
    popt, pcov = curve_fit(func, xdata, ydata, **kws)
    ydata_new = func(xdata, *popt)
    
    if show:
        plt.plot(xdata, ydata, 'b-', label='data')
        plt.plot(xdata, func(xdata, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        
    return ydata_new, popt, pcov 

def detect_station_position (
        s : Union[str, int] ,
        p: _SP, 
) -> Tuple [int, float]: 
    """ Detect station position and return the index in positions
    
    :param s: str, int - Station location  in the position array. It should 
        be the positionning of the drilling location. If the value given
        is type string. It should be match the exact position to 
        locate the drilling. Otherwise, if the value given is in float or 
        integer type, it should be match the index of the position array. 
         
    :param p: Array-like - Should be the  conductive zone as array of 
        station location values. 
            
    :returns: 
        - `s_index`- the position index location in the conductive zone.  
        - `s`- the station position in distance. 
        
    :Example: 
        
        >>> import numpy as np 
        >>> from gofast.tools.mathex  import detect_station_position 
        >>> pos = np.arange(0 , 50 , 10 )
        >>> detect_station_position (s ='S30', p = pos)
        ... (3, 30.0)
        >>> detect_station_position (s ='40', p = pos)
        ... (4, 40.0)
        >>> detect_station_position (s =2, p = pos)
        ... (2, 20)
        >>> detect_station_position (s ='sta200', p = pos)
        ... WATexError_station: Station sta200 \
            is out of the range; max position = 40
    """
    s = _assert_all_types( s, float, int, str)
    
    p = check_y (p, input_name ="Position array 'p'", to_frame =True )
    
    S=copy.deepcopy(s)
    if isinstance(s, str): 
        s =s.lower().replace('s', '').replace('pk', '').replace('ta', '')
        try : 
            s=int(s)
        except : 
            raise ValueError (f'could not convert string to float: {S}')
            
    p = np.array(p, dtype = np.int32)
    dl = (p.max() - p.min() ) / (len(p) -1) 
    if isinstance(s, (int, float)): 
        if s > len(p): # consider this as the dipole length position: 
            # now let check whether the given value is module of the station 
            if s % dl !=0 : 
                raise SiteError  (
                    f'Unable to detect the station position {S}')
            elif s % dl == 0 and s <= p.max(): 
                # take the index 
                s_index = s//dl
                return int(s_index), s_index * dl 
            else : 
                raise SiteError (
                    f'Station {S} is out of the range; max position = {max(p)}'
                )
        else : 
            if s >= len(p): 
                raise SiteError (
                    'Location index must be less than the number of'
                    f' stations = {len(p)}. {s} is gotten.')
            # consider it as integer index 
            # erase the last variable
            # s_index = s 
            # s = S * dl   # find 
            return s , p[s ]
       
    # check whether the s value is in the p 
    if True in np.isin (p, s): 
        s_index ,  = np.where (p ==s ) 
        s = p [s_index]
        
    return int(s_index) , s 
    
def _manage_colors (c, default = ['ok', 'ob-', 'r-']): 
    """ Manage the ohmic-area plot colors """
    c = c or default 
    if isinstance(c, str): 
        c= [c] 
    c = list(c) +  default 
    
    return c [:3] # return 3colors 
     


def convert_distance_to_m(
        value:_T ,
        converter:float =1e3,
        unit:str ='km'
)-> float: 
    """ Convert distance from `km` to `m` or vice versa even a string 
    value is given.
    
    :param value: value to convert. 
    :paramm converter: Equivalent if given in ``km`` rather than ``m``.
    :param unit: unit to convert to.
    
    """
    
    if isinstance(value, str): 
        try:
            value = float(value.replace(unit, '')
                              )*converter if value.find(
                'km')>=0 else float(value.replace('m', ''))
        except: 
            raise TypeError(f"Expected float not {type(value)!r}."
               )
            
    return value
       
def get_station_number (
        dipole:float,
        distance:float , 
        from0:bool = False,
        **kws
)-> float: 
    """ Get the station number from dipole length and 
    the distance to the station.
    
    :param distance: Is the distance from the first station to `s` in 
        meter (m). If value is given, please specify the dipole length in 
        the same unit as `distance`.
    :param dipole: Is the distance of the dipole measurement. 
        By default the dipole length is in meter.
    :param kws: :func:`convert_distance_to_m` additional arguments
    
    """
    dipole=convert_distance_to_m(dipole, **kws)
    distance =convert_distance_to_m(distance, **kws)

    return  distance/dipole  if from0 else distance/dipole + 1 
    
#FR0: #CED9EF # (206, 217, 239)
#FR1: #9EB3DD # (158, 179, 221)
#FR2: #3B70F2 # (59, 112, 242) #repl rgb(52, 54, 99)
#FR3: #0A4CEE # (10, 76, 238)

def get_profile_angle (
        easting: float =None, northing: float =None, msg:str ="ignore" ): 
    """
    compute geoprofile angle. 
    Parameters 
    -----------
    * easting : array_like 
            easting coordiantes values 
    * northing : array_like 
            northing coordinates values
    * msg: output a little message if msg is set to "raises"
    
    Returns 
    ---------
    float
         profile_angle 
    float 
        geo_electric_strike 
    """
    msg = (
        "Need to import scipy.stats as a single module. Sometimes import scipy "
        "differently  with stats may not work. Use either `import scipy.stats`"
        " rather than `import scipy as sp`" 
        )
    
    if easting is None or northing is None : 
        raise TypeError('NoneType can not be computed !')
        
        # use the one with the lower standard deviation
    try :
        easting = easting.astype('float')
        northing = northing.astype('float')
    except : 
        raise ValueError('Could not convert input argument to float!')
    try : 
        profile1 = spstats.linregress(easting, northing)
        profile2 =spstats.linregress(northing, easting)
    except:
        warnings.warn(msg)
        
    profile_line = profile1[:2]
    # if the profile is rather E=E(N),
    # the parameters have to converted  into N=N(E) form:
    
    if profile2[4] < profile1[4]:
        profile_line = (1. / profile2[0], -profile2[1] / profile2[0])

    # if self.profile_angle is None:
    profile_angle = (90 - (np.arctan(profile_line[0]) * 180 / np.pi)) % 180
    
    # otherwise: # have 90 degree ambiguity in 
    #strike determination# choose strike which offers larger
    #  angle with profile if profile azimuth is in [0,90].
    if msg=="raises": 
        print("+++ -> Profile angle is {0:+.2f} degrees E of N".format(
                profile_angle
                ) )
    return np.around( profile_angle,2)

        
def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None,
                  use="conv"):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be an odd positive integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.

    See Also
    --------
    savgol_filter

    Examples
    --------
    >>> from gofast.exmath.signal import savgol_coeffs
    >>> savgol_coeffs(5, 2)
    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
    >>> savgol_coeffs(5, 2, deriv=1)
    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,
           -2.00000000e-01])

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.

    >>> x = np.array([1, 0, 1, 4, 9])
    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    >>> c.dot(x)
    6.0
    """

    # An alternative method for finding the coefficients when deriv=0 is
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # The method implemented here is faster.

    # To recreate the table of sample coefficients shown in the chapter on
    # the Savitzy-Golay filter in the Numerical Recipes book, use
    #    window_length = nL + nR + 1
    #    pos = nL + 1
    #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if rem == 0:
        raise ValueError("window_length must be odd.")

    if pos is None:
        pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")

    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)
    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    # The coefficient assigned to y[deriv] scales the result to take into
    # account the order of the derivative and the sample spacing.
    y[deriv] = float_factorial(deriv) / (delta ** deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs


def _polyder(p, m):
    """Differentiate polynomials represented with coefficients.

    p must be a 1-D or 2-D array.  In the 2-D case, each column gives
    the coefficients of a polynomial; the first row holds the coefficients
    associated with the highest power. m must be a nonnegative integer.
    (numpy.polyder doesn't handle the 2-D case.)
    """

    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = np.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = np.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


def _fit_edge(x, window_start, window_stop, interp_start, interp_stop,
              axis, polyorder, deriv, delta, y):
    """
    Given an N-d array `x` and the specification of a slice of `x` from
    `window_start` to `window_stop` along `axis`, create an interpolating
    polynomial of each 1-D slice, and evaluate that polynomial in the slice
    from `interp_start` to `interp_stop`. Put the result into the
    corresponding slice of `y`.
    """

    # Get the edge into a (window_length, -1) array.
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)

    # Fit the edges.  poly_coeffs has shape (polyorder + 1, -1),
    # where '-1' is the same as in xx_edge.
    poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
                             xx_edge, polyorder)

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)

    # Compute the interpolated values for the edge.
    i = np.arange(interp_start - window_start, interp_stop - window_start)
    values = np.polyval(poly_coeffs, i.reshape(-1, 1)) / (delta ** deriv)

    # Now put the values into the appropriate slice of y.
    # First reshape values to match y.
    shp = list(y.shape)
    shp[0], shp[axis] = shp[axis], shp[0]
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    if swapped:
        values = values.swapaxes(0, axis)
    # Get a view of the data to be replaced by values.
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values

def _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y):
    """
    Use polynomial interpolation of x at the low and high ends of the axis
    to fill in the halflen values in y.

    This function just calls _fit_edge twice, once for each end of the axis.
    """
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis,
              polyorder, deriv, delta, y)
    n = x.shape[axis]
    _fit_edge(x, n - window_length, n, n - halflen, n, axis,
              polyorder, deriv, delta, y)

def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    See Also
    --------
    savgol_coeffs

    Notes
    -----
    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from gofast.tools.mathex  import savgol_filter
    >>> np.set_printoptions(precision=2)  # For compact display.
    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

    Filter with a window length of 5 and a degree 2 polynomial.  Use
    the defaults for all other parameters.

    >>> savgol_filter(x, 5, 2)
    array([1.66, 3.17, 3.54, 2.86, 0.66, 0.17, 1.  , 4.  , 9.  ])

    Note that the last five values in x are samples of a parabola, so
    when mode='interp' (the default) is used with polyorder=2, the last
    three values are unchanged. Compare that to, for example,
    `mode='nearest'`:

    >>> savgol_filter(x, 5, 2, mode='nearest')
    array([1.74, 3.03, 3.54, 2.86, 0.66, 0.17, 1.  , 4.6 , 7.97])

    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.size:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad. Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y        

def compute_errors (
    arr,  
    error ='std', 
    axis = 0, 
    return_confidence=False 
    ): 
    """ Compute Errors ( Standard Deviation ) and standard errors. 
    
    Standard error and standard deviation are both measures of variability:
    - The standard deviation describes variability within a single sample. Its
      formula is given as: 
          
      .. math:: 
          
          SD = \sqrt{ \sum |x -\mu|^2}{N}
          
      where :math:`\sum` means the "sum of", :math:`x` is the value in the data 
      set,:math:`\mu` is the mean of the data set and :math:`N` is the number 
      of the data points in the population. :math:`SD` is the quantity 
      expressing by how much the members of a group differ from the mean 
      value for the group.
      
    - The standard error estimates the variability across multiple 
      samples of a population. Different formulas are used depending on 
      whether the population standard deviation is known.
      
      - when the population standard deviation is known: 
      
        .. math:: 
          
            SE = \frac{SD}{\sqrt{N}} 
            
      - When the population parameter is unknwon 
      
        .. math:: 
            
            SE = \frac{s}{\sqrt{N}} 
            
       where :math:`SE` is the standard error, : math:`s` is the sample
       standard deviation. When the population standard is knwon the 
       :math:`SE` is more accurate. 
    
    Note that the :math:`SD` is  a descriptive statistic that can be 
    calculated from sample data. In contrast, the standard error is an 
    inferential statistic that can only be estimated 
    (unless the real population parameter is known). 
    
    Parameters
    ----------
    arr : array_like , 1D or 2D 
      Array for computing the standard deviation 
      
    error: str, default='std'
      Name of error to compute. By default compute the standard deviation. 
      Can also compute the the standard error estimation if the  argument 
      is passed to ``ste``. 
    return_confidence: bool, default=False, 
      If ``True``, returns the confidence interval with 95% of sample means 
      
    Returns 
    --------
    err: arraylike 1D or 2D 
       Error array. 
       
    Examples
    ---------
    >>> from gofast.tools.mathex  import compute_errors
    >>> from gofast.datasets import make_mining_ops 
    >>> mdata = make_mining_ops ( samples =20, as_frame=True, noises="20%", return_X_y=False)
    >>> compute_errors (mdata)
    Easting_m                    301.216454
    Northing_m                   301.284073
    Depth_m                      145.343063
    OreConcentration_Percent       5.908375
    DrillDiameter_mm              50.019249
    BlastHoleDepth_m               3.568771
    ExplosiveAmount_kg           142.908481
    EquipmentAge_years             4.537603
    DailyProduction_tonnes      2464.819019
    dtype: float64
    >>> compute_errors ( mdata, return_confidence= True)
    (Easting_m                   -100.015509
     Northing_m                  -181.088446
     Depth_m                      -67.948155
     OreConcentration_Percent      -3.316211
     DrillDiameter_mm              25.820805
     BlastHoleDepth_m               1.733541
     ExplosiveAmount_kg             3.505198
     EquipmentAge_years            -1.581202
     DailyProduction_tonnes      1058.839261
     dtype: float64,
     Easting_m                    1080.752992
     Northing_m                    999.945119
     Depth_m                       501.796651
     OreConcentration_Percent       19.844618
     DrillDiameter_mm              221.896260
     BlastHoleDepth_m               15.723123
     ExplosiveAmount_kg            563.706443
     EquipmentAge_years             16.206202
     DailyProduction_tonnes      10720.929814
     dtype: float64)
    """
    error = _validate_name_in(error , defaults =('error', 'se'),
                              deep =True, expect_name ='se')
    # keep only the numeric values.
    if hasattr (arr, '__array__') and hasattr(arr, 'columns'): 
        arr = to_numeric_dtypes ( arr, pop_cat_features =True )
        
    if not _is_numeric_dtype(arr): 
        raise TypeError("Numeric array is expected for operations.")
    err= np.std (arr) if arr.ndim ==1 else np.std (arr, axis= axis )
                  
    err_lower =  err_upper = None 
    if error =='se': 
        N = len(arr) if arr.ndim ==1 else arr.shape [axis ]
        err =  err / np.sqrt(N)
    if return_confidence: 
        err_lower = arr.mean() - ( 1.96 * err ) 
        err_upper = arr.mean() + ( 1.96 * err )
    return err if not return_confidence else ( err_lower, err_upper)  


def quality_control2(
    ar, 
    tol: float= .5 , 
    return_data=False,
    to_log10: bool =False, 
    return_qco:bool=False 
    )->Tuple[float, ArrayLike]: 
    """
    Check the quality control in the collection of Z or EDI objects. 
    
    Analyse the data in the EDI collection and return the quality control value.
    It indicates how percentage are the data to be representative.
   
    Parameters 
    ----------
    
    ar: Arraylike of (m_samples, n_features)
       Arraylike  two dimensional data.
        
    tol: float, default=.5 
        the tolerance parameter. The value indicates the rate from which the 
        data can be consider as meaningful. Preferably it should be less than
        1 and greater than 0.  Default is ``.5`` means 50 %. Analysis becomes 
        soft with higher `tol` values and severe otherwise. 
        
    return_data: bool, default= False, 
        returns the valid data from up to ``1-tol%`` goodness. 
        
    return qco: bool, default=False, 
       retuns quality control object that wraps all usefull informations after 
       control. The following attributes can be fetched as: 
           
       - rate_: the rate of the quality of the data  
       - component_: The selected component where data is selected for analysis 
         By default used either ``xy`` or ``yx``. 
       - mode_: The :term:`EM` mode. Either the ['TE'|'TM'] modes 
       - freqs_: The valid frequency in the data selected according to the 
         `tol` parameters. Note that if ``interpolate_freq`` is ``True``, it 
         is used instead. 
       - invalid_freqs_: Useless frequency dropped in the data during control 
       - data_: Valid tensor data either in TE or TM mode. 
       
    Returns 
    -------
    Tuple (float  )  or (float, array-like, shape (N, )) or QCo
        - return the quality control value and interpolated frequency if  
         `return_freq`  is set to ``True`` otherwise return the
         only the quality control ratio.
        - return the the quality control object. 
        
    Examples 
    -----------
    >>> import gofast as gf 
    >>> data = gf.fetch_data ('huayuan', samples =20, return_data =True ,
                              key='raw')
    >>> r,= gf.qc (data)
    r
    Out[61]: 0.75
    >>> r, = gf.qc (data, tol=.2 )
    0.75
    >>> r, = gf.qc (data, tol=.1 )
    
    """
    tol = assert_ratio(tol , bounds =(0, 1), exclude_value ='use lower bound',
                         name ='tolerance', in_percent =True )
    # by default , we used the resistivity tensor and error at TE mode.
    # force using the error when resistivity or phase tensors are supplied 
    # compute the ratio of NaN in axis =0 
    nan_sum  =np.nansum(np.isnan(ar), axis =1) 

    rr= np.around ( nan_sum / ar.shape[1] , 2) 
    # print(rr); print(nan_sum) 
    # print(rr[0])
    # print(nan_sum[rr[0]].sum())
    # compute the ratio ck
    # ck = 1. -    rr[np.nonzero(rr)[0]].sum() / (
    #     1 if len(np.nonzero(rr)[0])== 0 else len(np.nonzero(rr)[0])) 
    # ck =  (1. * len(rr) - len(rr[np.nonzero(rr)[0]]) )  / len(rr)
    
    # using np.nonzero(rr) seems deprecated 
    ck = 1 - nan_sum[np.nonzero(rr)[0]].sum() / (
        ar.shape [0] * ar.shape [1]) 
    # ck = 1 - nan_sum[rr[0]].sum() / (
    #     ar.shape [0] * ar.shape [1]) 
    # now consider dirty data where the value is higher 
    # than the tol parameter and safe otherwise. 
    index = reshape (np.argwhere (rr > tol))
    # ar_new = np.delete (rr , index , axis = 0 ) 
    # if return QCobj then block all returns  to True 
    if return_qco: 
        return_data = True 
        
    data =[ np.around (ck, 2) ] 

    if return_data :
        data += [ np.delete ( ar, index , axis =0 )] 
        
    data = tuple (data )
    # make QCO object 
    if return_qco: 
        data = KeyBox( **dict (
            tol=tol, 
            rate_= float(np.around (ck, 2)), 
            data_=  np.delete ( ar, index , axis =0 )
            )
        )
    return data
 

def get_distance(
    x: ArrayLike, 
    y:ArrayLike , *, 
    return_mean_dist:bool =False, 
    is_latlon= False , 
    **kws
    ): 
    """
    Compute distance between points.
    
    Parameters
    ------------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    return_mean_dist: bool, default =False, 
       Returns the average value of the distance between different points. 
       
    is_latlon: bool, default=False, 
        Convert `x` and `y` latitude  and longitude coordinates values 
        into UTM before computing the distance. `x`, `y` should be considered 
        as ``easting`` and ``northing`` respectively. 
        
    kws: dict, 
       Keyword arguments passed to :meth:`gofast.site.Location.to_utm_in`
       
    Returns 
    ---------
    d: Arraylike of shape (N-1) 
      Is the distance between points. 
      
    Examples 
    --------- 
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import get_distance 
    >>> x = np.random.rand (7) *10 
    >>> y = np.abs ( np.random.randn (7) * 12 ) 
    >>> get_distance (x, y) 
    array([ 8.7665511 , 12.47545656,  8.53730212, 13.54998351, 14.0419387 ,
           20.12086781])
    >>> get_distance (x, y, return_mean_dist= True) 
    12.91534996818084
    """
    x, y = _assert_x_y_positions (x, y, is_latlon , **kws  )
    d = np.sqrt( np.diff (x) **2 + np.diff (y)**2 ) 
    
    return d.mean()  if return_mean_dist else d 

def scale_positions (
    x: ArrayLike, 
    y:ArrayLike, 
    *, 
    is_latlon:bool=False, 
    step:float= None, 
    use_average_dist:bool=False, 
    utm_zone:str= None, 
    shift: bool=True, 
    view:bool = False, 
    **kws
    ): 
    """
    Correct the position coordinates. 
     
    By default, it consider `x` and `y` as easting/latitude and 
    northing/longitude coordinates respectively. It latitude and longitude 
    are given, specify the parameter `is_latlon` to ``True``. 
    
    Parameters
    ----------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    is_latlon: bool, default=False, 
       Convert `x` and `y` latitude  and longitude coordinates values 
       into UTM before computing the distance. `x`, `y` should be considered 
       as ``easting`` and ``northing`` respectively. 
           
    step: float, Optional 
       The positions separation. If not given, the average distance between 
       all positions should be used instead. 
    use_average_dist: bool, default=False, 
       Use the distance computed between positions for the correction. 
    utm_zone: str,  Optional (##N or ##S)
       UTM zone in the form of number and North or South hemisphere. For
       instance '10S' or '03N'. Note that if `x` and `y` are UTM coordinates,
       the `utm_zone` must be provide to accurately correct the positions, 
       otherwise the default value ``49R`` should be used which may lead to 
       less accuracy. 
       
    shift: bool, default=True,
       Shift the coordinates from the units of `step`. This is the default 
       behavor. If ``False``, the positions are just scaled. 
    
    view: bool, default=True 
       Visualize the scaled positions 
       
    kws: dict, 
       Keyword arguments passed to :func:`~.get_distance` 
    Returns 
    --------
    xx, yy: Arraylike 1d, 
       The arrays of position correction from `x` and `y` using the 
       bearing. 
       
    See Also 
    ---------
    gofast.tools.mathex .get_bearing: 
        Compute the  direction of one point relative to another point. 
      
    Examples
    ---------
    >>> from gofast.tools.mathex  import scale_positions 
    >>> east = [336698.731, 336714.574, 336730.305] 
    >>> north = [3143970.128, 3143957.934, 3143945.76]
    >>> east_c , north_c= scale_positions (east, north, step =20, view =True  ) 
    >>> east_c , north_c
    (array([336686.69198337, 336702.53498337, 336718.26598337]),
     array([3143986.09866306, 3143973.90466306, 3143961.73066306]))
    """
    from ..site import Location
    
    msg =("x, y are not in longitude/latitude format  while 'utm_zone' is not"
          " supplied. Correction should be less accurate. Provide the UTM"
          " zone to improve the accuracy.")
    
    if is_latlon: 
        xs , ys = np.array(copy.deepcopy(x)) , np.array(copy.deepcopy(y))

    x, y = _assert_x_y_positions( x, y, islatlon = is_latlon , **kws ) 
    
    if step is None: 
        warnings.warn("Step is not given. Average distance between points"
                      " should be used instead.")
        use_average_dist =True 
    else:  
        d = float (_assert_all_types(step, float, int , objname ='Step (m)'))
    if use_average_dist: 
        d = get_distance(x, y, return_mean_dist=use_average_dist,  **kws) 
        
    # compute bearing. 
    utm_zone = utm_zone or '49R'
    if not is_latlon and utm_zone is None: 
        warnings.warn(msg ) 
    if not is_latlon: 
        xs , ys = Location.to_latlon_in(x, y, utm_zone= utm_zone) 
  
    b = get_bearing((xs[0] , ys[0]) , (xs[-1], ys[-1]),
                    to_deg =False ) # return bearing in rad.
 
    xx = x + ( d * np.cos (b))
    yy = y +  (d * np.sin(b))
    if not shift: 
        xx, *_ = scalePosition(x )
        yy, *_ = scalePosition(y)
        
    if view: 
        state = f"{'scaled' if not shift else 'shifted'}"
        plt.plot (x, y , 'ok-', label =f"Un{state} positions") 
        plt.plot (xx , yy , 'or:', label =f"{state.title()} positions")
        plt.xlabel ('x') ; plt.ylabel ('y')
        plt.legend()
        plt.show () 
        
    return xx, yy 

def _assert_x_y_positions (x, y , islatlon = False, is_utm=True,  **kws): 
    """ Assert the position x and y and return array of x and y  """
    from ..site import Location 
    x = np.array(x, dtype = np.float64) 
    y = np.array(y, np.float64)
    for ii, ar in enumerate ([x, y]):
        if not _is_arraylike_1d(ar):
            raise TypeError (
                f"Expect one-dimensional array for {'x' if ii==0 else 'y'!r}."
                " Got {x.ndim}d.")
        if len(ar) <= 1:
            raise ValueError (f"A singleton array {'x' if ii==0 else 'y'!r} is"
                              " not admitted. Expect at least two points"
                              " A(x1, y1) and B(x2, y2)")
    if islatlon: 
        x , y = Location.to_utm_in(x, y, **kws)
    return x, y 

def get_bearing (latlon1, latlon2,  to_deg = True ): 
    """
    Calculate the bearing between two points. 
     
    A bearing can be defined as  a direction of one point relative 
    to another point, usually given as an angle measured clockwise 
    from north.
    The formula of the bearing :math:`\beta` between two points 1(lat1 , lon1)
    and 2(lat2, lon2) is expressed as below: 
        
    .. math:: 
        \beta = atan2(sin(y_2-y_1)*cos(x_2), cos(x_1)*sin(x_2) – \
                      sin(x_1)*cos(x_2)*cos(y_2-y_1))
     
    where: 
       
       - :math:`x_1`(lat1): the latitude of the first coordinate
       - :math:`y_1`(lon1): the longitude of the first coordinate
       - :math:`x_2`(lat2) : the latitude of the second coordinate
       - :math:`y_2`(lon2): the longitude of the second coordinate
    
    Parameters 
    ----------- 
    latlon: Tuple ( latitude, longitude) 
       A latitude and longitude coordinates of the first point in degree. 
    latlon2: Tuple ( latitude, longitude) 
       A latitude and longitude of coordinates of the second point in degree.  
       
    to_deg: bool, default=True 
       Convert the bearing from radians to degree. 
      
    Returns 
    ---------
    b: Value of bearing in degree ( default). 
    
    See More 
    ----------
    See more details by clicking in the link below: 
        https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
        
    Examples 
    ---------
    >>> from gofast.tools import get_bearing 
    >>> latlon1 = (28.41196763902007, 109.3328724432221) # (lat, lon) point 1
    >>> latlon2= (28.38756530909265, 109.36931920880758) # (lat, lon) point 2
    >>> get_bearing (latlon1, latlon2 )
    127.26739270447973 # in degree 
    """
    latlon1 = reshape ( np.array ( latlon1, dtype = np.float64)) 
    latlon2 = reshape ( np.array ( latlon2, dtype = np.float64)) 
    
    if len(latlon1) <2 or len(latlon2) <2 : 
        raise ValueError("Wrong coordinates values. Need two coordinates"
                         " (latitude and longitude) of points 1 and 2.")
    lat1 = np.deg2rad (latlon1[0]) ; lon1 = np.deg2rad(latlon1[1])
    lat2 = np.deg2rad (latlon2[0]) ; lon2 = np.deg2rad(latlon2[1])
    
    b = np.arctan2 (
        np.sin(lon2 - lon1 )* np.cos (lat2), 
        np.cos (lat1) * np.sin(lat2) - np.sin (lat1) * np.cos (lat2) * np.cos (lon2 - lon1)
                    )
    if to_deg: 
        # convert bearing to degree and make sure it 
        # is positive between 360 degree 
        b = ( np.rad2deg ( b) + 360 )% 360 
        
    return b 

def find_closest( arr, values ): 
    """Get the closest value in array  from given values.
    
    Parameters 
    -----------
    arr : Arraylike  
       Array to find the values 
       
    values: float, arraylike 
    
    Returns
    --------
    closest values in float or array containing in the given array.
    
    Examples
    -----------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import find_closest
    >>> find_closest (  [ 2 , 3, 4, 5] , ( 2.6 , 5.6 )  )
    array([3., 5.])
    >>> find_closest (  np.array ([[2 , 3], [ 4, 5]]), ( 2.6 , 5.6 ) )
    array([3., 5.])
    array([3., 5.])
    """

    arr = is_iterable(arr, exclude_string=True , transform =True  )
    values = is_iterable(values , exclude_string=True  , transform =True ) 
    
    for ar, v in zip ( [ arr, values ], ['array', 'values']): 
        if not _is_numeric_dtype(arr, to_array= True ) :
            raise TypeError(f"Non-numerical {v} are not allowed.")
        
    arr = np.array (arr, dtype = np.float64 )
    values = np.array (values, dtype = np.float64 ) 
    
    # ravel arr if ndim is not one-dimensional 
    arr  = arr.ravel() if arr.ndim !=1 else arr 
    # Could Find the absolute difference with each value   
    # Get the index of the smallest absolute difference. 
    
    # --> Using map is less faster than list comprehension 
    # close = np.array ( list(
    #     map (lambda v: np.abs ( arr - v).argmin(), values )
    #                   ), dtype = np.float64
    #     )
    return np.array ( [
        arr [ np.abs ( arr - v).argmin()] for v in values ]
        )
  
def gradient_descent(
    z: ArrayLike, 
    s:ArrayLike, 
    alpha:float=.01, 
    n_epochs:int= 100,
    kind:str="linear", 
    degree:int=1, 
    raise_warn:bool=False, 
    ): 
    """ Gradient descent algorithm to  fit the best model parameter.
    
    Model can be changed to polynomial if degree is greater than 1. 
    
    Parameters 
    -----------
    z: arraylike, 
       vertical nodes containing the values of depth V
    s: Arraylike, 
       vertical vector containin the resistivity values 
    alpha: float,
       step descent parameter or learning rate. *Default* is ``0.01`
    n_epochs: int, 
       number of iterations. *Default* is ``100``. Can be changed to other values
    kind: str, {"linear", "poly"}, default= 'linear'
      Type of model to fit. Linear model is selected as the default. 
    degree: int, default=1 
       As the linear model is selected as the default since the degree is set 
       to ``1``
    Returns 
    ---------
    - `_F`: New model values with the best `W` parameters found.
    - `W`: vector containing the parameters fits 
    - `cost_history`: Containing the error at each Itiretaions. 
        
    Examples 
    -----------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import gradient_descent
    >>> z= np.array([0, 6, 13, 20, 29 ,39, 49, 59, 69, 89, 109, 129, 
                     149, 179])
    >>> res= np.array( [1.59268,1.59268,2.64917,3.30592,3.76168,
                        4.09031,4.33606, 4.53951,4.71819,4.90838,
          5.01096,5.0536,5.0655,5.06767])
    >>> fz, weights, cost_history = gradient_descent(
        z=z, s=res,n_epochs=10,alpha=1e-8,degree=2)
    >>> import matplotlib.pyplot as plt 
    >>> plt.scatter (z, res)
    >>> plt.plot(z, fz)
    """
    
    #Assert degree
    try :degree= abs(int(degree)) 
    except:raise TypeError(f"Degree is integer. Got {type(degree).__name__!r}")
    
    if degree >1 :
        kind='poly'
        
    kind = str(kind).lower()    
    if kind.lower() =='linear': 
        # block degree to one.
        degree = 1 
    elif kind.find('poly')>=0 : 
        if degree <=1 :
            warnings.warn(
                "Polynomial function expects degree greater than 1."
                f" Got {degree!r}. Value is resetting to minimum equal 2."
                      ) if raise_warn else None 
            degree = 2
    # generate function with degree 
    Z, W = _kind_of_model(degree=degree,  x=z, y=s)
    
    # Compute the gradient descent 
    cost_history = np.zeros(n_epochs)
    s=s.reshape((s.shape[0], 1))
    
    for ii in range(n_epochs): 
        with np.errstate(all='ignore'): # rather than divide='warn'
            #https://numpy.org/devdocs/reference/generated/numpy.errstate.html
            W= W - (Z._T.dot(Z.dot(W)-s)/ Z.shape[0]) * alpha 
            cost_history[ii]= (1/ 2* Z.shape[0]) * np.sum((Z.dot(W) -s)**2)
       
    # Model function _F= Z.W where `Z` id composed of vertical nodes 
    # values and `bias` columns and `W` is weights numbers.
    _F= Z.dot(W) # model(Z=Z, W=W)     # generate the new model with the best weights 
             
    return _F,W, cost_history

def _kind_of_model(degree, x, y) :
    """ 
    An isolated part of gradient descent computing. 
    Generate kind of model. If degree is``1`` The linear subset 
    function will use. If `degree` is greater than 2,  Matrix will 
    generate using the polynomail function.
     
    :param x: X values must be the vertical nodes values 
    :param y: S values must be the resistivity of subblocks at node x 
    
    """
    c= []
    deg = degree 
    w = np.zeros((degree+1, 1)) # initialize weights 
    
    def init_weights (x, y): 
        """ Init weights by calculating the scope of the function along 
         the vertical nodes axis for each columns. """
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', 
                                    category=RuntimeWarning)
            for j in range(x.shape[1]-1): 
                a= (y.max()-y.min())/(x[:, j].max()-x[:, j].min())
                w[j]=a
            w[-1] = y.mean()
        return w   # return weights 

    for i in range(degree):
        c.append(x ** deg)
        deg= deg -1 

    if len(c)> 1: 
        x= concat_array_from_list(c, concat_axis=1)
        x= np.concatenate((x, np.ones((x.shape[0], 1))), axis =1)

    else: x= np.vstack((x, np.ones(x.shape)))._T # initialize z to V*2

    w= init_weights(x=x, y=y)
    return x, w  # Return the matrix x and the weights vector w 
    
def adaptive_moving_average(data,  window_size_factor=0.1):
    """ Adaptative moving average as  smoothing technique. 
 
    Parameters 
    -----------
    data: Arraylike 
       Noise data for smoothing 
       
    window_size_factor: float, default=0.1 
      Parameter to control the adaptiveness of the moving average.
       
    Return 
    --------
    result: Arraylike 
       Smoothed data 
    
    Example 
    ---------
    >>> import matplotlib.pyplot as plt
    >>> from gofast.tools.mathex  import adaptive_moving_average 
    >>> # Sample magnetotelluric data (replace this with your own data)
    >>> # Example data: a sine wave with noise
    >>> time = np.linspace(0, 10, 1000)  # Replace with your actual time values
    >>> mt_data = np.sin(2 * np.pi * 1 * time) + 0.2 * np.random.randn(1000)  # Example data
    >>> # Function to calculate the adaptive moving average
    >>> # Define the window size factor (adjust as needed)
    >>> window_size_factor = 0.1  # Adjust this value based on your data characteristics
    >>> # Apply adaptive moving average to the magnetotelluric data
    >>> smoothed_data = adaptive_moving_average(mt_data, window_size_factor)
    >>> # Plot the original and smoothed data
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(time, mt_data, 'b-', label='Original Data')
    >>> plt.plot(time, smoothed_data, 'r-', label='Smoothed Data (AMA)')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Adaptive Moving Average (AMA) Smoothing')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()
    """
    result = np.zeros_like(data)
    window_size = int(window_size_factor * len(data))
    
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        result[i] = np.mean(data[start:end])
    
    return result

def torres_verdin_filter(
    arr,  
    weight_factor: float = 0.1, 
    beta: float = 1.0, 
    logify: bool = False, 
    axis: int = None, 
    ):
    """
    Calculates the adaptive moving average of a given data array from 
    Torres and Verdin algorithm [1]_. 
    
    Parameters 
    -----------
    arr: Arraylike 1d 
      List or array-like of data points.  If two-dimensional array 
      is passed, `axis` must be specified to apply the filter onto. 
       
    weight_factor: float, default=.1
      Base smoothing factor for window size which gets adjusted by a factor 
      dependent on the rate of change in the data. 
        
    beta: float, default =1. 
       Scaling factor to adjust `weight_factor` during high volatility. 
       It controls how much the `weight_factor` is adjusted during 
       periods of high volatility.
       
    logify: bool, default=False, 
      By default , Torres uses exponential moving average. So if the 
      values can be logarithmized to ensure the weight be ranged between 
      0 and 1. This is important when data are resistivity or phase. 
      
    axis: int, default=0 
      Axis along which to apply the AMA filter.
    Return 
    -------
    ama: Adaptive moving average
    
    References 
    ------------
    .. [1] Torres-Verdin and Bostick, 1992,  Principles of spatial surface 
        electric field filtering in magnetotellurics: electromagnetic array profiling
        (EMAP), Geophysics, v57, p603-622.https://doi.org/10.1190/1.2400625

    Example
    --------
    >>> import matplotlib.pyplot as plt 
    >>> from gofast.tools.mathex  import torres_verdin_filter 
    >>> data = np.random.randn(100)  
    >>> ama = torres_verdin_filter(data)
    >>> plt.plot (range (len(data)), data, 'k', range(len(data)), ama, '-or')
    >>> # apply on two dimensional array 
    >>> data2d = np.random.randn(7, 10) 
    >>> ama2d = torres_verdin_filter ( data2d, axis =0)
    >>> fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                             figsize = (7,7) )
    >>> ax[0].imshow(data2d , label ='Raw data', cmap = 'binary' )
    >>> ax[1].imshow (ama2d,  label = 'AMA data', cmap ='binary' )
    >>> ax[0].set_title ('Raw data') 
    >>> ax[1].set_title ('AMA data') 
    >>> plt.legend
    >>> plt.show () 
    
    """
    arr = is_iterable(arr, exclude_string=True, transform=True)
    axis = 0 if axis is None else axis  # Set default axis to 0 if not specified
    logify = bool(logify)
    
    def _filtering_1d_array( ar, wf, b ): 
        if len(ar) < 2:
            return ar
        ama = [ar[0]]  # Initialize the adaptive moving average array
        for i in range(1, len(ar)):
            change = abs(ar[i] - ar[i-1])
            w = wf * (1 + beta * change)
            w = min(w, 1)  # Ensure weight stays between 0 and 1
            ama_value = w * ar[i] + (1 - w) * ama[-1]
            ama.append(ama_value)
            
        return np.array(ama)
    
    arr =np.array (arr )
    #+++++++++++++++++++
    if logify:
        arr = np.log10 ( arr )
    if arr.ndim >=2: 
        if axis is None:
            warnings.warn (f"Array dimension is {arr.ndim}. Axis must be"
                           " specified. Otherwise axis=0 is used .")
            axis =0
        if axis ==0: 
            arr = arr._T 
        for ii in range( len(arr )) : 
            arr [ii] = _filtering_1d_array (
                arr [ii ], wf = weight_factor, b = beta ) 
        # then transpose again 
        if axis ==0: 
            arr = arr._T 
    else: 
        arr = _filtering_1d_array ( arr, wf = weight_factor, b=beta  )
        
    if logify: arr = np.power (10, arr )
    
    return arr 

def gradient_boosting_regressor(
        X, y, n_estimators=100, learning_rate=0.1, max_depth=1):
    """
    Implement a simple version of Gradient Boosting Regressor.

    Gradient Boosting builds an additive model in a forward stage-wise fashion. 
    At each stage, regression trees are fit on the negative gradient of the loss function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,)
        The target values (real numbers).
    n_estimators : int, default=100
        The number of boosting stages to be run.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    max_depth : int, default=1
        The maximum depth of the individual regression estimators.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        The predicted values.

    Mathematical Formula
    --------------------
    Given a differentiable loss function L(y, F(x)), the general idea is 
    to iteratively construct additive models as follows:
    
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size,
    and h_{m} is the weak learner.

    Notes
    -----
    Gradient Boosting is widely used in machine learning for regression and 
    classification problems. It's effective in scenarios where data is not 
    linearly separable.

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> y_pred = gradient_boosting_regressor(X, y, n_estimators=100,
                                             learning_rate=0.1)
    >>> print(y_pred[:5])
    """
    from ..estimators import DecisionStumpRegressor
    # Initialize model
    F_m = np.zeros(len(y))
    # for m in range(n_estimators):
        # Compute negative gradient
        # residual = -(y - F_m)

        # # Fit a regression tree to the negative gradient
        # tree = DecisionTreeRegressor(max_depth=max_depth)
        # tree.fit(X, residual)

        # # Update the model
        # F_m += learning_rate * tree.predict(X)

    for m in range(n_estimators):
        # Compute negative gradient
        residual = -(y - F_m)
    
        # Fit a decision stump to the negative gradient
        stump = DecisionStumpRegressor()
        stump.fit(X, residual)
    
        # Update the model
        F_m += learning_rate * stump.predict(X)
    

    return F_m




    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
