# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Metrics are measures of quantitative assessment commonly used for 
estimating, comparing, and tracking performance or production. Generally,
a group of metrics will typically be used to build a dashboard that
management or analysts review on a regular basis to maintain performance
assessments, opinions, and business strategies.
"""
from __future__ import annotations 
import itertools 
import warnings  
import numpy as np 

from sklearn.metrics import precision_recall_curve, precision_score
from sklearn.metrics import recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.metrics import classification_report, mean_squared_error  
from sklearn.metrics import mean_absolute_error, r2_score, jaccard_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict 
from sklearn.preprocessing import label_binarize

from ._gofastlog import gofastlog 
from .api.formatter import MetricFormatter
from .tools.baseutils import standardize_input, filter_nan_from 
from .tools.baseutils import convert_array_dimensions
from .tools.coreutils import normalize_string 
from .tools.mathex import calculate_binary_iv, optimized_spearmanr 
from .tools.mathex import compute_sensitivity_specificity 
from .tools.mathex import calculate_multiclass_lr, calculate_multiclass_avg_lr
from .tools.mathex import compute_balance_accuracy
from .tools.validator import _is_numeric_dtype, _ensure_y_is_valid
from .tools.validator import check_epsilon, check_is_fitted
from .tools.validator import check_classification_targets, validate_nan_policy
from .tools.validator import ensure_non_negative, validate_multioutput 
from .tools.validator import parameter_validator, handle_zero_division
from .tools.validator import validate_sample_weights, validate_positive_integer 

_logger = gofastlog().get_gofast_logger(__name__)

__all__=[
    "precision_recall_tradeoff",
    "roc_tradeoff",
    "evaluate_confusion_matrix", 
    "mean_squared_log_error",
    "balanced_accuracy",
    "balanced_accuracy_score", 
    "information_value", 
    "mae_flex",
    "mse_flex", 
    "rmse_flex",
    "madev_flex", 
    "r2_flex", 
    "mean_absolute_percentage_error", 
    "explained_variance_score", 
    "median_absolute_error",
    "max_error_score",
    "mean_poisson_deviance", 
    "mean_gamma_deviance",
    "mean_absolute_deviation", 
    "dice_similarity_score", 
    "gini_score",
    "hamming_loss", 
    "fowlkes_mallows_score",
    "root_mean_squared_log_error", 
    "mean_percentage_error",
    "percentage_bias", 
    "spearmans_rank_score",
    "precision_at_k", 
    "ndcg_at_k", 
    "mean_reciprocal_score", 
    "jaccard_flex", 
    "geo_information_value", 
    "assess_regression_metrics", 
    "assess_classifier_metrics", 
    "log_likelihood_score", 
    "adjusted_r2_score", 
    "display_confusion_matrix", 
    "display_roc", 
    "likelihood_score", 
    "display_precision_recall", 
    "ndcg_at_k_with_controls", 
    "make_scorer", 
    "get_scorer", 
    "fetch_sklearn_scorers", 
    "fetch_scorer_functions", 
    "get_scorer_names", 
    "fetch_scorers", 
    "SCORERS"
    ]

def log_likelihood_score(
    y_true, y_pred, *, 
    consensus='positive', 
    strategy='ovr', 
    sample_weight=None, 
    epsilon=1e-8, 
    multioutput='uniform_average', 
    detailed_output=False,
    zero_division="warn"
):
    """
    Compute the log of the likelihood ratio (LLR) for multiclass classification, 
    using either one-versus-rest (OvR) or one-versus-one (OvO) strategies, with 
    an option to apply a logarithmic transformation to stabilize the numerical 
    range of the likelihood ratios.

    Parameters
    ----------
    y_true : array-like
        True class labels as integers. Represents the actual classifications
        of the samples.
    y_pred : array-like
        Predicted class labels as integers. Represents the classifications
        as predicted by a model or classifier.
    consensus : str, optional
        Type of likelihood ratio to compute: 'positive' or 'negative'.
        The 'positive' consensus (default) calculates the likelihood ratio
        favoring the presence of a condition (sensitivity divided by
        one minus specificity), which reflects how well the condition is
        identified when it is present. The 'negative' consensus calculates
        the likelihood ratio favoring the absence of a condition
        (one minus sensitivity divided by specificity), indicating how
        effectively a condition can be ruled out when the test is negative.
    strategy : str, optional
        Computation strategy: 'ovr' (one-versus-rest, default) or 'ovo'
        (one-versus-one). 'ovr' compares each class against all other classes
        collectively, while 'ovo' compares each pair of classes individually.
    sample_weight : array-like, optional
        Sample weights. If None, each sample contributes equally to the
        computation. Useful for weighting samples differently in the
        calculation, typically in cases where some classes are overrepresented
        or underrepresented.
    epsilon : float, optional
        A small constant added to the denominator to prevent division by zero.
        This helps in maintaining numerical stability. Default is 1e-8.
    multioutput : str, optional
        Determines how to return the output: 'uniform_average' or 'raw_values'.
        'uniform_average' computes the average of the metrics across all classes,
        treated equally. 'raw_values' returns a separate metric for each class.
    detailed_output : bool, optional
        If True, returns a detailed output including individual sensitivity
        and specificity values for each class or class pair, which can be
        particularly useful for detailed statistical analysis and diagnostics.
    zero_division : str, optional
        How to handle division by zero: 'warn' (default) will issue a warning,
        'ignore' will suppress any warnings and proceed with the computation,
        potentially substituting infinities where division by zero occurs.

    Returns
    -------
    float or MetricFormatter
        The log likelihood ratio or a structured metric formatter if 
        detailed_output is True.

    Examples
    --------
    >>> from gofast.metrics import log_likelihood_score
    >>> y_true = [0, 1, 2, 2, 1, 0]
    >>> y_pred = [0, 2, 1, 2, 1, 0]
    >>> log_likelihood_score(y_true, y_pred, consensus='positive', strategy='ovr')
    1.2296264763713125

    Notes
    -----
    The likelihood ratio (LR) quantifies the strength of the evidence in favor 
    of one class over another. Specifically, it is computed for a given class 
    or class pair as follows:
    
    .. math::
        LR_+ = \\frac{\\text{sensitivity}}{1 - \\text{specificity}}
    
    .. math::
        LR_- = \\frac{1 - \\text{sensitivity}}{\\text{specificity}}
    
    The LLR, or logarithm of the likelihood ratio, transforms these ratios by 
    taking the natural logarithm:
    
    When :math:`\\text{consensus}` is 'positive', the LLR is computed by:
    
    .. math::
        LLR = \\log(LR_+)

    Similarly, when :math:`\\text{consensus}` is 'negative', the LLR is:

    .. math::
        LLR = \\log(LR_-)
    
    The logarithmic transformation applied to the likelihood ratios (LLR) stabilizes 
    the numerical range, mitigating the influence of extreme values typically 
    encountered in raw LR calculations. This transformation makes LLR a more 
    stable and interpretable metric in settings where decisions are sensitive to 
    the magnitude of diagnostic evidence.
    
    The LLR is derived from the sensitivity and specificity of the diagnostic test:
    
    - Sensitivity (true positive rate) measures the proportion of actual positives 
      correctly identified.
    - Specificity (true negative rate) measures the proportion of actual negatives 
      correctly identified.
    
    When LLR is zero, it indicates that the evidence for or against a hypothesis 
    (class) is balanced; the sensitivity and specificity yield a ratio of one, 
    implying no preference toward positive or negative classification. This 
    near-balance point occurs when the product of the test's sensitivity and 
    the complement of specificity is approximately equal to the product of 
    specificity and the complement of sensitivity, expressed as:

    .. math::
        \\frac{\\text{sensitivity}}{1 - \\text{specificity}} \\approx 1

    And:

    .. math::
        \\frac{1 - \\text{sensitivity}}{\\text{specificity}} \\approx 1
        
    Positive LLR values indicate strong evidence in favor of a class, suggesting 
    that the test's sensitivity outweighs the false positive rate (1-specificity). 
    Conversely, negative LLR values suggest that the evidence supports the absence 
    of the class, indicating a higher false positive rate relative to sensitivity.
    
    By compressing the scale of outputs, the LLR facilitates easier comparison and 
    interpretation of results across different models and clinical conditions. This 
    compression enhances the utility of LLR in diagnostic and predictive settings by 
    providing a clear, interpretable measure of how much more (or less) likely the 
    test results are to suggest one class over another under varied clinical 
    scenarios.

    See Also 
    ---------
    likelihood_score: 
        Compute the likelihood ratio metric for binary or  multiclass 
        classification.
    """
    # Validate inputs
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    ensure_non_negative(
        y_true, y_pred, 
        err_msg="y must contain non-negative values for LR calculation.")
    epsilon = check_epsilon(epsilon, y_true, y_pred)
    multioutput = validate_multioutput(multioutput)
    consensus = parameter_validator(
        "consensus", target_strs={'negative', 'positive'})(consensus)
    # Handle unexpected strategy
    strategy = parameter_validator(
        "strategy", target_strs={'ovr', 'ovo'})(strategy)

    # Prepare to collect metrics
    scores = MetricFormatter(descriptor="Log_likelihood_score_metric")
    
    # Calculate metrics
    if strategy in ['ovo', 'ovr'] and len(np.unique(y_true)) > 2:
        results = calculate_multiclass_lr(
            y_true=y_true,
            y_pred=y_pred,
            consensus=consensus,
            strategy=strategy,
            epsilon=epsilon,
            multi_output=multioutput,
            apply_log_scale=True,
            include_metrics=True, 
            sample_weight= sample_weight, 
        )
        result, sensitivity, specificity = results
    else : 
        # Binary classification
        sensitivity, specificity = compute_sensitivity_specificity(
            y_true, y_pred,
            sample_weight= sample_weight,
            epsilon= epsilon 
            )
        if consensus == 'positive':
            result = sensitivity / (1 - specificity + epsilon)
        elif consensus == 'negative':
            result = (1 - sensitivity) / specificity + epsilon

        scores.sensitivity= sensitivity 
        scores.specificity= specificity 
        
    # Check if result contains zero or negative values, which will cause issues with log.
    if (result <= 0).any():
        if zero_division == 'warn':
            warnings.warn(
                "Division by zero or log of zero/negative encountered in log",
                RuntimeWarning
            )
 
    # Safe to compute logarithm where result is positive
    positive_results_mask = result > 0
    llr_score = np.empty_like(result)
    llr_score[positive_results_mask] = np.log(result[positive_results_mask])
    # Assign NaN for non-positive results if any remain
    llr_score[~positive_results_mask] = np.nan  

    # Store results in the formatter
    scores.llr_score = llr_score
    scores.sensitivity = sensitivity 
    scores.specificity = specificity
    scores.consensus = consensus

    # Return detailed output if requested
    if detailed_output:
        return scores

    return result

def likelihood_score(
    y_true, y_pred, *, 
    consensus='positive', 
    sample_weight=None, 
    strategy='ovr', 
    epsilon='auto', 
    zero_division='warn', 
    multioutput='uniform_average', 
    detailed_output=False, 
    ):
    """
    Compute the likelihood ratio metric for binary or multiclass classification.

    The likelihood ratio is computed based on the chosen 'consensus'. 
    In the 'positive' consensus, it is the ratio of sensitivity to one minus 
    specificity, and in the 'negative' consensus, it is the ratio of one minus 
    sensitivity to specificity [1]_. It supports both binary classification 
    and multiclass classification through one-vs-rest (OvR) or one-vs-one 
    (OvO) strategies.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels for the samples. Must be provided as integers
        representing class labels.
    
    y_pred : array-like of shape (n_samples,)
        Predicted labels or probabilities for each sample. These predictions
        are used to calculate sensitivity and specificity relative to the
        true labels.
    
    consensus : str, optional
        Specifies the type of likelihood ratio to compute: 'positive' or 'negative'.
        The default 'positive' computes the positive likelihood ratio, which
        assesses the probability of correctly identifying a condition when it is
        present. Conversely, 'negative' computes the ratio based on the probability
        of correctly dismissing a condition when it is not present.
    
    sample_weight : array-like of shape (n_samples,), optional
        Weights to apply to each sample during the calculation of metrics.
        If not provided, all samples are assumed to have equal weight.
    
    strategy : str, optional
        Strategy to manage multiclass classification scenarios: 'ovr' 
        (one-versus-rest) or 'ovo' (one-versus-one). 'ovr' assesses each class
        against the aggregate of other classes, while 'ovo' compares every pair
        of classes individually.
    
    epsilon : float or 'auto', optional
        A small number added to the denominator to prevent division by zero
        and ensure numerical stability. If 'auto', the machine epsilon is used,
        providing the smallest representable positive number such that
        1.0 + epsilon != 1.0.
    
    zero_division : str, optional
        Defines how to handle scenarios where division by zero occurs:
        'warn' (default) issues a warning; 'ignore' proceeds without warning,
        potentially leading to infinite or NaN results in the output.
    
    multioutput : str, optional
        Determines the method for aggregating output values in multiclass
        problems. 'uniform_average' (default) averages the computed metrics
        across all classes with equal weighting.
    
    detailed_output : bool, optional
        If set to True, the function returns a MetricFormatter object that
        provides a detailed breakdown of the likelihood scores, including
        individual calculations of sensitivity, specificity, and the selected
        computation strategy for each class or class pair. Default is False.

    Returns
    -------
    float or MetricFormatter
        The calculated likelihood ratio, or a MetricFormatter object with 
        detailed results if `detailed_output` is True.

    Examples
    --------
    >>> from gofast.metrics import likelihood_score
    >>> y_true = [1, 0, 1, 1, 0]
    >>> y_pred = [1, 1, 1, 0, 0]
    >>> likelihood_score(y_true, y_pred)
    1.3333222223259247

    >>> y_true = [0, 1, 2, 2, 1, 0]
    >>> y_pred = [0, 2, 1, 2, 1, 0]
    >>> likelihood_score(y_true, y_pred, strategy='ovr')
    3.9996900226983447

    >>> y_true = [0, 1, 1, 0, 0]
    >>> y_pred = [0, 0, 1, 0, 1]
    >>> result = likelihood_score(y_true, y_pred, detailed_output=True)
    >>> print(result.lr_score, result.sensitivity, result.specificity)
    0.66 0.5 0.75

    The first example calculates the likelihood ratio for a simple binary 
    classification.
    The second example demonstrates the use of the 'ovr' strategy in a 
    multiclass scenario.
    The third example returns detailed outputs including the likelihood score, 
    sensitivity, and specificity values.

    Notes
    -----

    The likelihood ratio (LR) provides a measure of a test's ability to 
    distinguish between those with and without the condition in question.

    For the 'positive' consensus, the LR (LR+) is calculated as:
    
    .. math::
        LR_+ = \\frac{\\text{sensitivity}}{1 - \\text{specificity}}

    LR+ indicates how many times more likely a positive test result is to be 
    observed in a true positive case compared to a false positive case. 
    A high LR+ value (significantly greater than 1) suggests that the test is 
    effective at identifying the condition when it is present.

    For the 'negative' consensus, the LR (LR-) is calculated as:
    
    .. math::
        LR_- = \\frac{1 - \\text{sensitivity}}{\\text{specificity}}

    LR- indicates how many times less likely a negative test result is to be 
    observed in a true negative case compared to a false negative case. 
    A low LR- value (close to 0 but greater than 0) suggests that the test 
    is effective at ruling out the condition when it is not present.

    References
    ----------
    .. [1] Mahbod Issaiy1, Hossein Ghanaati, Shahriar Kolahi, Madjid Shakiba,
           Amir Hossein Jalali, Diana Zarei, Sina Kazemian, Mahsa Alborzi
           Avanaki, and Kavous Firouznia (2024). Methodological insights into ChatGPT’s
           screening performance in systematic reviews. BMC Medical Research,
           24:78 https://doi.org/10.1186/s12874-024-02203-8

    """
    # Ensure y_true and y_pred are valid and have consistent lengths
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True,)
    
    ensure_non_negative(y_true, y_pred, 
        err_msg="y must contain non-negative values for LRE calculation."
    )
    # check epsilon 
    epsilon = check_epsilon(epsilon, y_true, y_pred  )
    multioutput= validate_multioutput(multioutput)

    consensus = parameter_validator( 
        "consensus", target_strs={'negative', 'positive'})(consensus)
    strategy = parameter_validator( 
        "strategy", target_strs={'ovr', 'ovo'})( strategy)
    
    scores= MetricFormatter (descriptor ="likelihood_score_metric")
    
    # Handle multiclass cases
    if len(np.unique(y_true)) > 2:
        # Apply One-vs-Rest or One-vs-One strategy
        result, sensitivity, specificity = calculate_multiclass_avg_lr ( 
            y_true, y_pred, 
            strategy=strategy, 
            consensus=consensus,
            epsilon=epsilon, 
            sample_weight = sample_weight, 
            multi_output=multioutput, 
            )
        scores.strategy= strategy
        scores.sensitivity= sensitivity
        scores.specificity= specificity
    else:
        # Binary classification
        sensitivity, specificity = compute_sensitivity_specificity(
            y_true, y_pred,
            sample_weight= sample_weight,
            epsilon= epsilon 
            )
        if consensus == 'positive':
            result = sensitivity / max(1 - specificity, epsilon)
        elif consensus == 'negative':
            result = (1 - sensitivity) / max(specificity,  epsilon) 

        scores.sensitivity= sensitivity
        scores.specificity= specificity
         
    if zero_division == 'warn' and (np.isinf(result).any() or np.isnan(
            result).any()):
        import warnings
        warnings.warn("Division by zero occurred", RuntimeWarning)
        return np.nan

    scores.lr_score= result 
    scores.consensus=consensus 
    if detailed_output: 
        return scores 
        
    return result

def percentage_bias(
    y_true, y_pred, *, 
    sample_weight=None, 
    epsilon='auto', 
    zero_division='warn', 
    multioutput='uniform_average',
):
    """
    Calculates the Percentage Bias (PBIAS) between true and predicted values.

    The Percentage Bias measures the average tendency of the predictions to
    overestimate or underestimate the actual values, expressed as a percentage.
    PBIAS is particularly valuable in hydrology [1]_, environmental science, 
    and economics for evaluating model performance in simulation and forecasting.
    It indicates the model's average deviation from observed values, allowing
    for the assessment of model bias.

    See more in :ref:`User Guide` 
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values. Must be non-negative. Represents observed or 
        actual values in the context of the model evaluation.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values by the model. Represents estimated values 
        that are compared against `y_true`.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample, allowing for the differential 
        consideration of the importance of each prediction.
    epsilon : {'auto', float}, optional
        Small constant added to `y_true` to avoid division by zero. If 'auto',
        dynamically determined based on the range of `y_true` values.
    zero_division : {'warn', 'ignore'}, optional
        Specifies how to handle the division by zero: 'warn' raises a warning;
        'ignore' suppresses the warning and proceeds with the calculation.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Defines how multiple output values are aggregated: 'raw_values' returns 
        a full set of errors, 'uniform_average' averages errors across all outputs.

    Returns
    -------
    float or ndarray
        The PBIAS value. If `multi_output` is 'raw_values', returns an array of
        PBIAS values for each output. A positive PBIAS indicates a model tendency
        to overestimate, and a negative value indicates an underestimation.

    Notes
    -----
    The Percentage Bias is crucial for understanding model accuracy in terms of
    systemic bias. It helps identify whether a model consistently overpredicts or
    underpredicts the observed values[2]_. In water resources and climate modeling,
    PBIAS provides insights into the reliability of hydrological forecasts and
    their potential biases [3]_.
    
    The Percentage Bias is defined as:

    .. math:: \text{PBIAS} = \frac{100}{n} \sum_{i=1}^{n}\\
        \left( \frac{y_{\text{pred},i} - y_{\text{true},i}}{y_{\text{true},i}} \right)

    where `n` is the number of samples, `y_pred` is the predicted value, and `y_true`
    is the actual value. A positive value indicates a tendency to overestimate, while
    a negative value indicates a tendency to underestimate.
    
    See Also
    --------
    mean_absolute_error : Compute the mean absolute error.
    mean_squared_error : Compute mean squared error.
    mean_absolute_percentage_error : Compute mean absolute percentage error.
    
    References
    ----------
    .. [1] Gelete, Gebre (2023). "Application of hybrid machine learning-based ensemble
          techniques for rainfall-runoff modeling". Earth Sciences Informatics, 2475-2495,
          https://doi.org/10.1007/s12145-023-01041-4.
          
    .. [2] Gupta, H.V., Sorooshian, S., and Yapo, P.O. (1999). Status of Automatic
          Calibration for Hydrologic Models: Comparison with Multilevel 
          Expert Calibration. Journal of Hydrologic Engineering, 4(2), 135-143.
          
    .. [3] Moriasi, D.N., Arnold, J.G., Van Liew, M.W., Bingner, R.L., Harmel,
           R.D., and Veith, T.L. (2007). Model evaluation guidelines for 
           systematic quantification of accuracy in watershed simulations. 
           Transactions of the ASABE, 50(3), 885-900.
         
    Examples
    --------
    >>> from gofast.metrics import percentage_bias
    >>> y_true = [100, 150, 200, 250, 300]
    >>> y_pred = [110, 140, 210, 230, 310]
    >>> print(percentage_bias(y_true, y_pred))
    1.3333
    
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True, multi_output=True )
    ensure_non_negative(y_true, 
        err_msg="y_true must contain non-negative values for PBIAS calculation."
    )
    # Determine epsilon value
    epsilon = check_epsilon(epsilon, y_true, scale_factor=1e-15)
    
    # Adjust y_true to avoid division by zero
    adjusted_y_true = np.clip(y_true, epsilon, np.inf)

    # Compute percentage bias
    percentage_bias = (y_pred - adjusted_y_true) / adjusted_y_true
    
    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        weighted_percentage_bias = np.average(percentage_bias, weights=sample_weight)
    else:
        weighted_percentage_bias = np.mean(percentage_bias)

    # Handle zero division if necessary
    if zero_division == 'warn' and np.any(adjusted_y_true == epsilon):
        warnings.warn("Potential division by zero encountered in"
                      " Percentage Bias calculation.", UserWarning)

    # Handle multioutput aggregation
    multioutput=validate_multioutput(multioutput)
    if multioutput == 'uniform_average':
        return np.mean(weighted_percentage_bias) * 100
    elif multioutput == 'raw_values':
        return weighted_percentage_bias * 100
    
def mean_squared_log_error(
    y_true, y_pred, *,  
    sample_weight=None, 
    clip_value=0, 
    epsilon="auto", 
    zero_division='warn', 
    multioutput='uniform_average'
):
    """
    Compute the Mean Squared Logarithmic Error (MSLE) between true and 
    predicted values.
    
    This metric is especially useful for regression problems where the target 
    values are expected to be in a multiplicative scale, as it penalizes 
    underestimates more than overestimates. The function allows for clipping 
    predictions to a minimum value for numerical stability and includes an 
    epsilon to ensure values are strictly positive before logarithmic 
    transformation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values. Must be non-negative.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values. Can contain any real numbers.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample.
    clip_value : float, optional
        The value to clip `y_pred` values at minimum. This ensures that
        the logged values are not negative, by default 0.
    epsilon : float or "auto", optional
        A small value added to `y_pred` and `y_true` after clipping, to
        ensure they are strictly positive before applying the logarithm.
        If "auto", epsilon is dynamically determined based on the data.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during MSLE calculation.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Defines aggregating of multiple output values. Default is 'uniform_average'.

    Returns
    -------
    float or ndarray
        The MSLE between `y_true` and `y_pred`. If `multioutput` is 'raw_values',
        returns an array of MSLE values for each output.

    Notes
    -----
    The MSLE is defined as:

    .. math:: \mathrm{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(p_i + 1) - \log(a_i + 1))^2

    where :math:`p_i` and :math:`a_i` are the predicted and actual values, respectively,
    and :math:`n` is the number of samples.

    MSLE can be interpreted as a measure of the ratio between the true and 
    predicted values. By using the logarithmic scale, it penalizes relative 
    differences - making it useful for predicting exponential growths without 
    being too sensitive to large errors when the predicted and true values are
    both large numbers [1]_.
    
    It is important that `y_true` contains non-negative values only, as the
    logarithm of negative values is undefined. The function enforces `y_pred`
    values to be at least `clip_value` to avoid taking the logarithm of
    negative numbers or zero, potentially leading to `-inf` or `NaN`. The
    addition of `epsilon` ensures that even after clipping, no value is exactly
    zero before the logarithm is applied, providing a buffer against numerical
    instability.

    See Also
    --------
    mean_squared_error : Compute the mean squared error.
    mean_absolute_error : Compute the mean absolute error.
    r2_score : R^2 (coefficient of determination) regression score function.

    References
    ----------
    .. [1] Wikipedia, "Mean squared logarithmic error",
           https://en.wikipedia.org/wiki/Mean_squared_logarithmic_error
           
    Examples
    --------
    >>> from gofast.metrics import mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y_true, y_pred)
    0.03973
    
    >>> mean_squared_log_error(y_true, [2, -5, 4, 8], clip_value=0.01)
    0.8496736598821342
    # Example output with clipping and epsilon adjustment
    
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric= True, multi_output =True  )

    # Ensure non-negativity of y_true 
    ensure_non_negative(
        y_true, err_msg= "y_true must contain non-negative values only."
    ) 

    # Check epsilon 
    epsilon = check_epsilon(epsilon, y_true, scale_factor= 1e-15)
    # Adjust y_pred for log calculation
    y_pred = np.clip(y_pred, clip_value, np.inf) + epsilon
    y_true = np.clip(y_true, 0, np.inf) + epsilon

    # Compute squared log error
    squared_log_error = (np.log1p(y_pred) - np.log1p(y_true)) ** 2

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        squared_log_error *= sample_weight
        denominator = np.sum(sample_weight)
    else:
        denominator = y_true.size 
    # Avoid division by zero
    if zero_division == 'warn' and denominator == 0:
        warnings.warn("Division by zero encountered in MSLE calculation.",
                      UserWarning)
        return 0.0 if epsilon > 0 else np.nan

    msle = np.sum(squared_log_error) / denominator

    # Handle multioutput
    validate_multioutput(multioutput)
    
    if multioutput == 'uniform_average':
        # Ensure msle is not an array when calculating uniform average
        msle = np.mean(msle) if np.ndim(msle) > 0 else msle

    return msle 

def balanced_accuracy(
    y_true, y_pred, *, 
    normalize=False, 
    sample_weight=None, 
    strategy='ovr', 
    epsilon=1e-15, 
    zero_division=0,
    ):
    """
    Compute the Balanced Accuracy score for binary and multiclass classification
    tasks. This metric is especially useful in situations with imbalanced
    classes. It allows choosing between One-vs-Rest (OVR) and One-vs-One
    (OVO) strategies for multiclass scenarios, providing flexibility based
    on the specific characteristics of the dataset and the problem at hand.

    Parameters
    ----------
    y_true : array-like
        True labels of the data.
    y_pred : array-like
        Predicted labels by the classifier.
    normalize : bool, optional
        If True, normalize the confusion matrix before computing metrics, by
        default False.
    sample_weight : array-like, optional
        Array of weights that are assigned to individual samples, by default None.
    strategy : str, optional
        Strategy for multiclass metrics calculation - either 'ovr' for One-vs-Rest
        or 'ovo' for One-vs-One, by default 'ovr'.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-15.
    zero_division : int, optional
        Value to return when there is a division by zero, by default 0.
        
    Returns
    -------
    float
        The Balanced Accuracy score.

    Notes
    -----
    The Balanced Accuracy in the binary classification scenario is defined as the
    average of sensitivity (true positive rate) and specificity (true negative rate):

    .. math:: BA = \frac{TPR + TNR}{2}

    Where:
    - TPR (True Positive Rate) = \frac{TP}{TP + FN}
    - TNR (True Negative Rate) = \frac{TN}{TN + FP}

    In multiclass scenarios, the metric is computed as the average of sensitivity
    for each class, considering each class as the positive class in turn, which
    corresponds to the One-vs-Rest (OVR) strategy. The One-vs-One (OVO) strategy
    involves averaging the metric over all pairs of classes.

    Examples
    --------
    Binary classification example:

    >>> from gofast.metrics import balanced_accuracy
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0]
    >>> balanced_accuracy(y_true, y_pred)
    0.75

    Multiclass classification example with OVR strategy:

    >>> y_true = [0, 1, 2, 2, 1]
    >>> y_pred = [0, 2, 2, 1, 1]
    >>> balanced_accuracy(y_true, y_pred, strategy='ovr')
    0.666

    Multiclass classification example with OVO strategy:

    >>> balanced_accuracy(y_true, y_pred, strategy='ovo')
    0.666
    """
    # Ensure y_true and y_pred are valid and have consistent lengths
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred)

    # Check that y_true and y_pred are suitable for classification
    y_true, y_pred = check_classification_targets(
        y_true, y_pred, strategy="custom logic")
    labels = unique_labels(y_true, y_pred)
    
    if len(labels) == 2:  # Binary classification scenario
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0] + epsilon)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + epsilon)
        
        # Return the Balanced Accuracy
        return (sensitivity + specificity) / 2
    
    else:  # Multiclass classification scenario
        strategy =normalize_string(
            strategy, target_strs=['ovr', 'ovo'], match_method='contains', 
            return_target_only= True, raise_exception= True, 
            error_msg=("strategy parameter must be either 'ovr' or 'ovo'") 
            ) 
        if strategy == 'ovr':
            sensitivities = []
            for label in labels:
                binary_y_true = (y_true == label).astype(int)
                binary_y_pred = (y_pred == label).astype(int)
                cm = confusion_matrix(binary_y_true, binary_y_pred, sample_weight=sample_weight)
                
                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                tp = cm[1, 1]
                fn = cm[1, 0]
                sensitivity = tp / (tp + fn + epsilon) if (tp + fn) > 0 else zero_division
                sensitivities.append(sensitivity)
            # Compute average sensitivity across all classes
            return np.mean(sensitivities)
        
        elif strategy == 'ovo':
            # For OVO, use a simplified approach as an example, like averaging pairwise AUC scores
            # Note: This is a simplification and may not align with all 
            # interpretations of OVO strategy
            y_bin = label_binarize(y_true, classes=labels)
            auc_scores = []
            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    if i >= j:
                        continue  # Avoid duplicate pairs and self-comparison
                    specific_sample_weight = sample_weight[(y_true == label_i) | (
                        y_true == label_j)] if sample_weight is not None else None
                    specific_y_true = y_bin[:, i][(y_true == label_i) | (y_true == label_j)]
                    specific_y_pred = y_bin[:, i][(y_true == label_i) | (y_true == label_j)]
                    auc_score = roc_auc_score(specific_y_true, specific_y_pred,
                                              sample_weight=specific_sample_weight)
                    auc_scores.append(auc_score)
            return np.mean(auc_scores)
        
def information_value(
    y_true, y_pred, *, 
    problem_type='binary', 
    epsilon='auto', 
    scale='binary_scale', 
    method='binning', 
    bins='auto', 
    bins_method='freedman_diaconis',
    data_range=None, 
    ):
    """
    Calculate the Information Value (IV) for various types of classification 
    and regression problems. 
    
    The :term:`IV` quantifies the predictive power of a feature or model, with  
    higher values indicating greater predictive power. This function supports 
    ``binary``, ``multiclass``, ``multilabel`` classifications, and ``regression``
     problems, allowing for flexibility in its application across different data 
    science tasks.

    Parameters
    ----------
    y_true : array-like
        True labels or values. For classification problems, these should be class
        labels. For regression problems, these are continuous target values.
        
    y_pred : array-like
        Predicted probabilities or values. For classification, these are typically
        the probabilities of belonging to the positive class (in binary classification)
        or class probabilities (in multiclass classification). For regression 
        problems, these are the predicted continuous values.
        
    problem_type : str, optional
        Specifies the type of problem for which the IV is being calculated. Valid
        options are ``'binary'``, ``'multiclass'``, ``'multilabel'``, and 
        ``'regression'``. This parameter determines how the IV calculation is 
        adapted to fit the problem context. Default is 'binary'.
        
    epsilon : float or "auto", optional
        A small epsilon value added to probabilities to prevent division by zero
        in the logarithmic calculation. If "auto", an appropriate epsilon value
        is dynamically determined based on the predicted values. Default is "auto".
        
    scale : str or float, optional
        For multiclass and multilabel problems, defines how the IV should be scaled
        or normalized. The default ``'binary_scale'`` scales these problems 
        to a binary IV scale for easier interpretation. If a float is provided,
        it custom scales the IV by this factor.
        
    method : str, optional
        Specifies the calculation method for Information Value (IV). Two methods 
        are supported:
        - 'base': A straightforward calculation based on the overall distribution 
          of events and non-events in the dataset. Suitable for a high-level 
          overview of predictive power.
        - 'binning': A more detailed analysis that divides the predicted probabilities
          (`y_pred`) into bins and calculates IV within each bin. This method is 
          valuable for examining how the model's predictive power varies across 
          different probability ranges. It requires specifying `bins` and can be 
          fine-tuned further with `bins_method` and `data_range`. 
          Default is ``'binning'``.
        
    bins : int, 'auto', optional
        Defines the number of bins used for the 'binning' method. This parameter 
        is crucial for segmenting the predicted probabilities (`y_pred`) into 
        discrete bins, allowing for a more granular analysis of the model's 
        predictive power. If set to 'auto', the number of bins is determined 
        using the method specified by `bins_method`. Providing an integer directly 
        specifies the fixed number of bins to use. Default is ``'auto'``.

    bins_method : str, optional
        Specifies the method to determine the optimal number of bins when 
        `bins` is set to ``'auto'``. Available methods are:
        - 'freedman_diaconis': Employs the Freedman-Diaconis rule, which bases 
          the bin width on the data's interquartile range and the cube root of 
          data size. Ideal for a wide range of distributions.
          .. math:: \text{bin width} = 2 \cdot \frac{IQR}{\sqrt[3]{n}}
        - 'sturges': Uses Sturges' formula, which is a function of the data size. 
          Suitable for normal distributions but may perform poorly for large, 
          skewed datasets.
          .. math:: \text{bins} = \log_2(n) + 1
        - 'sqrt': Calculates the square root of the data size to determine the 
          number of bins. A simple heuristic useful for smaller datasets.
          .. math:: \text{bins} = \sqrt{n}
        The chosen method can significantly impact the IV calculation, especially 
        in the 'binning' approach, influencing the resolution of the predictive 
        power analysis across different probability ranges.
        Default method is 'freedman_diaconis'.
        
    data_range : tuple, optional
        A tuple specifying the minimum and maximum values to consider for binning 
        when `method` is set to 'binning'. This parameter allows focusing the IV 
        analysis on a specific range of predicted probabilities, which can be 
        useful for excluding outliers or focusing on a region of interest. If None,
        the entire range of `y_pred` is used. Default is None.
        
    Returns
    -------
    float
        The calculated Information Value (IV) score. Positive values indicate
        predictive power, with higher values being preferable. For regression
        and multiclass/multilabel classifications with `scale='binary_scale'`,
        the IV is adjusted to fit a binary classification context.

    Raises
    ------
    ValueError
        If an invalid `problem_type` is specified.
        
    Notes
    -----
    The Information Value (IV) quantifies the predictive power of a feature or model, 
    illustrating its ability to distinguish between classes or predict outcomes. It is 
    a measure of the effectiveness of a variable or model in predicting the target.

    Mathematical formulations for IV across different problem types 
    are as follows:

    - For binary classification:
      The IV is computed as the sum of differences between the proportions of 
      non-events and events, each multiplied by the logarithm of the ratio of
      these proportions, adjusted by a small epsilon (`\epsilon`) to prevent 
      division by zero:
          
      .. math::
        IV = \sum \left((\% \text{{ of non-events}} - \% \text{{ of events}}) \times 
        \ln\left(\frac{\% \text{{ of non-events}} + \epsilon}
        {\% \text{{ of events}} + \epsilon}\right)\right)

    - For multiclass classification:
      The IV is adapted from the average log loss for multiclass classification, 
      normalized to the binary IV scale. This normalization is essential for 
      comparing the predictive power of multiclass models to binary models:
      The log loss for multiclass classification is given by:
      
      .. math::
        \text{{log_loss}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} 
        y_{ij} \log(p_{ij})
        
        IV = -\frac{1}{\log(2)} \cdot \text{{log_loss}}(y\_true, y\_pred)
        
      where \(N\) is the number of samples, \(M\) is the number of classes, 
      \(y_{ij}\) is a binary indicator of whether class \(j\) is the correct 
      classification for sample \(i\), and \(p_{ij}\) is the model probability 
      of assigning class \(j\) to sample \(i\).

    - For multilabel classification:
      Similar to multiclass, the IV for multilabel classification uses the 
      negative average binary cross-entropy loss, normalized to the binary scale. 
      This approach allows direct comparison between the predictive powers 
      of multilabel and binary scale. The binary cross-entropy for multilabel
      classification is given as:
      
      .. math::
        \text{{binary\_crossentropy}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} 
        \left( y_{ij} \log(p_{ij}) + (1 - y_{ij}) \log(1 - p_{ij}) \right)
        
        IV = -\frac{1}{\log(2)} \cdot \text{{binary\_crossentropy}}(y\_true, y\_pred)
        
      where \(N\) is the number of samples, \(M\) is the number of labels, 
      \(y_{ij}\) indicates whether label \(j\) is relevant to sample \(i\), 
      and \(p_{ij}\) is the predicted probability of label \(j\) for sample \(i\).

    - For regression:
      In regression problems, the IV is determined by the negative mean squared error 
      (MSE) between the true and predicted values. Lower (more negative) MSE values, 
      indicating closer predictions to the actual values, translate into higher IV 
      scores, signifying better predictive capability:
      
      .. math::
        IV = -\frac{1}{N} \sum_{i=1}^{N} (y_{i} - \hat{y_{i}})^2
        
      where \(N\) is the number of samples, \(y_{i}\) is the true value for 
      sample \(i\), and \(\hat{y_{i}}\) is the predicted value for sample \(i\).

    These formulations adjust the traditional concept of IV for a broad spectrum 
    of applications beyond binary classification, enhancing its versatility as a 
    metric for assessing model performance across various types of predictive 
    modeling tasks.

    IV is invaluable in fields such as credit scoring and risk modeling, aiding in 
    the evaluation of a model's or variable's predictive power. By analyzing IV, 
    data scientists can pinpoint the most informative features, leading to the 
    development of superior predictive models.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import log_loss, mean_squared_error
    >>> from gofast.metrics import information_value 
    
    Binary classification with automatically determined epsilon:

    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0.1, 0.9, 0.8, 0.2]
    >>> print(information_value(y_true, y_pred, problem_type='binary'))
    1.3219280948873623
    >>> y_true_binary = np.array([0, 1, 1, 0, 1, 0])
    >>> y_pred_binary = np.array([0.2, 0.7, 0.6, 0.3, 0.8, 0.1])
    >>> iv_binary = information_value(y_true_binary, y_pred_binary, problem_type='binary')
    >>> iv_binary
    0.7621407247492977
    
    Multiclass classification with specified scale:

    >>> y_true = [2, 1, 0, 1, 2]
    >>> y_pred = [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1],
    ...           [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]]
    >>> print(information_value(y_true, y_pred, problem_type='multiclass'))
    -0.6365141682948128

    >>> y_true_multiclass = np.array([0, 1, 2, 1, 0])
    >>> y_pred_multiclass = np.array([[0.2, 0.4, 0.4], [0.7, 0.1, 0.2],
    ...                                  [0.1, 0.2, 0.7], [0.3, 0.3, 0.4], 
    ...                                  [0.8, 0.1, 0.1]])
    >>> iv_multiclass = information_value(y_true_multiclass, y_pred_multiclass,
    ...                                      problem_type='multiclass')
    >>> iv_multiclass
    -0.6729845552217573
    
    Multilabel classification with specified scale:
 
    >>> y_true_multilabel = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> y_pred_multilabel = np.array([[0.8, 0.2, 0.7], [0.1, 0.9, 0.3], 
    ...                                  [0.9, 0.7, 0.1], [0.2, 0.1, 0.8]])
    >>> iv_multilabel = information_value(y_true_multilabel, y_pred_multilabel,
    ...                                      problem_type='multilabel')
    >>> iv_multilabel
    -0.4750837692748695
 
    >>> y_true_regression = np.array([10.5, 12.1, 9.8, 11.2, 10.0])
    >>> y_pred_regression = np.array([10.2, 12.3, 9.5, 11.0, 10.1])
    >>> iv_regression = information_value(y_true_regression, y_pred_regression,
    ...                                      problem_type='regression')
    >>> iv_regression
    -0.04239999999999994
    
    Regression with custom scale factor:

    >>> y_true = [3.5, 2.5, 4.0, 5.5]
    >>> y_pred = [3.0, 2.7, 4.1, 5.0]
    >>> print(information_value(y_true, y_pred, problem_type='regression', scale=1.0))
    -0.0475
    
    Using the 'binning' method with automatically determined bins:
    
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.6, 0.8, 0.05, 0.9])
    >>> iv_auto_bins = information_value(y_true, y_pred, 
    ...                                  problem_type='binary', 
    ...                                  method='binning', 
    ...                                  bins='auto', 
    ...                                  bins_method='freedman_diaconis')
    >>> print(f"IV with auto bins: {iv_auto_bins}")

    Specifying a fixed number of bins for the 'binning' method:
    
    >>> iv_fixed_bins = information_value(y_true, y_pred, 
    ...                                   problem_type='binary', 
    ...                                   method='binning', 
    ...                                   bins=5)
    >>> print(f"IV with fixed bins: {iv_fixed_bins}")
    
    Using the 'binning' method with a specified data range:

    >>> y_true = np.array([0, 1, 0, 1, 1, 0])
    >>> y_pred = np.array([0.05, 0.95, 0.2, 0.85, 0.9, 0.1])
    >>> iv_data_range = information_value(y_true, y_pred, 
    ...                                   problem_type='binary', 
    ...                                   method='binning', 
    ...                                   bins='auto', 
    ...                                   bins_method='freedman_diaconis',
    ...                                   data_range=(0.1, 0.9))
    >>> print(f"IV with specified data range: {iv_data_range}")

    The 'binning' method allows for a nuanced understanding of model performance, 
    especially useful in scenarios where predictive power might vary significantly 
    across the probability spectrum. Specifying `data_range` can refine this 
    analysis, offering insights into specific segments of the prediction range.
    
    See Also 
    ----------
    gofast.tools.calculate_binary_iv: 
        Calculate the Information Value (IV) for binary classification problems
        using a base or binning approach.
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True, 
                                        multi_output= True )
    # Determine an appropriate epsilon value if set to "auto"
    epsilon = check_epsilon(epsilon, y_pred)
    # Initialize IV calculation
    iv = None
    if problem_type == 'binary':
        iv = calculate_binary_iv(
            y_true, y_pred, method=method,
            bins=bins, epsilon= epsilon, 
            bins_method=bins_method,
            data_range=data_range, 
            )
    elif problem_type == 'multiclass':
        # Normalize to binary IV scale if requested
        iv = -log_loss(y_true, y_pred, eps=epsilon) / np.log(
            2) if scale == 'binary_scale' else -log_loss(y_true, y_pred, eps=epsilon)
        
    elif problem_type == 'multilabel':
        iv =( 
            -np.mean(y_true * np.log(y_pred + epsilon) 
            + (1 - y_true) * np.log(1 - y_pred + epsilon)) / np.log(2)
            if scale == 'binary_scale' 
            else -np.mean(y_true * np.log(y_pred + epsilon) 
                 + (1 - y_true) * np.log(1 - y_pred + epsilon))
            )
        
    elif problem_type == 'regression':
        iv = -mean_squared_error(y_true, y_pred)
        
    else:
        raise ValueError("Invalid 'problem_type'. Use 'binary', 'multiclass',"
                         " 'multilabel', or 'regression'.")

    # Apply custom scale if specified
    if isinstance(scale, float):
        iv *= scale
 
    return iv

def geo_information_value(
        Xp, Np, Sp, *, aggregate=True, epsilon=1e-15, clip_upper_bound=None):
    """
    Calculate the Geographic Information Value (Geo-IV) for assessing the influence
    of various factors on landslide occurrences. Geo-IV quantifies the sensitivity 
    and relevance of each factor in predicting spatial events such as landslides.

    This metric implements the Information Value (InV) method, a statistical 
    approach that evaluates the predictive power of spatial parameters and their 
    relationships with landslide occurrences. The method is particularly useful in 
    geosciences for landslide susceptibility mapping and risk assessment [1]_.

    Parameters
    ----------
    Xp : array-like
        An array representing causative factors, with shape (n_samples,).
    Np : array-like
        The number of pixels associated with each causative factor `Xp`,
        with shape (n_samples,).
    Sp : array-like
        The number of landslide pixels in the presence of causative factor `Xp`,
        with shape (n_samples,).
    aggregate : bool, optional, default=True
        If True, returns the overall Geo-IV value by summing up individual
        information values.
    epsilon : float, optional, default=1e-15
        A small value added to avoid division by zero during calculations.
    clip_upper_bound : float, optional
        An upper bound for clipping `Np` values to prevent excessively high
        ratios. If None, clipping is only applied at the lower bound (`epsilon`).

    Returns
    -------
    array-like or float
        The calculated Geo-IV for each causative factor, or the overall Geo-IV
        if `aggregate` is True.

    Notes
    -----
    The Geo-IV of each causative factor `X_i` is calculated using the formula:

    .. math::
        I_i = \log\left(\frac{S_i / N_i}{S / N}\right)

    where `S_i` is the landslide pixel number in the presence of `X_i`, `N_i` is
    the number of pixels associated with `X_i`, `S` is the total number of
    landslide pixels, and `N` is the total number of pixels in the study area.

    Positive values of Geo-IV indicate a factor's relevance in the occurrence
    of landslides, while negative values suggest irrelevance. The magnitude
    of the value reflects the strength of the correlation [1]_.

    References
    ----------
    .. [1] Sarkar S, Kanungo D, Patra A (2006). GIS-Based Landslide Susceptibility
           Mapping: A Case Study in the Indian Himalayas. In: Disaster Mitigation
           of Debris Flows, Slope Failures, and Landslides. Universal Academic
           Press, Tokyo, pp. 617-624.
           DOI: https://doi.org/10.1007/s11629-019-5839-3

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.metrics import geo_information_value
    >>> Xp = np.array([1, 2, 3])  # Example causative factors
    >>> Np = np.array([100, 200, 300])  # Pixels associated with each factor
    >>> Sp = np.array([10, 20, 30])  # Landslide pixels for each factor
    >>> geo_iv = geo_information_value(Xp, Np, Sp, aggregate=True)
    >>> print(f"Overall Geographic Information Value: {geo_iv}")
    """
    # Ensure inputs are numpy arrays and convert single numbers to arrays
    try: 
        Xp, Np, Sp = map(lambda x: np.asarray(x, dtype=float), [Xp, Np, Sp])
    except ValueError:
        raise ValueError(
            "All inputs (Xp, Np, Sp) must be convertible to float arrays.")
    # Ensure all inputs have compatible shapes
    if not (Xp.shape == Np.shape == Sp.shape):
        raise ValueError(
            "Shapes of Xp, Np, and Sp must be identical. Received shapes"
            f" Xp: {Xp.shape}, Np: {Np.shape}, Sp: {Sp.shape}.")
    
    # Validate inputs are not empty
    if Xp.size == 0 or Np.size == 0 or Sp.size == 0:
        raise ValueError("Inputs Xp, Np, and Sp must not be empty.")
    
    # Apply clipping with epsilon as lower bound and clip_upper_bound as upper bound
    Np = np.clip(Np, epsilon, clip_upper_bound if clip_upper_bound is not None else np.max(Np))
    total_N = np.sum(Np)
    if total_N <= 0:
        raise ValueError("Sum of Np must be greater than zero after applying epsilon.")
        
    total_S = np.sum(Sp)
    if total_S <= 0:
        raise ValueError("Sum of Sp must be greater than zero.")

    # Calculate the Geo-IV for each element
    Iv = np.log10((Sp / Np) / (total_S / total_N))
    
    # Aggregate the Geo-IV values if requested
    return np.sum(Iv) if aggregate else Iv

def assess_regression_metrics(
    y_true, 
    y_pred=None, 
    X=None, 
    model=None, *,
    sample_weight=None, 
    multioutput='uniform_average',
    force_finite=True, 
    clip_value=0, 
    epsilon=1e-15
    ):
    """
    Assess a comprehensive set of metrics for evaluating the performance of
    regression models. 
    
    Function simplifies the process of model evaluation by computing common 
    regression metrics including Mean Absolute Error (MAE), Mean Squared Error
    (MSE), Root Mean Squared Error (RMSE), :math:`R^2` (coefficient of determination),
    adjusted :math:`R^2`, Mean Squared Logarithmic Error (MSLE), and
    Median Absolute Error (MedAE) [1]_.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like, optional
        Predicted target values. If None, `model` and `X` must be provided to
        generate predictions.
    X : array-like, optional
        Feature dataset. Required if `y_pred` is None.
    model : estimator object implementing 'fit', optional
        The object to use to fit the data and generate predictions. Required if
        `y_pred` is None.
    sample_weight : array-like, optional
        Sample weights.
    multioutput : str, optional
        Defines aggregating of multiple output scores. Default is
        'uniform_average'.
    force_finite : bool, optional
        Forces the output scores to be finite by clipping. Default is True.
    clip_value : float, optional
        Minimum log value to use for mean_squared_log_error to avoid taking log
        of zero. Default is 0.
    epsilon : float, optional
        Small offset to add to input values to avoid division by zero or taking
        log of zero in metrics calculation. Default is 1e-15.

    Returns
    -------
    scores : Bunch
        A dictionary-like object containing the calculated metrics.

    Notes
    -----
    - The function assumes `y_true` and `y_pred` are already validated for
      their shapes and data types.
    - Adjusted R² is calculated only if `X` is provided and has more than one
      feature, otherwise, it's set to None.
    - Mean Squared Logarithmic Error (MSLE) uses `clip_value` and `epsilon` to
      ensure stability in its calculation.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import fetch_california_housing
    >>> from gofast.metrics import assess_regression_metrics
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> metrics = assess_regression_metrics(y_test, model=model, X=X_test)
    >>> print(metrics)

    References
    ----------
    .. [1] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). 
       An Introduction to Statistical Learning. Springer. 
       DOI: 10.1007/978-1-4614-7138-7

    See Also
    --------
    sklearn.metrics.mean_absolute_error : 
        Compute the mean absolute error regression loss.
    sklearn.metrics.mean_squared_error : 
        Compute the mean squared error regression loss.
    sklearn.metrics.r2_score : 
        Compute R², the coefficient of determination.
    sklearn.metrics.mean_squared_log_error :
        Compute the mean squared logarithmic error regression loss.
    sklearn.metrics.median_absolute_error : 
        Compute the median absolute error regression loss.
    sklearn.metrics.make_scorer :
        Make a scorer from a performance metric or loss function.

    Examples of other libraries that provide similar functionality:
    - scikit-learn (sklearn.metrics): Offers a comprehensive set of metrics 
      for evaluating regression models.
    - Statsmodels: Provides classes and functions for the estimation of many 
      different statistical models, as well as for conducting statistical tests,
      and statistical data exploration which includes some regression metrics.
    
    """
    y_true = np.asarray(y_true)
    if y_pred is None:
        if X is None or model is None:
            raise ValueError("When 'y_pred' is None, both 'X' and 'model' must be provided.")
        check_is_fitted(model)
        y_pred = model.predict(X)
    
    # Ensure y_true and y_pred are valid.
    # Assuming _ensure_y_is_valid is defined to validate the inputs
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    # Calculate common regression evaluation scores
    scores = MetricFormatter(
        mean_absolute_error=mean_absolute_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput),
        mean_squared_error=mean_squared_error(y_true, y_pred),
        rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score=r2_score(
            y_true, y_pred, sample_weight=sample_weight, multioutput= multioutput,
            force_finite=force_finite) ,
        # Adjusted R2 can be calculated if X is provided and has more than one feature
        adjusted_r2_score=adjusted_r2_score(y_true, y_pred, X) if X is not None else None,
        mean_squared_log_error=mean_squared_log_error(
            y_true, y_pred, clip_value=clip_value, epsilon =epsilon
            ) if not np.any (y_true < 0 ) else None,
        median_absolute_error=median_absolute_error(y_true, y_pred),
    )
    return scores

def assess_classifier_metrics(
    y_true, y_pred=None, X=None, model=None, *,
    average="binary", multi_class="raise", 
    normalize=True, sample_weight=None, 
    **scorer_kws):
    """
    Evaluate classification model performance through a comprehensive 
    set of metrics, including accuracy, recall, precision, F1 score, and 
    ROC-AUC score. This function provides a unified approach to model 
    evaluation across binary, multiclass, and multilabel classification 
    scenarios.
    
    Parameters
    ----------
    y_true : array-like
        True labels for classification.
    y_pred : array-like, optional
        Predicted labels or probabilities. Required unless `X` and `model` 
        are provided.
    X : array-like, optional
        Feature set for making predictions using `model`. Required if `y_pred` 
        is None.
    model : object
        A fitted classifier object that has a `predict` or `predict_proba` method.
        Required if `y_pred` is None.
    average : str, optional
        Strategy for averaging binary metrics in multiclass/multilabel 
        scenarios. Default is "binary".
    multi_class : str, optional
        Strategy for treating multiclass data for ROC-AUC score calculation. 
        Default is "raise".
    normalize : bool, optional
        Whether to normalize the accuracy score. Default is True.
    sample_weight : array-like, optional
        Sample weights for metrics calculation.
    **scorer_kws : dict
        Additional keyword arguments for metric functions.

    Returns
    -------
    scores : Bunch
        Dictionary-like object containing evaluation metrics: accuracy, 
        recall, precision, F1 score, and ROC-AUC score.

    Notes
    -----
    - The ROC-AUC score calculation depends on the `predict_proba` method. 
      If it is not available or `multi_class` is "raise", ROC-AUC score 
      calculation will be skipped with a warning.
    - This function automatically validates input shapes and data types 
      for `y_true` and `y_pred`.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.metrics import assess_classifier_metrics
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ... X, y, test_size=0.25, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X_train, y_train)
    >>> metrics = assess_classifier_metrics(
    ...    y_test, model=model, X=X_test,average='macro')
    >>> print(metrics)

    See Also
    --------
    sklearn.metrics.accuracy_score : 
        Compute the accuracy classification score.
    sklearn.metrics.recall_score : 
        Compute the recall, the ability of the classifier to find all positive samples.
    sklearn.metrics.precision_score : 
        Compute the precision, the ability of the classifier not to label as 
        positive a sample that is negative.
    sklearn.metrics.f1_score : 
        Compute the F1 score, the weighted average of precision and recall.
    sklearn.metrics.roc_auc_score : 
        Compute the Area Under the Receiver Operating Characteristic 
        Curve (ROC AUC) from prediction scores.
    sklearn.metrics.classification_report :
        Build a text report showing the main classification metrics.
    
    Examples of other libraries that provide similar functionality:
    - scikit-learn (sklearn.metrics): Offers a wide range of performance metrics
      for classification, including those used here.
    - Yellowbrick: A suite of visual diagnostic tools built on Scikit-Learn 
      and Matplotlib that also offers model evaluation metrics visualization.
      
    References
    ----------
    - Powers, D.M.W. (2011). "Evaluation: From Precision, Recall and F-Score 
      to ROC, Informedness, Markedness & Correlation." Journal of Machine 
      Learning Technologies.
    """
    # Convert y_true to numpy array for consistency
    y_true = np.asarray(y_true)
    
    # Validate model and prediction logic
    if y_pred is None:
        if X is None or model is None:
            raise ValueError("When 'y_pred' is None, both 'X' and 'model' must be provided.")
        # Check if the model is fitted
        check_is_fitted(model)
        y_pred = model.predict(X)
        
    # Ensure y_true and y_pred are valid. 
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True )
    # Calculate evaluation scores
    scores = MetricFormatter(
        accuracy_score=accuracy_score(
            y_true, y_pred, normalize=normalize, sample_weight=sample_weight),
        recall_score=recall_score(
            y_true, y_pred, average=average, sample_weight=sample_weight, 
            **scorer_kws),
        precision_score=precision_score(
            y_true, y_pred, average=average, sample_weight=sample_weight,
            **scorer_kws),
        f1_score=f1_score(
            y_true, y_pred, average=average, sample_weight=sample_weight,
            **scorer_kws),
    )
    # Attempt to compute ROC-AUC score if possible
    try:
        if multi_class != 'raise':
            y_score = model.predict_proba(X)
        else:
            # Use y_pred as y_score if predict_proba is not 
            # available or not applicable
            y_score = y_pred 
        scores.roc_auc_score = roc_auc_score(
            y_true, y_score, average=average, multi_class=multi_class, 
            sample_weight=sample_weight, **scorer_kws)
    except Exception as e:
        scores.roc_auc_score = None
        warnings.warn(f"Unable to compute ROC-AUC score: {e}")

    return scores

def adjusted_r2_score(
        y_true, y_pred, X, sample_weight=None, epsilon=1e-7, zero_division="warn"):
    """
    Calculate the adjusted R-squared score, a modification of the R-squared
    score that accounts for the number of predictors in the model.

    The adjusted R-squared increases only if the new term improves the model
    more than would be expected by chance. It decreases when a predictor improves
    the model by less than expected by chance. It is thus especially useful for
    comparing models with different numbers of independent variables.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    X : array-like of shape (n_samples, n_features)
        Input features matrix.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    epsilon : float, default=1e-7
        A small value added to the denominator to prevent division by zero.
    zero_division : {"warn", "error"} or numeric, default="warn"
        Defines how to handle the case when the calculation denominator is zero:
        - "warn": issues a warning and returns `np.nan`
        - "error": raises a `ZeroDivisionError`
        - numeric: returns this value

    Returns
    -------
    float
        The adjusted R-squared score.

    Notes
    -----
    The adjusted R-squared is calculated as:

    .. math::
        1 - (1-R^2)\\frac{n-1}{n-p-1}

    where:
    - :math:`R^2` is the coefficient of determination,
    - :math:`n` is the number of samples,
    - :math:`p` is the number of independent variables.

    Examples
    --------
    >>> from gofast.metrics import adjusted_r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> X = [[1], [2], [3], [4]]
    >>> adjusted_r2_score(y_true, y_pred, X)
    0.948...

    References
    ----------
    - Wikipedia entry for the Coefficient of determination:
      https://en.wikipedia.org/wiki/Coefficient_of_determination

    See Also
    --------
    r2_score : R^2 (coefficient of determination) regression score function.
    mean_squared_error : Mean squared error regression loss.
    """
    # Ensure X is a numeric numpy array
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    
    n, p = X.shape  # Number of observations and independent variables
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    
    denominator = n - p - 1
    
    if denominator <= 0:
        if zero_division == "warn":
            warnings.warn(
                "The calculation of the adjusted R-squared involves division"
                " by zero or a negative number, which can lead to overfitting."
                " Result may not be reliable.", UserWarning)
            adjusted_r2 = np.nan # Return NaN to indicate the calculation isn't reliable
        elif zero_division == "error":
            raise ZeroDivisionError(
                "The calculation of the adjusted R-squared involves division"
                " by zero or a negative number.")
        elif isinstance(zero_division, (int, float)):
            adjusted_r2 = zero_division # Custom Fallback Value:
        else:
            raise ValueError("zero_division must be 'warn', 'error', or a numeric value.")
    else:
        # Adjusted R-squared calculation
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / max(denominator, epsilon)
    
    return adjusted_r2

def _assert_binary_classification_args(y, target_class=None):
    """
    Ensures that the arguments passed for evaluating a binary classification 
    are valid. It checks if 'y' is binary and if the 'target_class' specified 
    exists within 'y'.

    Parameters
    ----------
    y : array-like
        Target labels. Expected to be binary for precision-recall metrics.
    target_class : int, optional
        The class of interest for which to evaluate metrics. If 'y' is binary,
        'target_class' is optional. If specified, it must exist in 'y'.

    Raises
    ------
    ValueError
        - If 'y' is not binary and 'target_class' is not specified.
        - If 'target_class' is specified but does not exist in 'y'.
    TypeError
        - If 'target_class' cannot be converted to an integer.
    """
    unique_labels = np.unique(y)

    # Validate binary nature of 'y'
    if len(unique_labels) != 2:
        if target_class is None:
            raise ValueError(
                "Precision-recall metrics require binary classification, but "
                f"{len(unique_labels)} unique labels were found. Please specify "
                "'target_class' for multiclass data."
            )

    # Validate 'target_class'
    if target_class is not None:
        try:
            target_class = int(target_class)
        except (ValueError, TypeError) as e:
            raise TypeError(
                "Expected 'target_class' to be an integer,"
                f" got {type(target_class).__name__}: {e}"
            )

        if target_class not in unique_labels:
            raise ValueError(
                f"Specified 'target_class' ({target_class}) does"
                " not exist in the target labels."
            )
def precision_recall_tradeoff(
    y_true, 
    y_scores=None, 
    X=None, 
    estimator=None, 
    *,
    cv=None,
    label=None,
    scoring_method="decision_function",
    pos_label=None, 
    sample_weight=None, 
    threshold=None,
    return_scores=False, 
    display_chart=False, 
    **cv_kwargs
):
    """
    Evaluate and visualize the precision-recall tradeoff for classification models.

    This function computes precision, recall, and F1 scores for binary classification
    tasks. It supports direct score inputs or model predictions. If `display_chart`
    is True, it also plots the precision-recall curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_scores : array-like, optional
        Scores or probabilities for the positive class. If None, scores are 
        computed using the `estimator` and `X`.
    X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        Input features matrix. Required if `y_scores` is None.
    estimator : estimator object implementing 'fit', optional
        A trained classifier instance. Required if `y_scores` is None.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. 
        Only used when `y_scores` is None.
    label : int or str, optional
        The label of the positive class. Required for multiclass classification.
    scoring_method : {'decision_function', 'predict_proba'}, default='decision_function'
        The method to call on `estimator` to obtain scores.
    pos_label : int or str, optional
        The label of the positive class. Required if `y_true` contains more 
        than two classes.
    sample_weight : array-like, shape (n_samples,), optional
        Optional array of weights for the samples.
    threshold : float, optional
        A threshold to determine the binary classification from scores. If not set,
        the threshold that maximizes the F1 score is used.
    return_scores : bool, default=False
        If True, return the scores array along with the metrics.
    display_chart : bool, default=False
        If True, display the precision-recall curve.

    Returns
    -------
    scores : Bunch
        A `Bunch` object with the precision, recall, F1 score, and optionally the
        scores array if `return_scores` is True.

    Notes
    -----
    The precision-recall curve plots the tradeoff between precision and recall for
    different threshold values. Precision is defined as the number of true positives
    divided by the number of true positives plus the number of false positives. Recall
    is defined as the number of true positives divided by the number of true positives
    plus the number of false negatives [1]_.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.metrics import precision_recall_tradeoff
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = LogisticRegression()
    >>> clf.fit(X_train, y_train)
    >>> b=precision_recall_tradeoff(y_test, X=X_test, estimator=clf,
    ...                           scoring_method='predict_proba',
    ...                           display_chart=True)
    >>> print(b) 
    
    References
    ----------
    .. [1] Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall
       and ROC curves. Proceedings of the 23rd International Conference on Machine
       learning (pp. 233-240).
    """
    # Validate scoring method
    scoring_method = normalize_string(
        scoring_method, target_strs= ['decision_function', 'predict_proba'], 
        match_method= "contains", return_target_only= True, 
        error_msg= (f"Invalid scoring method '{scoring_method}'. Expected"
                    " 'decision_function' or 'predict_proba'.")
        )
    
    _assert_binary_classification_args(y_true, label)
    if label is not None: 
        # Ensure binary classification
        y_true = np.asarray(y_true == label, dtype=int)  
    
    # Validate and predict scores if not provided
    if y_scores is None:
        if X is None or estimator is None:
            raise ValueError(
                "When 'y_scores' is None, both 'X' and 'estimator' must be provided.")
        if not isinstance (X, np.ndarray): 
            X = np.asarray(X )
        y_scores = cross_val_predict(
            estimator, X, y_true, cv=cv, method=scoring_method, **cv_kwargs)
    
    #y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric =True )
    
    # Handle 'predict_proba' scoring for classifiers
    if scoring_method == 'predict_proba' and y_scores.ndim > 1:
        y_scores = y_scores[:, 1]  # Use probabilities for the positive class
    
    # Calculate precision, recall, and F1 scores
    metrics = MetricFormatter(
        f1_score=f1_score(y_true, y_scores >= (threshold or 0.5)),
        precision_score=precision_score(y_true, y_scores >= (threshold or 0.5)),
        recall_score=recall_score(y_true, y_scores >= (threshold or 0.5))
    )

    # Calculate precision-recall curve
    metrics.precisions, metrics.recalls, metrics.thresholds = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label, sample_weight=sample_weight)
    
    # Display precision-recall chart if requested
    if display_chart:
        display_precision_recall(metrics.precisions, metrics.recalls,
                                       metrics.thresholds)
    
    return y_scores if return_scores else metrics

def display_precision_recall(precisions, recalls, thresholds):
    """
    Displays a precision-recall tradeoff chart for given precision, recall,
    and threshold values, aiding in the visualization of the tradeoff between
    precision and recall across different threshold settings.

    Parameters
    ----------
    precisions : array-like
        An array of precision scores corresponding to various 
        threshold levels.
    recalls : array-like
        An array of recall scores corresponding to various threshold levels.
    thresholds : array-like
        An array of decision thresholds corresponding to the precision 
        and recall scores.

    Notes
    -----
    The precision-recall tradeoff chart is a useful tool for understanding the
    balance between the true positive rate (recall) and the positive 
    predictive value (precision) at various threshold levels. High precision
    relates to a low false positive rate, while high recall relates to a low 
    false negative rate. Adjusting the decision threshold can control the 
    balance between precision and recall.

    This function plots precision and recall as functions of the decision 
    threshold. The precision array must have one more element than the 
    thresholds array, representing the precision at threshold set to 
    `-inf` (recall at 100%).

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import precision_recall_curve
    >>> from gofast.metrics import display_precision_recall
    >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = LogisticRegression()
    >>> clf.fit(X_train, y_train)
    >>> y_scores = clf.decision_function(X_test)
    >>> precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    >>> display_precision_recall(precisions, recalls, thresholds)

    This function produces a line plot with the decision thresholds on the x-axis
    and precision and recall scores on the y-axis, showcasing how precision and recall
    vary with different threshold settings.

    References
    ----------
    .. [1] Davis, J. and Goadrich, M., 2006, June. The relationship between 
       Precision-Recall and ROC curves. In Proceedings of the 23rd international
       conference on Machine learning (pp. 233-240).
    """
    import matplotlib.pyplot as plt

    # Convert inputs to numpy arrays and validate their dimensions
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    thresholds = np.asarray(thresholds)

    if not (precisions.ndim == recalls.ndim == thresholds.ndim == 1):
        raise ValueError("All inputs must be 1-dimensional arrays.")

    if not (len(precisions) == len(recalls) == len(thresholds) + 1):
        raise ValueError("Length of precisions and recalls must be equal"
                         " and one more than the length of thresholds.")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.title("Precision-Recall Tradeoff", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def roc_tradeoff(
    y_true, 
    y_scores=None, 
    X=None, 
    estimator=None, 
    *,
    cv=None,
    pos_label=None, 
    sample_weight=None, 
    return_scores=False, 
    display_chart=False, 
    **cv_kwargs
):
    """
    Evaluates and visualizes the Receiver Operating Characteristic (ROC) curve and
    computes the Area Under the Curve (AUC) for classification models. This
    function is flexible, allowing for direct input of scores or computation
    from a provided estimator and features.

    Parameters
    ----------
    y_true : array-like
        True labels of the classification targets.
    y_scores : array-like, optional
        Scores predicted by the classifier. Required unless `X` and `estimator`
        are provided.
    X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        Input features, required if `y_scores` is not provided.
    estimator : estimator object implementing 'fit', optional
        The classification estimator. Required if `y_scores` is not provided.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    pos_label : int or str, optional
        The label of the positive class.
    sample_weight : array-like, optional
        Optional array of weights to be applied to the samples.
    return_scores : bool, default=False
        Whether to return the computed scores along with the ROC metrics.
    display_chart : bool, default=False
        If True, the ROC curve will be plotted.
    cv_kwargs : dict, optional
        Additional parameters to pass to `cross_val_predict`.

    Returns
    -------
    scores : Bunch
        A dictionary-like object with keys `auc_score`, `fpr`, `tpr`, `thresholds`,
        and optionally `y_scores` (if `return_scores` is True). 

        - `auc_score` : float
            The area under the ROC curve.
        - `fpr` : array
            False Positive Rates.
        - `tpr` : array
            True Positive Rates.
        - `thresholds` : array
            Thresholds on the decision function used to compute `fpr` and `tpr`.

    Notes
    -----
    The ROC curve is a graphical plot that illustrates the diagnostic ability of a
    binary classifier system as its discrimination threshold is varied. The curve
    is created by plotting the True Positive Rate (TPR) against the False Positive
    Rate (FPR) at various threshold settings. The AUC score represents the measure
    of separability. It tells how much the model is capable of distinguishing
    between classes.
    
    The ROC curve is defined by:

    .. math::
        \text{TPR} = \frac{TP}{TP + FN}

    .. math::
        \text{FPR} = \frac{FP}{FP + TN}

    Where:
    - TP is the number of true positives
    - FN is the number of false negatives
    - FP is the number of false positives
    - TN is the number of true negatives
    
    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from gofast.metrics import roc_tradeoff
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> results = roc_tradeoff(y_test, estimator=clf, X=X_test, display_chart=True)
    >>> print(results)
    
    This will display the ROC curve for the RandomForestClassifier on the test data.
    """
    if y_scores is None:
        if X is None or estimator is None:
            raise ValueError("When 'y_scores' is None, both 'X' and"
                             " 'estimator' must be provided.")
        # Check if the estimator is fitted and predict the scores
        check_is_fitted(estimator)
        if hasattr(estimator, "decision_function"):
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, method="decision_function", **cv_kwargs)
        elif hasattr(estimator, "predict_proba"):
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, method="predict_proba", **cv_kwargs)[:, 1]
        elif hasattr ( estimator, 'predict'): 
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, **cv_kwargs)
        else:
            raise ValueError("Estimator must have a 'decision_function',"
                             " 'predict_proba' or 'predict' method .")

    y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric=True)

    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, sample_weight=sample_weight)
    auc_score = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)

    if display_chart:
        # Example of plotting
        display_roc (fpr, tpr, auc_score)

    scores = MetricFormatter(
        auc_score=auc_score, fpr=fpr, tpr=tpr, thresholds=thresholds, 
        y_scores= y_scores )

    return y_scores if return_scores else scores

def display_roc(fpr, tpr, auc_score, *, title=None, figsize=None):
    """
    Visualize the Receiver Operating Characteristic (ROC) curve along with the
    area under the ROC curve (AUC) for a classification model's performance.

    The ROC curve is a graphical representation of the tradeoff between 
    the true positive rate (TPR) and false positive rate (FPR) across a 
    series of thresholds. The AUC provides a scalar measure of the model's 
    ability to distinguish between classes, with a higher value indicating 
    better performance [1]_.

    Parameters
    ----------
    fpr : array-like
        False positive rates for each threshold.
    tpr : array-like
        True positive rates for each threshold.
    auc_score : float
        The area under the ROC curve.
    title : str, optional
        The title for the plot. Defaults to "Receiver Operating Characteristic (ROC)"
        if not specified.
    figsize : tuple, optional
        Figure size as a tuple (width, height). Defaults to (10, 8).

    Raises
    ------
    ValueError
        If `fpr` and `tpr` are not 1-dimensional arrays or if `auc_score` 
        is not a scalar.

    Notes
    -----
    The ROC curve is defined by:

    .. math::
        \text{TPR} = \frac{TP}{TP + FN}

    .. math::
        \text{FPR} = \frac{FP}{FP + TN}

    Where:
    - TP is the number of true positives
    - FN is the number of false negatives
    - FP is the number of false positives
    - TN is the number of true negatives

    Examples
    --------
    >>> from sklearn.metrics import roc_curve, roc_auc_score
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8, 0.7]
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    >>> auc_score = roc_auc_score(y_true, y_scores)
    >>> display_roc(fpr, tpr, auc_score)

    References
    ----------
    .. [1] Fawcett, T. (2006). An introduction to ROC analysis. Pattern 
       Recognition Letters,27(8), 861-874.
    """
    import matplotlib.pyplot as plt
    # Validate inputs
    fpr, tpr, auc_score = map(np.asarray, [fpr, tpr, auc_score])
    if not (fpr.ndim == tpr.ndim == 1):
        raise ValueError("fpr and tpr must be 1-dimensional arrays.")
    try: 
        auc_score= float(auc_score)
    except: 
        if not np.isscalar(auc_score):
            raise ValueError("auc_score must be a scalar value.")

    plt.figure(figsize=figsize or (10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title or "Receiver Operating Characteristic (ROC)", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def evaluate_confusion_matrix(
    y_true, 
    y_pred=None, 
    classifier=None, 
    X=None, 
    *, 
    cv=None, 
    labels=None, 
    sample_weight=None, 
    normalize=False, 
    display=False, 
    cmap='viridis', 
    **cv_kwargs
):
    """
    Evaluates the confusion matrix for a classification model, optionally using
    cross-validation. This function can also normalize and display the confusion
    matrix for visual analysis.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels for the classification tasks.
    y_pred : array-like of shape (n_samples,), optional
        Predicted labels. If None, predictions will be made using the
        classifier and X.
    classifier : object implementing 'fit', optional
        The classifier instance to use if `y_pred` is None.
    X : array-like of shape (n_samples, n_features), optional
        Input features, required if `y_pred` is None.
    cv : int, cross-validation generator or iterable, optional
        Cross-validation strategy, used if `y_pred` is None.
    labels : array-like, optional
        The list of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
    sample_weight : array-like of shape (n_samples,), optional
        Weights of samples.
    normalize : bool, default=False
        If True, the confusion matrix will be normalized.
    display : bool, default=False
        If True, the confusion matrix will be displayed using matplotlib.
    cmap : str or matplotlib Colormap, default='viridis'
        The colormap for displaying the confusion matrix.
    **cv_kwargs : additional keyword arguments
        Additional arguments passed to `cross_val_predict` if `y_pred` is None.

    Returns
    -------
    Bunch
        A Bunch object with the confusion matrix, and optionally the normalized
        confusion matrix if 'normalize' is True.

    Notes
    -----
    The confusion matrix \(C\) is defined as:
    
    .. math:: C_{i, j} = \text{number of observations known to be in group } i
             \text{ and predicted to be in group } j.
    
    For normalized confusion matrix, each element of \(C\) is divided by the sum
    of its row.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.metrics import evaluate_confusion_matrix 
    >>> X, y = make_classification(
    ...    n_samples=1000, n_features=4, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, test_size=0.25, random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> results = evaluate_confusion_matrix(
    ...    y_test, classifier=clf, X=X_test, display=True, normalize=True)
    >>> print(results)
    This will output a Bunch object containing the confusion matrix and display
    the normalized confusion matrix.

    See Also
    --------
    display_confusion_matrix : Function to visualize the confusion matrix.
    sklearn.metrics.confusion_matrix : The confusion matrix computation.
    """

    if y_pred is None:
        if not (X is not None and classifier is not None):
            raise ValueError("Provide 'y_pred' or both 'X' and 'classifier'.")
        check_is_fitted(classifier, 'predict')
        y_pred = cross_val_predict(classifier, X, y_true, cv=cv, **cv_kwargs)
    
    y_true, y_pred =_ensure_y_is_valid(y_true, y_pred, y_numeric=True )
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.fill_diagonal(cm, 0)

    if display:
        display_confusion_matrix(cm, labels=labels, cmap=cmap, normalize=normalize)

    return MetricFormatter(confusion_matrix=cm)

def display_confusion_matrix(cm, labels=None, cmap='viridis', normalize=False):
    """
    Displays a confusion matrix using matplotlib, providing options for 
    normalization and color mapping.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        The confusion matrix to be displayed.
    labels : list of strings, optional
        The labels for the classes corresponding to the indices of the 
        confusion matrix. If None, labels will be inferred from the indices 
        of the confusion matrix.
    cmap : str or matplotlib.colors.Colormap, default='viridis'
        The colormap for the matrix visualization. Can be any valid colormap 
        recognized by matplotlib.
    normalize : bool, default=False
        If True, the confusion matrix will be normalized before display.
        Normalization is performed by dividing each element by the sum of its row.

    Notes
    -----
    The confusion matrix \(C\) is defined as:
    
    .. math:: C_{i, j} = \text{number of observations known to be in group } i
             \text{ and predicted to be in group } j.
    
    For a normalized confusion matrix, each element of \(C\) is divided by 
    the sum of its row to represent the proportion of predictions.

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> display_confusion_matrix(cm, labels=['Class 0', 'Class 1', 'Class 2'], 
    ...                          normalize=True)
    
    This will display a normalized confusion matrix for the provided true 
    and predicted labels with custom class labels.

    See Also
    --------
    evaluate_confusion_matrix : 
        Function to compute and optionally display a confusion matrix.
    matplotlib.pyplot.imshow : 
        Used to display the confusion matrix as an image.

    """
    import matplotlib.pyplot as plt
 
    # Validate cm is a square matrix
    if not isinstance(cm, np.ndarray) or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square matrix (2D numpy array).")
    
    # Validate labels, if provided
    if labels is not None:
        if not isinstance(labels, (list, np.ndarray)) or len(labels) != cm.shape[0]:
            raise ValueError(
                "labels must be a list or array of length matching cm dimensions.")

    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels)) if labels is not None else np.arange(cm.shape[0])
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def mae_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Mean Absolute Error (MAE) with options for detailed error 
    analysis, error scaling, and handling of insignificant errors, encapsulated
    in a Bunch object.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics in addition to MAE.
    scale_errors : bool, optional
        If True, scale errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering an error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the MAE and, optionally, detailed error statistics
        and scaled MAE.

    Notes
    -----
    The MAE is defined as the average of the absolute differences between the 
    predicted values and the true values. It provides a straightforward measure 
    of prediction accuracy for regression models. This implementation allows for 
    the exclusion of insignificant errors (smaller than `epsilon`) from the MAE 
    calculation, and for scaling errors relative to the true values, providing
    a more nuanced error analysis [1]_.

    The mathematical expression of the modified MAE when considering `epsilon` is:

    .. math::

        \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \max(0, |y_{\text{true},i} - 
        y_{\text{pred},i}| - \epsilon)

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B. (2006). Another look at measures of forecast 
          accuracy. International Journal of Forecasting, 22(4), 679-688.
    
    See Also
    --------
    mean_squared_error : Mean Squared Error metric.
    mean_absolute_percentage_error : Mean Absolute Percentage Error metric.

    Examples
    --------
    >>> from gofast.metrics import mae_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = mae_flex(y_true, y_pred)
    >>> print(result.MAE)
    0.5
    >>> result_detailed = mae_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.MAE, result_detailed.min_error)
    0.5 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Function determines a suitable 
    # epsilon value if set to ``auto``.
    epsilon= check_epsilon(epsilon, y_pred )

    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Update errors based on epsilon, if necessary
    # Only consider errors significant if they are greater than epsilon
    significant_errors = np.where(errors > epsilon, errors, 0)
    
    # Compute the modified MAE using significant errors
    mae = np.mean(significant_errors)
    
    result = MetricFormatter(score=mae)
    
    if detailed:
        result.min_error = np.min(significant_errors)
        result.max_error = np.max(significant_errors)
        result.std_error = np.std(significant_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            # Implement warning for zero division if needed
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_errors = np.divide(
                significant_errors, np.abs(y_true) + epsilon, where=y_true != 0)
        
        result.scaled_MAE = np.mean(scaled_errors)
        if detailed:
            result.min_scaled_error = np.min(
                scaled_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_errors, where=y_true != 0)
    
    return result

def mse_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Mean Squared Error (MSE) with options for detailed error 
    analysis, error scaling, and handling of insignificant errors, 
    encapsulated in a Bunch object. This flexible version allows for more 
    nuanced error analysis by providing additional controls over the 
    computation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics (minimum, maximum, and 
        standard deviation of squared errors) in addition to MSE.
    scale_errors : bool, optional
        If True, scale squared errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering a squared error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the MSE and, optionally, detailed error statistics
        and scaled MSE.

    Notes
    -----
    The MSE is defined as the average of the squared differences between the 
    predicted values and the true values. This implementation allows for the 
    exclusion of insignificant errors (smaller than `epsilon` squared) from 
    the MSE calculation, and for scaling errors relative to the true values, 
    providing a more nuanced error analysis.

    The mathematical expression of the modified MSE when considering `epsilon` 
    is:

    .. math::

        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \max(0, (y_{\text{true},i} - 
        y_{\text{pred},i})^2 - \epsilon^2)

    References
    ----------
    .. [1] Gneiting, T., Raftery, A.E. (2007). Strictly Proper Scoring Rules, 
           Prediction, and Estimation. Journal of the American Statistical 
           Association, 102(477), 359-378.

    See Also
    --------
    mae_flex : A flexible version of Mean Absolute Error.
    mean_squared_log_error : Mean Squared Logarithmic Error metric.
    mean_absolute_error : Mean Absolute Error metric.

    Examples
    --------
    >>> from gofast.metrics import mse_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = mse_flex(y_true, y_pred)
    >>> print(result.score)
    0.375
    >>> result_detailed = mse_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.score, result_detailed.min_error)
    0.375 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Assuming this function determines a suitable epsilon value
    epsilon= check_epsilon(epsilon, y_pred )
    
    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the squared errors
    squared_errors = (y_true - y_pred) ** 2
    
    # Update squared errors based on epsilon, if necessary
    # Only consider errors significant if they are greater
    # than epsilon squared
    significant_squared_errors = np.where(
        squared_errors > epsilon**2, squared_errors, 0)
    
    # Compute the modified MSE using significant squared errors
    mse = np.mean(significant_squared_errors)
    
    result = MetricFormatter(score=mse)
    
    if detailed:
        result.min_error = np.min(significant_squared_errors)
        result.max_error = np.max(significant_squared_errors)
        result.std_error = np.std(significant_squared_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_squared_errors = np.divide(
                significant_squared_errors, (
                    np.abs(y_true) + epsilon)**2, where=y_true != 0)
        
        result.scaled_score = np.mean(scaled_squared_errors)
        if detailed:
            result.min_scaled_error = np.min(
                scaled_squared_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_squared_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_squared_errors, where=y_true != 0)
    return result

def rmse_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Root Mean Squared Error (RMSE) with options for detailed error
    analysis, error scaling, and handling of insignificant errors, 
    encapsulated in a Bunch object. This flexible version allows for more
    nuanced error analysis by providing additional controls over the
    computation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics (minimum, maximum, and 
        standard deviation of squared errors) in addition to RMSE.
    scale_errors : bool, optional
        If True, scale squared errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering a squared error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the RMSE and, optionally, detailed error statistics
        and scaled RMSE.

    Notes
    -----
    The RMSE is defined as the square root of the average of the squared 
    differences between the predicted values and the true values. This 
    implementation allows for the exclusion of insignificant errors 
    (smaller than `epsilon` squared) from the RMSE calculation, and for 
    scaling errors relative to the true values, providing a more nuanced 
    error analysis [1]_.

    The mathematical expression of the modified RMSE when considering 
    `epsilon` is:

    .. math::

        \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \max(0, (y_{\text{true},i} - 
        y_{\text{pred},i})^2 - \epsilon^2)}

    References
    ----------
    .. [1] Chai, T. and Draxler, R.R. (2014). Root mean square error (RMSE) or mean 
          absolute error (MAE)? – Arguments against avoiding RMSE in the literature.
          Geoscientific Model Development, 7, 1247–1250.

    See Also
    --------
    mae_flex : A flexible version of Mean Absolute Error.
    mse_flex : A flexible version of Mean Squared Error.
    sklearn.metrics.mean_squared_error : Mean Squared Error metric.

    Examples
    --------
    >>> from gofast.metrics import rmse_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = rmse_flex(y_true, y_pred)
    >>> print(result.score)
    0.612...
    >>> result_detailed = rmse_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.score, result_detailed.min_error)
    0.612... 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Assuming this function determines a suitable epsilon value
    epsilon= check_epsilon(epsilon, y_pred )
    
    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the squared errors
    squared_errors = (y_true - y_pred) ** 2
    
    # Update squared errors based on epsilon, if necessary
    # Only consider errors significant if they are greater than epsilon squared
    significant_squared_errors = np.where(
        squared_errors > epsilon**2, squared_errors, 0)
    
    # Compute the RMSE using significant squared errors
    rmse = np.sqrt(np.mean(significant_squared_errors))
    
    result = MetricFormatter(title="RMSE Results",  score=rmse)
    
    if detailed:
        result.min_error = np.min(significant_squared_errors)
        result.max_error = np.max(significant_squared_errors)
        result.std_error = np.std(significant_squared_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_squared_errors = np.divide(
                significant_squared_errors, (
                    np.abs(y_true) + epsilon)**2, where=y_true != 0)
        
        result.scaled_RMSE = np.sqrt(np.mean(scaled_squared_errors))
        if detailed:
            result.min_scaled_error = np.min(
                scaled_squared_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_squared_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_squared_errors, where=y_true != 0)
    
    return result

def r2_flex(
    y_true, y_pred, *, 
    epsilon=1e-15, 
    adjust_for_n=False,
    n_predictors=None  
    ):
    """
    Compute the R-squared (Coefficient of Determination) with options to handle
    edge cases and to adjust for the number of predictors. 
    
    Funtion offers enhanced flexibility by allowing for a minimum variance 
    threshold (epsilon) and providing an option to calculate an adjusted 
    R-squared value, making it more robust against overfitting when using 
    models with multiple predictors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    epsilon : float, default=1E-15
        A small value to prevent division by zero in the total sum of squares
        calculation, ensuring numerical stability. This value is used as the
        minimum allowable variance.
    adjust_for_n : bool, default=False
        If True, the adjusted R-squared value is computed to account for the
        number of predictors in the model. This can provide a more accurate
        measure of the model's explanatory power when comparing models with
        different numbers of predictors.
    n_predictors : int, optional
        The number of predictors used in the model. This parameter is required
        if `adjust_for_n` is True to properly calculate the adjusted R-squared.

    Returns
    -------
    Bunch
        An object containing the R-squared value and, if `adjust_for_n` is True,
        the adjusted R-squared value.

    Notes
    -----
    The standard R-squared is calculated as 1 minus the ratio of the residual
    sum of squares (RSS) to the total sum of squares (TSS). The adjusted
    R-squared additionally considers the number of predictors in the model to
    adjust for the penalty of model complexity.

    Adjusted R-squared is especially useful in multiple regression models to
    prevent the R-squared value from artificially inflating when additional
    predictors are added to the model.

    R-squared is defined as:

    .. math::
        R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    Where:

    - \(SS_{res}\) is the sum of squares of residuals.
    - \(SS_{tot}\) is the total sum of squares.
    
    Adjusted R-squared is calculated as:

    .. math::
        \text{Adjusted } R^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}

    Where:

    - \(n\) is the number of samples.
    - \(p\) is the number of predictors.

    The `epsilon` parameter helps to ensure \(SS_{tot}\) is not zero by providing
    a minimum value, thereby preventing undefined R-squared values.

    References
    ----------
    .. [1] James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013). 
           An Introduction to Statistical Learning. New York: Springer.

    See Also
    --------
    mae_flex : Mean Absolute Error with flexibility.
    mse_flex : Mean Squared Error with flexibility.
    rmse_flex : Root Mean Squared Error with flexibility.

    Examples
    --------
    >>> from gofast.metrics import r2_flex
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2.5, 0.5, 2, 8])
    >>> result = r2_flex(y_true, y_pred)
    >>> print(result.score)
    -0.4576271186440677...
    >>> result_adjusted = r2_flex(y_true, y_pred, adjust_for_n=True, n_predictors=1)
    >>> print(result_adjusted.adjusted_score)
    -1.1864406779661016...
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Validation for epsilon
    epsilon= check_epsilon(epsilon, y_pred )
  
    # Calculate R-squared
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / max (sst, epsilon)) # Prevent division by zero
    
    result = MetricFormatter(title="R2 Results", score=r2)
    
    if n_predictors is None: 
        n_predictors = max (1, epsilon)
        
    # Check if n_predictors is a numeric type, including NumPy numeric types
    if isinstance(n_predictors, (int, float, np.integer, np.floating)):
        n_predictors = int(n_predictors)  # Convert to int (handles float and np.floating)
    else:
        raise ValueError("n_predictors must be a numeric value.")

    # Adjusted R-squared calculation if requested 
    # and if n_predictors is specified
    if adjust_for_n:
        n = len(y_true)
        p = n_predictors  # Use the specified number of predictors
        if n == 1 or n - p - 1 == 0:
            warnings.warn ("Adjustment for n is not meaningful with only"
                           " one sample or one predictor.")
        elif n <= p:
            warnings.warn("Cannot compute adjusted R2 with number of"
                          " samples <= number of predictors.")
        else:
            result.adjusted_score = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
    elif adjust_for_n and n_predictors is None:
        warnings.warn("Number of predictors must be specified to adjust R2.")
    
    return result

def mean_absolute_percentage_error(
    y_true, y_pred, *, 
    epsilon=1e-15, 
    zero_division='warn', 
    sample_weight=None, 
    multioutput='uniform_average'
):
    """
    Compute the Mean Absolute Percentage Error (MAPE).

    The MAPE measures the average of the percentage errors by which
    forecasts of a model differ from actual values of the quantity being
    forecasted. It is commonly used in financial and operational forecasting
    to express the predictive accuracy of a model as a percentage.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    epsilon : float, optional
        Small value to avoid division by zero, default is 1e-15.
    zero_division : {'warn', 'raise'}, optional
        Action to perform if zero is encountered in `y_true`. 
        'warn' issues a warning, and 'raise' raises a ValueError, 
        default is 'warn'.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Defines aggregating of multiple output scores. 
        'uniform_average' to average an array of errors or 'raw_values'
        to return a full error array, default is 'uniform_average'.

    Returns
    -------
    float or ndarray
        The MAPE value. If `multioutput` is 'raw_values', then MAPE is
        returned for each output separately. If `multioutput` is
        'uniform_average', the average MAPE across all outputs is returned.

    Notes
    -----
    The MAPE is defined as:

    .. math:: 
        
        \text{MAPE} = \left( \frac{1}{n} \sum_{i=1}^{n} 
              \left| \frac{y_{\text{true}_i} - y_{\text{pred}_i}}
              {y_{\text{true}_i}} \right| \right) \times 100

    The `epsilon` parameter is used to modify the denominator as 
    \(y_{\text{true}_i} + \epsilon\) to ensure numerical stability. 
    This is especially useful when `y_true` contains zeros.

    References
    ----------
    .. [1] Hyndman, R.J., & Koehler, A.B. (2006). Another look at measures
       of forecast accuracy. International Journal of Forecasting, 22(4), 679-688.

    See Also
    --------
    mean_squared_error : Compute mean square error.
    mean_absolute_error : Compute mean absolute error.
    
    Examples
    --------
    >>> from gofast.metrics import mean_absolute_percentage_error
    >>> y_true = np.array([3, 5, 2.5, 7])
    >>> y_pred = np.array([2.5, 5.5, 2, 8])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    12.8
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, multi_output=True)
    
    # Handle zero division
    y_true = handle_zero_division(
        y_true, zero_division= zero_division, 
        replace_with= np.nan,  
        metric_name="MAPE"
    )
    # Calculate the absolute percentage error, 
    # adding epsilon to avoid division by zero
    epsilon= check_epsilon(epsilon, y_true )
    ape = np.abs((y_true - y_pred) / (y_true + epsilon))
    
    # Handling sample weights if provided
    if sample_weight is not None:
        sample_weight = validate_sample_weights(
            sample_weight, y=ape,normalize= True )
        # sample_weight = np.asarray(sample_weight)
        weighted_ape = np.average(ape, weights=sample_weight)
    else:
        weighted_ape = np.mean(ape, axis =0 )
    
    # Scale the result by 100 to get a percentage
    mape = weighted_ape * 100
    multioutput = validate_multioutput(multioutput )
    if multioutput == 'raw_values':
        return mape
    elif multioutput == 'uniform_average':
        # Return the average MAPE if multioutput is set to 'uniform_average'
        return np.average(mape) if mape.ndim != 0 else mape
    
def explained_variance_score(
    y_true, y_pred, *, 
    sample_weight=None, 
    multioutput='uniform_average',
    epsilon=1e-8,
    zero_division='warn'
):
    """
    Compute the Explained Variance Score (EVS) for regression models.

    The EVS measures the proportion of variance in the dependent variable 
    that is predictable from the independent variable(s). It is an indicator
    of the goodness of fit of a model, with 1 indicating perfect prediction 
    and 0 indicating that the model performs no better than a trivial baseline
    that simply predicts the mean of the dependent variable [1]_.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values from the regression model.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample, default is None.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Defines aggregating of multiple output values. 'uniform_average' to
        average all outputs (default). 'raw_values' to return a full set of scores.
    epsilon : float, optional
        Tolerance for variance near zero, preventing division by zero errors,
        default is 1e-8.
    zero_division : {'warn', 'raise'}, optional
        Behavior when the variance of `y_true` is zero or near zero. 'warn'
        returns 0.0 after printing a warning; 'raise' raises a ZeroDivisionError,
        default is 'warn'.

    Returns
    -------
    float or ndarray
        The explained variance score, a single float if multioutput is 
        'uniform_average' or an array if 'raw_values'.

    Notes
    -----
    The explained variance score is defined as:

    .. math:: EVS = 1 - \\frac{Var(y_{true} - y_{pred})}{Var(y_{true})}

    Where:
    - :math:`Var` is the variance,
    - :math:`y_{true}` are the true target values,
    - :math:`y_{pred}` are the predicted target values.

    The score can be negative if the model is arbitrarily worse than the simple
    mean predictor.

    Examples
    --------
    >>> from gofast.metrics import explained_variance_score
    >>> y_true = np.array([3, 5, 2.5, 7])
    >>> y_pred = np.array([2.5, 5.5, 2, 8])
    >>> explained_variance_score(y_true, y_pred)
    0.957...

    References
    ----------
    .. [1] G. Hughes. On the mean accuracy of statistical pattern recognizers.
           IEEE Transactions on Information Theory, 14(1):55–63, January 1968.

    See Also
    --------
    sklearn.metrics.mean_squared_error : Mean squared error regression loss.
    sklearn.metrics.mean_absolute_error : Mean absolute error regression loss.
    sklearn.metrics.r2_score : 
        R^2 (coefficient of determination) regression score function.
    """

    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, multi_output=True)
    
    # Compute weights only once if provided
    weights = sample_weight if sample_weight is not None else np.ones_like(y_true)
    # Check epsilon 
    epsilon= check_epsilon(epsilon, y_pred , scale_factor=1e-8)
    # Compute the weighted mean of y_true directly
    mean_y_true = np.average(y_true, weights=weights)
    
    # Compute variance residuals and total variance using weighted averages
    var_res = np.average((y_true - y_pred) ** 2, weights=weights)
    var_true = np.average((y_true - mean_y_true) ** 2, weights=weights)
    
    # Adjust variances near zero according to zero_division strategy
    if zero_division == 'warn' and np.isclose(var_true, 0, atol=epsilon):
        warnings.warn("Variance of `y_true` is near zero; returning 0.0"
                      " for indeterminate forms.", RuntimeWarning )
        var_true_adjusted = np.inf
    elif zero_division == 'raise' and np.isclose(var_true, 0, atol=epsilon):
        raise ZeroDivisionError("Variance of `y_true` is too close to zero,"
                                " causing division by zero.")
    else:
        var_true_adjusted = np.where(
            np.isclose(var_true, 0, atol=epsilon), np.nan, var_true)
    
    explained_variance = 1 - var_res / var_true_adjusted

    # Handle multioutput scenarios efficiently
    multioutput= validate_multioutput(multioutput)
    if multioutput == 'raw_values':
        return explained_variance
    elif multioutput == 'uniform_average':
        if np.isnan(explained_variance).all():
            return np.nan
        return np.nanmean(explained_variance)

def median_absolute_error(
    y_true, y_pred, *, 
    sample_weight=None,
    multioutput='uniform_average',
    nan_policy='propagate', 
):
    """
    Compute the Median Absolute Error (MedAE) between true and predicted values,
    optionally handling sample weights, multiple outputs, and missing values.

    MedAE is a robust measure of the accuracy of a prediction model. It represents
    the median of the absolute differences between the predicted and actual values,
    providing a measure of the central tendency of the prediction error that is
    less sensitive to outliers than the mean absolute error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample, default is None.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Defines aggregating of multiple output values. 'uniform_average' to
        average all outputs (default). 'raw_values' to return a full set of
        scores for each output.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', ignore NaN values in `y_true` and `y_pred`. Useful for 
        datasets with missing values.
        
    Returns
    -------
    float or ndarray
        The median absolute error or an array of errors for each output if
        `multioutput='raw_values'`. If `multioutput='uniform_average'`, the
        average MedAE across all outputs is returned.

    Notes
    -----
    The MedAE is defined as:

    .. math:: \text{MedAE} = \text{median}(\left| y_{true} - y_{pred} \right|)

    For weighted MedAE, each absolute error is multiplied by its corresponding
    sample weight before computing the median.

    When `ignore_nan` is True, NaN values in `y_true` and `y_pred` are ignored,
    allowing the calculation to proceed with valid pairs only.

    Examples
    --------
    >>> from gofast.metrics import median_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> median_absolute_error(y_true, y_pred)
    0.5

    References
    ----------
    Rousseeuw, P.J., Leroy, A.M., 1987. Robust Regression and Outlier Detection.
    John Wiley & Sons, Inc.

    See Also
    --------
    mean_absolute_error : Mean absolute error regression loss.
    mean_squared_error : Mean squared error regression loss.
    mean_absolute_percentage_error : Mean absolute percentage error.
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, allow_nan=True, multi_output=True)
  
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
    nan_policy, y_true, y_pred, sample_weights=sample_weight 
    )
    # If sample_weight was provided and thus returned, update sample_weight
    # variable, else keep it unchanged
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
        
    absolute_errors = np.abs(y_true - y_pred)
    
    if sample_weight is not None:
        sample_weight = validate_sample_weights(# Normalize sample weights
            sample_weight, y=absolute_errors, normalize= True  )
    
        # Compute weighted median using np.average with quantiles
        def weighted_median(data, weights):
            sorter = np.argsort(data)
            sorted_data = data[sorter]
            sorted_weights = weights[sorter]
            cumulative_weights = np.cumsum(sorted_weights)
            # Find the position where cumulative weight equals or exceeds 0.5
            median_idx = np.searchsorted(cumulative_weights, 0.5)
            return sorted_data[median_idx]
        
        medAE = np.apply_along_axis(
            weighted_median, 0, absolute_errors, sample_weight)
    else:
        medAE = np.median(absolute_errors, axis=0)
    
    multioutput= validate_multioutput(multioutput )

    return medAE if multioutput == 'raw_values' else np.average(medAE)

def max_error_score(
    y_true, y_pred,  *, 
    sample_weight=None, 
    multioutput='uniform_average',
    nan_policy='propagate', 
):
    """
    Compute the maximum absolute error between true and predicted values,
    with support for sample weights, multi-output responses, and handling 
    of NaN values.

    The maximum error metric captures the worst case error between the predicted
    value and the true value. It is particularly useful in applications where
    the maximum possible error has special significance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Specifies how to aggregate errors for multiple output predictions.
        'uniform_average' calculates an average of all output errors.
        'raw_values' returns a full set of errors for each output.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', NaN values in `y_true` or `y_pred` are ignored.

    Returns
    -------
    float or ndarray
        The maximum absolute error. If `multioutput` is 'raw_values', an array
        of maximum errors is returned, one for each output. If
        `multioutput` is 'uniform_average', the average of all maximum errors
        across outputs is returned.

    Notes
    -----
    The maximum error is defined as:

    .. math:: \text{Max Error} = \max(\left| y_{true} - y_{pred} \right|)

    If `sample_weight` is provided, the maximum error is calculated after
    applying the weights to the absolute errors.

    Examples
    --------
    >>> from gofast.metrics import max_error_score
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    >>> max_error_score(y_true, y_pred)
    1.5

    References
    ----------
    .. [1] Willmott, C.J., & Matsuura, K. (2005). Advantages of the mean absolute
           error (MAE) over the root mean square error (RMSE) in assessing average
           model performance. Climate Research, 30, 79-82.

    See Also
    --------
    sklearn.metrics.mean_absolute_error : Compute the mean absolute error.
    sklean.metrics.mean_squared_error : Compute the mean squared error.
    mean_absolute_percentage_error : Compute the mean absolute percentage error.
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, allow_nan=True, multi_output=True)
    
    # Filter out NaN values from both y_true and y_pred
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights=sample_weight 
    )
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    errors = np.abs(y_true - y_pred)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        # Apply sample weights to errors
        weighted_errors = errors * sample_weight
        max_error_value = np.max(weighted_errors)
    else:
        max_error_value = np.max(errors)
    
    multioutput= validate_multioutput(multioutput)
    
    if multioutput == 'raw_values':
        return max_error_value
    elif multioutput == 'uniform_average':
        # For uniform average, the behavior is the same as raw_values 
        # since max_error is a scalar
        # This branch is kept for consistency with other metrics' API
        return np.average(max_error_value)
  
def mean_poisson_deviance(
    y_true, y_pred, *, 
    sample_weight=None, 
    epsilon=1e-8,
    nan_policy='propagate', 
    zero_division='warn',
    multioutput='uniform_average'
):
    """
    Compute the Mean Poisson Deviance between true and predicted values,
    with support for sample weights, handling of NaNs, and multi-output targets.

    The Mean Poisson Deviance is a metric that quantifies the goodness of fit
    for models predicting counts or rates. It is especially useful in Poisson 
    regression and other count-based model assessments [1]_.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample.
    epsilon : float, optional
        Small value added to predictions to avoid taking the log of zero,
        default is 1e-8.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', ignore NaN values in `y_true` and `y_pred`.
    zero_division : {'warn', 'ignore'}, optional
        Behavior when division by zero is encountered in the deviance 
        calculation, default is 'warn'.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Specifies how to aggregate errors for multiple output predictions,
        default is 'uniform_average'.

    Returns
    -------
    float or ndarray
        The Mean Poisson Deviance score. If `multioutput` is 'raw_values',
        an array of deviance scores for each output is returned. If
        `multioutput` is 'uniform_average', the average of all deviance scores
        across outputs is returned.

    Notes
    -----
    The Mean Poisson Deviance is defined as:

    .. math:: \frac{1}{n} \sum_{i=1}^{n} 2 \left( y_{\text{pred}_i} - 
              y_{\text{true}_i} \log{\frac{y_{\text{pred}_i}}
              {\max(y_{\text{true}_i}, \epsilon)}} \right)

    This metric is suitable for models where the target variable follows a
    Poisson distribution, typically involving counting processes.

    Examples
    --------
    >>> from gofast.metrics import mean_poisson_deviance
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_poisson_deviance(y_true, y_pred)
    0.5

    References
    ----------
    .. [1] Cameron, A.C., & Trivedi, P.K. (1998). Regression Analysis of Count Data.
           Cambridge University Press.

    See Also
    --------
    mean_squared_error : Compute the mean squared error.
    mean_absolute_error : Compute the mean absolute error.
    mean_absolute_percentage_error : Compute the mean absolute percentage error.
    """
    
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, allow_nan=True, multi_output=True)

    # Filter out NaN values from both y_true and y_pred
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights=sample_weight 
    )
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    if zero_division not in ['warn', 'ignore']:
        raise ValueError("zero_division must be either 'warn' or 'ignore'")
    
    # Adjust predictions to ensure they are positive
    epsilon= check_epsilon(epsilon, y_pred, scale_factor=1e-8)
    y_pred = np.maximum(y_pred, epsilon)
    
    with np.errstate(divide=zero_division, invalid=zero_division):
        deviance_components = y_pred - y_true * np.log(
            y_pred / np.maximum(y_true, epsilon))
        deviance = 2 * deviance_components
        if zero_division == 'warn' and np.isinf(deviance).any():
            warnings.warn("Encountered division by zero in"
                          " `mean_poisson_deviance` calculation.")
    
    # Handling multioutput scenarios
    multioutput= validate_multioutput(multioutput)
    if multioutput == 'raw_values':
        mean_deviance = deviance
    elif multioutput == 'uniform_average':
        # Compute the average of the deviance scores across all outputs
        if sample_weight is not None:
            mean_deviance = np.average(deviance, weights=sample_weight, axis=0)
        else:
            mean_deviance = np.mean(deviance, axis=0)
        # Further average if multiple outputs    
        mean_deviance = np.average(mean_deviance)  
  
    return mean_deviance

def mean_gamma_deviance(
    y_true, y_pred, *, 
    sample_weight=None, 
    epsilon=1e-8,
    clip_value=None,
    nan_policy='propagate',
    zero_division='warn',
    multioutput='uniform_average'
):
    """
    Compute the Mean Gamma Deviance, a measure of fit for models predicting
    positive continuous quantities that are assumed to follow a gamma
    distribution.

    Function is particularly useful for models where the target variable
    is strictly positive and the variance of each measurement is proportional
    to its mean [1]_.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values, must be non-negative.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample.
    epsilon : float, optional
        Small value to offset predictions to ensure positivity and stability
        in division and logarithmic operations, default=1e-8.
    clip_value : float, optional
        Minimum value to clip predictions, ensuring no prediction is below
        this value. Used to avoid undefined logarithmic operations and division
        by zero. If None, epsilon is used as the minimum value.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', samples with NaN values in `y_true` or `y_pred` are ignored.
    zero_division : {'warn', 'ignore'}, optional
        Behavior when a zero division occurs, default='warn'.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Aggregation method for multiple outputs, default='uniform_average'.

    Returns
    -------
    float or ndarray
        The computed mean gamma deviance. If `multioutput` is 'raw_values',
        an array of deviance scores for each output is returned. If
        `multioutput` is 'uniform_average', the average of all deviance scores
        across outputs is returned.

    Notes
    -----
    The mean gamma deviance is defined as:

    .. math:: \frac{1}{n} \sum_{i=1}^{n} 2 \left( \log{\frac{y_{\text{true}_i}}
              {y_{\text{pred}_i}}} + \frac{y_{\text{pred}_i} - y_{\text{true}_i}}
              {y_{\text{pred}_i}} \right)

    This metric assumes both the predictions and true values are strictly positive.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.metrics import mean_gamma_deviance
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_gamma_deviance(y_true, y_pred)
    0.087

    References
    ----------
    .. [1] McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models.
          Chapman and Hall/CRC.

    See Also
    --------
    mean_squared_error : Compute mean squared error.
    mean_absolute_error : Compute mean absolute error.
    mean_poisson_deviance : Compute mean Poisson deviance.
    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True,
        allow_nan=True, multi_output=True
    )
    # Clip y_pred to a minimum of epsilon or clip_value if specified
    if clip_value is not None:
        y_pred = np.clip(y_pred, clip_value, np.max(y_pred))
    else:
        y_pred = np.maximum(y_pred, epsilon)
    
    # Filter out NaN values from both y_true and y_pred
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights=sample_weight 
    )
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    with np.errstate(divide=zero_division, invalid=zero_division):
        gamma_deviance = 2 * (np.log(y_true / y_pred) + (y_pred - y_true) / y_pred)
        
        if zero_division == 'warn' and (
                np.isinf(gamma_deviance).any() or np.isnan(gamma_deviance).any()):
            warnings.warn("Encountered zero division or invalid value in"
                          " `mean_gamma_deviance` calculation.")
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        mean_deviance = np.average(gamma_deviance, weights=sample_weight)
    else:
        mean_deviance = np.mean(gamma_deviance)
    
    # Handling multioutput scenarios
    multioutput = validate_multioutput(multioutput)
    if multioutput == 'raw_values':
        return mean_deviance
    elif multioutput == 'uniform_average':
        if np.ndim(mean_deviance) > 0:
            return np.average(mean_deviance)
        else:
            return mean_deviance

def mean_absolute_deviation(
    y_true, y_pred, *, 
    sample_weight=None, 
    epsilon=1e-8,
    nan_policy='propagate',
    zero_division='warn',
    multioutput='uniform_average'
):
    """
    Compute the Mean Absolute Deviation (MADev) between true and predicted 
    values, offering flexibility in handling NaNs, applying sample weights, 
    and controlling division by near-zero values.

    Parameters
    ----------
    y_true : array-like
        True target values. It can be a list, numpy array, or a pandas series.
    y_pred : array-like
        Predicted target values, must have the same shape as y_true.
        Predictions from a model or any estimations corresponding to y_true.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample. If specified, the MAD calculation 
        will be weighted, giving more importance to certain samples. Defaults
        to None, treating all samples equally.
    epsilon : float, optional
        A small positive value to prevent division by zero when calculating
        scaled MAD or handling edge cases. Defaults to 1e-8.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', NaN values in both y_true and y_pred are ignored during the
        MAD computation. This allows for handling datasets with missing values.
    zero_division : {'warn', 'ignore'}, optional
        Strategy for handling division by zero errors during computation. 'warn'
        issues a warning, while 'ignore' suppresses warning and proceeds with
        the calculation, potentially resulting in infinite or NaN values.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Determines how to aggregate the outputs. 'raw_values' returns an array
        with the MAD for each output. 'uniform_average' computes the average of
        these values, providing a single summary statistic. Defaults to 
        'uniform_average'.

    Returns
    -------
    float or ndarray
        The Mean Absolute Deviation (MADev). If `multioutput` is 'raw_values',
        returns an array of MAD values for each output variable. If 
        'uniform_average', returns the overall average MADev [1]_.

    Notes
    -----
    The Mean Absolute Deviation is a measure of dispersion around the mean, 
    showing how much the data varies from the average value:

    .. math:: \text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |y_{\text{true},i} - y_{\text{pred},i}|

    It is less sensitive to outliers than variance-based measures, making it
    a robust statistic for indicating prediction accuracy and variability in
    data analysis and forecasting models [2]_.

    Examples
    --------
    >>> from gofast.metrics import mean_absolute_deviation
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_deviation(y_true, y_pred)
    0.5

    See Also
    --------
    mean_squared_error : Compute mean squared error for regression models.
    mean_absolute_error : Compute mean absolute error between true and predicted values.
    r2_score : Compute the coefficient of determination, indicating prediction accuracy.

    References
    ----------
    .. [1] Rousseeuw, P.J., Croux, C. (1993). "Alternatives to the Median Absolute
           Deviation," Journal of the American Statistical Association, 88(424),
           1273-1283. This paper discusses robust measures of scale and the
           MAD's role as a highly robust estimator of scale.
    .. [2] Lehmann, E. L. (1998). Nonparametrics: Statistical Methods Based on Ranks.
           Prentice Hall.
    """
    # Validate and prepare the inputs
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True,
        allow_nan=True, multi_output=True
    )
    y_true = handle_zero_division(
        y_true, zero_division=zero_division, epsilon=epsilon, metric_name="MADev" )
    # Filter out NaN values from both y_true and y_pred
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(nan_policy, 
        y_true, y_pred, sample_weights=sample_weight 
    )
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
     
    epsilon = check_epsilon(epsilon, y_true, y_pred, scale_factor=1e-8) 
    
    # Calculate the absolute deviations
    deviations = np.abs(y_true - y_pred)
    
    if sample_weight is not None:
        # Apply sample weights
        weighted_deviations = deviations * sample_weight
        sum_weights = np.sum(sample_weight) + epsilon  
        madev = np.sum(weighted_deviations) / sum_weights
    else:
        madev = np.mean(deviations)
    
    # Handle multioutput scenarios
    multioutput = validate_multioutput(multioutput)
    # This conditional is redundant for MADev but included for API consistency
    # Average MADev across all outputs, if y_true/y_pred were multi-dimensional
    return madev if multioutput == 'raw_values' else np.mean(madev)  

def madev_flex(
    data, *, 
    sample_weight=None,
    nan_policy='propagate',
    epsilon=1e-8, 
    axis=None, 
    detailed=False, 
    scale_errors=False, 
    zero_division='warn'
):
    """
    Compute the flexible Mean Absolute Deviation (MADev) with options for
    handling NaN values, scaling errors, and providing detailed statistics.
    This version allows for more nuanced analysis and robust handling of
    data, encapsulated in a Bunch object.

    Parameters
    ----------
    data : array-like
        The dataset for which the Mean Absolute Deviation (MADev) is to be 
        computed. This can be a list, numpy array, or a pandas DataFrame/Series.
        The data should represent a sample or population from which to measure
        central dispersion.

    sample_weight : array-like of shape (n_samples,), optional
        Weights associated with the elements of `data`, representing the importance
        or frequency of each data point. Useful in weighted statistical analyses
        where some observations are considered more significant than others. 
        Default is None, where each data point has equal weight.

    nan_policy : str, {'omit', 'propagate', 'raise'}, defaut = 'propagate'
        If set to 'omit', the function will ignore NaN (Not a Number) values 
        in the `data`. This is particularly useful when dealing with datasets
        that have missing values, ensuring they do not affect the computation 
        of the MADev. Default is ``propagate`` which returns the input data 
        without changes. 'raise' throws an error if NaNs are detected.

    epsilon : float, optional
        A small positive constant added to the denominator to prevent division
        by zero when calculating scaled deviations or when the mean of 
        `data` is zero. Default value is ``1e-8``, which is sufficiently small
        to not distort the calculations.

    axis : int, optional
        The axis along which to compute the mean absolute deviation. For a 2D 
        array, ``axis=0`` computes the deviation column-wise, while ``axis=1``
        row-wise. Default is None, flattening the `data` and computing the 
        deviation for the entire dataset.

    detailed : bool, optional
        If True, the function returns detailed statistics including the minimum,
        maximum, and standard deviation of the absolute deviations. This provides
        a deeper insight into the variability and dispersion of the dataset.
        Default is False.

    scale_errors : bool, optional
        If True, the absolute deviations are scaled by the mean value of `data`,
        resulting in a relative measure of dispersion. This is useful for comparing
        the variability of datasets with different scales. Default is False.

    zero_division : {'warn', 'ignore'}, optional
        Specifies how to handle situations where division by zero might occur
        during the computation. 'warn' will issue a warning, and 'ignore' will 
        suppress the warning and proceed with the calculation. 
        Default is ``'warn'``.

    Returns
    -------
    Bunch
        A Bunch object containing the computed Mean Absolute Deviation (MADev) 
        and,optionally, detailed and scaled error statistics based on the 
        specified parameters. This structured output allows for easy access 
        to the results of the computation.

    Notes
    -----
    The Mean Absolute Deviation (MADev) is a robust measure of variability that
    quantifies the average absolute dispersion of a dataset around its mean.
    Unlike variance or standard deviation, MADev is not influenced as heavily
    by extreme values, making it a reliable measure of spread, especially in
    datasets with outliers [1]_.

    This flexible implementation of MADev allows for detailed and scaled analysis,
    handling of missing values, and application to multi-dimensional datasets,
    enhancing its utility for a broad range of statistical and data analysis tasks.

    The Mean Absolute Deviation is defined as:

    .. math:: \text{MADev} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|

    where :math:`\bar{x}` is the mean of the data and :math:`n` is the number
    of observations.

    Examples
    --------
    >>> from gofast.metrics import madev_flex
    >>> data = [1, 2, 3, 4, 5]
    >>> result=madev_flex(data)
    >>> result.score
    1.2

    >>> result= madev_flex(data, detailed=True)
    >>> print(result)
    MADev         : 1.2
    max_deviation : 2.0
    min_deviation : 0.0
    std_deviation : 0.7483314773547883

    >>> result=madev_flex(data, scale_errors=True)
    >>> result.scaled_MADev 
    0.4

    References
    ----------
    .. [1] Lehmann, E. L. (1998). Nonparametrics: Statistical Methods Based on Ranks.
           Prentice Hall.

    See Also
    --------
    numpy.mean : Compute the arithmetic mean along the specified axis.
    numpy.nanmean : Compute the arithmetic mean along the specified axis,
                    ignoring NaNs.
    """
    data = np.asarray(data)
    if not _is_numeric_dtype(data): 
        raise TypeError("MADev computation expects `data` to be numeric.")
        
    # Handle NaN values by filtering them out
    data, *opt_sample_weight= validate_nan_policy(
        nan_policy, data, sample_weights= sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    # Calculate the mean, taking into account whether NaN values should be ignored
    mean = np.nanmean(data, axis=axis) if nan_policy=='omit' else np.mean(data, axis=axis)
    
    # Calculate absolute deviation from the mean
    abs_deviation = np.abs(data - mean)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        # Compute weighted absolute deviation
        weighted_abs_deviation = abs_deviation * sample_weight
        sum_weights = np.sum(sample_weight, axis=axis)
        
        # Ensure that division by zero is handled according to 
        # zero_division parameter
        with np.errstate(divide=zero_division, invalid=zero_division):
            madev = np.sum(weighted_abs_deviation, axis=axis) / np.maximum(
                sum_weights, epsilon)
    else:
        madev = np.mean(abs_deviation, axis=axis)
    
    result = MetricFormatter(title="MADev Results", score=madev)
    
    if detailed:
        # Add detailed statistics if requested
        result.min_deviation = np.min(abs_deviation, axis=axis)
        result.max_deviation = np.max(abs_deviation, axis=axis)
        result.std_deviation = np.std(abs_deviation, axis=axis)
    
    if scale_errors:
        # Scale errors relative to the mean, with protection
        # against division by zero
        scaled_abs_deviation = abs_deviation / np.maximum(mean, epsilon)
        scaled_madev = np.mean(scaled_abs_deviation, axis=axis)
        result.scaled_score = scaled_madev
        
        if detailed:
            # Include detailed statistics for scaled deviations if requested
            result.min_scaled_deviation = np.min(scaled_abs_deviation, axis=axis)
            result.max_scaled_deviation = np.max(scaled_abs_deviation, axis=axis)
            result.std_scaled_deviation = np.std(scaled_abs_deviation, axis=axis)
    
    return result

def dice_similarity_score(
    y_true, y_pred, *, 
    sample_weight=None, 
    nan_policy='propagate',
    epsilon='auto', 
    zero_division='warn', 
    multioutput='uniform_average',
    to_boolean=True  
):
    """
    Compute the Dice Similarity Coefficient (DSC) between two boolean arrays,
    providing flexibility in handling NaN values, converting non-boolean arrays
    to boolean, and adjusting for division by near-zero values.

    Parameters
    ----------
    y_true : array-like of bool
        Ground truth binary labels.
    y_pred : array-like of bool
        Predicted binary labels. Must have the same shape as `y_true`.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If specified, each sample contributes its associated
        weight towards the DSC calculation.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If 'omit', NaN values in `y_true` and `y_pred` are ignored.
    epsilon : {'auto'} or float, optional
        A small positive value to prevent division by zero. If 'auto',
        `epsilon` is dynamically determined based on `y_pred`.
    zero_division : {'warn', 'ignore'}, optional
        Specifies how to handle division by zero. If 'warn', a warning is
        issued. If 'ignore', the case is silently ignored, and NaN is returned.
    multioutput : {'uniform_average'}, optional
        Due to DSC inherently producing a single score, `multioutput` only
        supports 'uniform_average'. Any deviation from this default emits
        a warning.
    to_boolean : bool, optional
        If True, automatically convert `y_true` and `y_pred` to boolean
        arrays if they are not already. Defaults to True.

    Returns
    -------
    float
        The Dice Similarity Coefficient, ranging from 0 (no overlap) to 1
        (perfect overlap) between the predicted and true labels.

    Notes
    -----
    The Dice Similarity Coefficient is defined as:

    .. math:: DSC = \frac{2 |Y_{true} \cap Y_{pred}|}{|Y_{true}| + |Y_{pred}|}

    It measures the overlap between two sets, making it suitable for binary
    classification tasks and image segmentation in medical imaging [1]_. The DSC
    is particularly useful in datasets where the positive class is rare [2]_.

    Examples
    --------
    >>> from gofast.metrics import dice_similarity_score
    >>> y_true = [True, False, True, False, True]
    >>> y_pred = [True, True, True, False, False]
    >>> dice_similarity_score(y_true, y_pred)
    0.6

    See Also
    --------
    jaccard_score : Similarity measure for label sets.
    sklearn.metrics.f1_score : Harmonic mean of precision and recall.

    References
    ----------
    .. [1] Dice, L. R. (1945). Measures of the Amount of Ecologic Association
           Between Species. Ecology, 26(3), 297-302.
    .. [2] Zou, K. H., Warfield, S. K., Bharatha, A., et al. (2004). Statistical
          Validation of Image Segmentation Quality Based on a Spatial Overlap
          Index. Academic Radiology, 11(2), 178-189.

    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, allow_nan=True, 
        multi_output=True # We keep it just to warn user
    )
    if not np.issubdtype(y_true.dtype, np.bool_) or not np.issubdtype(
            y_pred.dtype, np.bool_):
        if to_boolean:
            y_true, y_pred = np.asarray(
                y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
        else:
            raise ValueError("y_true and y_pred must be boolean arrays. "
                             "Use `to_boolean=True` to auto-convert.")

    y_true, y_pred, *opt_sample_weight= validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights= sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    epsilon_value = check_epsilon ( epsilon, y_pred)

    intersection = np.sum(y_true & y_pred)
    sum_total = np.sum(y_true) + np.sum(y_pred)

    if zero_division == 'warn' and sum_total + epsilon_value == 0:
        warnings.warn("Division by zero encountered in Dice Similarity"
                      " Coefficient calculation.", UserWarning)
        return np.nan  # or 0.0 based on how you want to handle this case

    dice_score = 2. * intersection / (np.sum(y_true) + np.sum(y_pred) + epsilon_value)

    if multioutput != 'uniform_average':
        validate_multioutput('warn', extra=' for Dice Similarity Coefficient')

    return dice_score

def gini_score(
     y_true, y_pred, *,  
     sample_weight=None, 
     nan_policy='propagate',
     epsilon='auto', 
     zero_division='warn', 
     multioutput='uniform_average',
     detailed_output=False
):
    """
    Compute the Gini Coefficient, offering a normalized measure of inequality
    among predicted and true values. This metric is traditionally used in economics
    to assess income or wealth distribution but can be adapted to assess the
    inequality of predictive error distributions in various fields, including
    machine learning and statistics.

    Parameters
    ----------
    y_true : array-like
        True observed values. Must be a 1-dimensional array of numeric data.
    y_pred : array-like
        Predicted values by the model. Must have the same shape as `y_true`.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample. Each weight contributes to the
        overall calculation of the Gini coefficient, allowing for unequal impact
        of different observations.
    nan_policy : bool, {'omit', 'propagate', 'raise'}, default='propagate'
        Defines how to handle NaNs in the input arrays. 'propagate' returns the
        input data without changes. 'raise' throws an error if NaNs are detected.
        If set to 'omit', NaN values in both `y_true` and `y_pred` are ignored,
        ensuring the calculation only considers valid numerical entries.
    epsilon : {'auto'} or float, optional
        A small positive value added to the denominator to prevent division by
        zero. When set to 'auto', the epsilon value is dynamically determined
        based on the range and scale of `y_pred`.
    zero_division : {'warn', 'ignore'}, optional
        Defines the behavior when the calculation encounters a division by zero.
        'warn' issues a user warning and returns NaN, while 'ignore' silently
        returns NaN without issuing a warning.
    multioutput : {'uniform_average'}, optional
        This parameter is included for compatibility with other metrics' API and
        does not affect the computation since Gini coefficient inherently provides
        a single scalar value regardless of the number of outputs.
    detailed_output : bool, optional
        When set to True, the function returns a detailed Bunch object containing
        the Gini coefficient, total weighted observations, and the weighted sum of
        absolute differences. This can be useful for in-depth analysis.

    Returns
    -------
    float or Bunch
        The Gini Coefficient as a float if `detailed_output=False`. If
        `detailed_output=True`, returns a Bunch object containing the Gini
        coefficient along with additional detailed information.

    Notes
    -----
    The Gini Coefficient is a measure of statistical dispersion intended to
    represent the inequality among values of a frequency distribution (for
    example, levels of income) [1]_. It is defined mathematically as:

    .. math::
        G = \frac{\sum_i \sum_j |y_{\text{true},i} - \\
                  y_{\text{pred},j}|}{2n\sum_i y_{\text{true},i}}

    where `n` is the number of observations. The coefficient ranges from 0,
    indicating perfect equality (where all values are the same), to 1,
    indicating maximum inequality among values [2]_.

    Examples
    --------
    >>> from gofast.metrics import gini_score
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [2, 2, 3, 4, 4]
    >>> gini_score(y_true, y_pred)
    0.2

    The Gini coefficient of 0.2 in this example indicates a low level of
    inequality between the true and predicted values.

    See Also
    --------
    sklearn.metrics.mean_squared_error : Mean squared error regression loss.
    sklearn.metrics.mean_absolute_error : Mean absolute error regression loss.

    References
    ----------
    .. [1] Gini, C. (1912). "Variability and Mutability". C. Cuppini, Bologna, 156 pages.
    .. [2] Yitzhaki, S. (1983). "On an Extension of the Gini Inequality Index".
           International Economic Review, 24(3), 617-628.
    """

    y_true, y_pred = _ensure_y_is_valid(# We keep it just to warn user
        y_true, y_pred, allow_nan=True, multi_output=True 
    )
    
    y_true, y_pred, *opt_sample_weight= validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights= sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    epsilon_value = check_epsilon(epsilon, y_true, y_pred)
    abs_diff = np.abs(np.subtract.outer(y_true, y_pred))
    gini_sum = np.sum(abs_diff)

    if sample_weight is not None:
        # Apply weights to each pair's absolute difference
        weighted_gini_sum = np.sum(abs_diff * sample_weight[:, None])
        total_weighted = np.sum(y_true * sample_weight)
    else:
        weighted_gini_sum = gini_sum
        total_weighted = np.sum(y_true)

    total_weighted = handle_zero_division(
        total_weighted, zero_division=zero_division, 
        metric_name="Gini Coefficient", 
        epsilon=epsilon_value, 
        replace_with=np.nan # or return np.nan as per handling strategy
        ) 

    gini_coefficient = weighted_gini_sum / (2 * len(y_true) * total_weighted 
                                            + epsilon_value)

    if multioutput != 'uniform_average': # keep it for API consistency
        validate_multioutput('warn', extra=' for Gini Coefficient calculation')

    if detailed_output:
        gini_coefficient=MetricFormatter (
            title="Gini Results",
            gini_score = gini_coefficient, 
            total_weighted = total_weighted, 
            weighted_gini_sum= weighted_gini_sum 
            )

    return gini_coefficient
    
def hamming_loss(
    y_true, y_pred,*,  
    sample_weight=None, 
    nan_policy='propagate',  
    epsilon=1e-8,  
    zero_division='warn', 
    normalize=True, 
    to_boolean=False        
):
    """
    Compute the Hamming loss, the fraction of labels that are incorrectly 
    predicted.

    The Hamming loss is the fraction of labels that are incorrectly predicted,
    relative to the total number of labels. It is a useful metric in multi-label
    classification tasks, where each instance may have multiple labels [1]_.

    Parameters
    ----------
    y_true : array-like
        True labels of the data. Can be a 1D array for single-label tasks or
        a 2D array for multi-label tasks.
    y_pred : array-like
        Predicted labels of the data, must have the same shape as y_true.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If provided, the Hamming loss will be averaged across
        the samples accordingly.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle NaNs in the input data. 'propagate' returns NaN if
        NaNs are detected, 'raise' throws an error, and 'omit' ignores elements
        with NaNs. If 'omit', ignore NaN values in both y_true and y_pred 
        during the loss calculation. This is particularly useful in datasets 
        with missing labels.
    epsilon : float, optional
        A small value to prevent division by zero. This is useful in ensuring
        numerical stability of the loss calculation.
    zero_division : {'warn', 'ignore'}, optional
        Defines how to handle the scenario when division by zero occurs. If 'warn',
        a warning is issued. If 'ignore', the division by zero is silently ignored.
    normalize : bool, optional
        If True, normalizes the Hamming loss by the total number of labels per sample.
        This is recommended for multi-label classification tasks to ensure the loss
        is within a [0, 1] range.
    to_boolean : bool, optional
        If True, automatically convert y_true and y_pred to boolean arrays. 
        This is useful for binary classification tasks where labels might
        not be in boolean format.

    Returns
    -------
    float
        The Hamming loss, ranging from 0 (perfect match) to 1 (complete mismatch)
        if `normalize` is True. Otherwise, it returns the average number of label
        mismatches per sample.

    Notes
    -----
    The Hamming loss is defined as:

    .. math::
        HL = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{L_i} \sum_{j=1}^{L_i}\\
            1(y_{\text{true},ij} \neq y_{\text{pred},ij})

    where `N` is the number of samples, `L_i` is the number of labels for the 
    `i`-th sample,and `1(.)` is the indicator function.

    Examples
    --------
    >>> from gofast.metrics import hamming_loss
    >>> y_true = np.array([[1, 1], [1, 0]])
    >>> y_pred = np.array([[1, 0], [1, 1]])
    >>> print(hamming_loss(y_true, y_pred, normalize=True))
    0.5

    References
    ----------
    .. [1] Zhao, M., & Zhang, Z. (2012). Multi-label learning by exploiting 
           label dependency. In Proceedings of the 16th ACM SIGKDD international
           conference on Knowledge discovery and data mining (pp. 999-1008).

    See Also
    --------
    sklearn.metrics.accuracy_score : Compute the accuracy.
    sklearn.metrics.jaccard_score : Jaccard similarity coefficient score.

    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, allow_nan=True, 
        multi_output=True 
    )
    # validate epsilon 
    epsilon= check_epsilon(epsilon, y_pred , scale_factor=1e-8)
     # Optionally convert inputs to boolean for binary classification tasks
    if to_boolean:
        y_true = y_true.astype(bool)
        y_pred = y_pred.astype(bool)
        
    # Directly unpack the result from validate_nan_policy function 
    # call into y_true, y_pred, and possibly sample_weight
    # The use of * in the assignment allows it to work with
    # or without sample_weight being returned
    y_true, y_pred, *optional_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights=sample_weight 
    )
    # If sample_weight was provided and thus returned, update sample_weight
    # variable, else keep it unchanged
    sample_weight = optional_sample_weight[0] if optional_sample_weight else sample_weight 

    # Calculate mismatches between y_true and y_pred
    mismatch = y_true != y_pred
    if normalize:
        # Normalize the Hamming loss by the total number of labels
        n_labels = np.size(y_true, axis=1)
        mismatch_sum = np.sum(mismatch, axis=1) / n_labels
    else:
        # Count mismatches without normalization
        mismatch_sum = np.sum(mismatch, axis=1)

    # Apply sample weights if provided
    if sample_weight is not None:
        hamming_loss_value = np.average(mismatch_sum, weights=sample_weight)
    else:
        hamming_loss_value = np.mean(mismatch_sum)
        
    # Handling division by zero based on zero_division paramete
    if zero_division == 'warn' and np.isclose(hamming_loss_value, 0, atol=epsilon):
        warnings.warn("Potential division by zero encountered in Hamming"
                      " loss calculation.", UserWarning)

    return hamming_loss_value

def fowlkes_mallows_score(
    y_true, y_pred, *, 
    sample_weight=None, 
    average='macro', 
    epsilon=1e-10, 
    zero_division='warn', 
    multioutput='uniform_average', 
):
    """
    Compute the Fowlkes-Mallows Index (FMI) score between true and predicted 
    cluster labels.
    
    The FMI is a measure of similarity between two sets of clusters, defined as the
    geometric mean of the pairwise precision and recall. It is particularly useful
    in clustering validation by evaluating the similarity between the ground truth
    clustering and the predicted clustering.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels or clusters.
    y_pred : array-like of shape (n_samples,)
        Predicted labels or clusters.
    sample_weight : array-like of shape (n_samples,), optional
        Weights assigned to the samples.
    average : {'macro', 'weighted'}, optional
        Determines the type of averaging performed on the data:
        
        - 'macro': Calculate metrics for each label, and find their unweighted
           mean. This does not take label imbalance into account.
        - 'weighted': Calculate metrics for each label, and find their average 
           weighted by support (the number of true instances for each label).
          
    epsilon : float, optional
        A small constant added to denominators to prevent division by zero.
    zero_division : {'warn', 'ignore'}, optional
        Defines how to handle the case when there is a zero division:
        
        - 'warn': Emit a warning and return 0.0
        - 'ignore': Return `np.nan`
          
    multioutput : {'uniform_average'}, optional
        Currently only supports 'uniform_average' for API consistency. The
        Fowlkes-Mallows index inherently combines multiple outputs into a
        single score.

    Returns
    -------
    float
        The Fowlkes-Mallows Index, ranging from 0 (no shared members) to
        1 (all members shared).
        
    Notes
    -----
    The Fowlkes-Mallows Index is defined as:
    
    .. math::
        FMI = \sqrt{\frac{TP}{TP + FP} \cdot \frac{TP}{TP + FN}}
    
    where `TP` is the number of True Positive (pairwise instances correctly classified as
    belonging to the same cluster), `FP` is the number of False Positives, and `FN` is the
    number of False Negatives.
    
    References
    ----------
    .. [1] Fowlkes, E. B., & Mallows, C. L. (1983). A method for comparing 
          two hierarchical clusterings. Journal of the American Statistical 
          Association, 78(383), 553-569.
    
    See Also
    --------
    sklearn.metrics.adjusted_rand_score : 
        Adjusted Rand index for clustering performance evaluation.
    sklearn.metrics.silhouette_score : 
        Mean silhouette coefficient for all samples.
    
    Examples
    --------
    >>> from gofast.metrics import fowlkes_mallows_score
    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 3, 3]
    >>> fowlkes_mallows_score(y_true, y_pred)
    0.8606
    
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True )
    epsilon= check_epsilon (epsilon, y_pred, scale_factor=1e-10)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tp = np.diag(cm)  # True Positives
    fp = np.sum(cm, axis=0) - tp  # False Positives
    fn = np.sum(cm, axis=1) - tp  # False Negatives

    # Adjusted calculations with epsilon to avoid division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    average = normalize_string(
        average, target_strs= ['average', 'macro'], 
        match_method='contains',raise_exception=True, 
        return_target_only=True,
        error_msg=f"Invalid average method: '{average}'"
    )
    if average == 'macro':
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        precision_avg = np.average(precision, weights=weights)
        recall_avg = np.average(recall, weights=weights)
 
    fmi = np.sqrt(precision_avg * recall_avg)

    if multioutput != 'uniform_average':
        validate_multioutput('warn', extra =' for Fowlkes-Mallows Index')
    
    # Handle division by zero after calculations
    if zero_division == 'warn' and (fmi == 0 or np.isnan(fmi)):
        warnings.warn("Division by zero encountered in Fowlkes-Mallows"
                      " calculation.", UserWarning)
        return 0.0 if zero_division == 'warn' else np.nan

    return fmi

def root_mean_squared_log_error(
    y_true, y_pred, *, 
    sample_weight=None, 
    clip_value=0,  
    multioutput='uniform_average',
):
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE) between true
    and predicted values. The RMSLE is a measure of accuracy for predictions
    of positive-valued targets, emphasizing the relative error between
    predictions and actual values and penalizing underestimates more than
    overestimates.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values. Must be non-negative.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values. Must be non-negative and have the same shape
        as `y_true`.
    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample, contributing to the calculation
        of the average error.
    clip_value : float, optional
        Minimum value for `y_true` and `y_pred` after clipping to avoid taking
        a logarithm of zero. Defaults to 0, ensuring non-negativity.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Strategy for aggregating errors across multiple output dimensions:
        - 'raw_values' : Returns an array of RMSLE values for each output.
        - 'uniform_average' : Averages errors across all outputs.

    Returns
    -------
    float or ndarray
        The RMSLE between `y_true` and `y_pred`. If `multioutput` is 'raw_values',
        an array of RMSLE values for each output is returned. If 'uniform_average',
        a single float value is returned.

    Notes
    -----
    RMSLE is defined as the square root of the average squared difference between
    the logarithms (base e) of the predicted and actual values, incremented by one:

    .. math:: \sqrt{\frac{1}{n} \sum_{i=1}^{n} [\log(p_i + 1) - \log(a_i + 1)]^2}

    Here, :math:`p_i` and :math:`a_i` are the predicted and actual values,
    respectively, for each sample :math:`i`. The addition of one inside the 
    logarithm allows handling of zero values in inputs.

    The RMSLE is less sensitive to large errors when both predicted and true 
    values are large numbers. Unlike mean squared error (MSE) or root mean 
    squared error (RMSE), RMSLE does not penalize overestimates more than 
    underestimates, making it particularly suitable for data and models where 
    underestimates are more undesirable.

    Examples
    --------
    >>> from gofast.metrics import root_mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> rmse_log_error(y_true, y_pred)
    0.199

    See Also
    --------
    mean_squared_error : Compute mean squared error.
    mean_absolute_error : Compute mean absolute error.
    r2_score : R^2 (coefficient of determination) regression score function.

    References
    ----------
    .. [1] Wikipedia on Root Mean Squared Logarithmic Error: 
           https://en.wikipedia.org/wiki/Root-mean-square_deviation#RMSLE
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True )

    # Ensure non-negativity for log calculation
    if clip_value is not None: 
        y_true = np.clip(y_true, clip_value, None) 
        y_pred = np.clip(y_pred, clip_value, None)  

    # Check to ensure non-negativity
    ensure_non_negative(y_true, y_pred)

    # Compute log1p = log(x + 1) to ensure no log(0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Compute squared log error
    squared_log_error = (log_pred - log_true) ** 2

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1)
        squared_log_error *= sample_weight

    # Compute mean squared log error
    msle = np.mean(squared_log_error, axis=0)

    # Compute root mean squared log error
    rmsle = np.sqrt(msle)

    # Aggregate outputs
    multioutput =validate_multioutput(multioutput)
    
    return rmsle if multioutput == 'raw_values' else  np.mean(rmsle)

def mean_percentage_error(
    y_true, y_pred, *, 
    sample_weight=None, 
    epsilon='auto', 
    zero_division='warn', 
    multioutput='uniform_average',
):
    """
    Compute the Mean Percentage Error (MPE) between true and predicted values,
    offering options for handling edge cases and applying weights to the errors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True target values. Can be a 1D array for single-output regression, or a
        2D array for multi-output regression, where each column represents a
        different output to predict.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values by the model. Must have the same shape as `y_true`.
        Predictions provided by the model for evaluation against `y_true`.

    sample_weight : array-like of shape (n_samples,), optional
        Individual weights for each sample. This parameter assigns a specific weight
        to each observation, allowing for differential importance to be given to
        certain observations over others during error calculation.

    epsilon : {'auto'} or float, optional
        Threshold for considering values as non-zero to prevent division by zero.
        If 'auto', the threshold is determined dynamically based on the data range
        and scale. This prevents infinite or undefined percentage errors when
        `y_true` contains zeros.

    zero_division : {'warn', 'ignore'}, optional
        How to handle cases when division by zero might occur during percentage
        error calculation. If 'warn', a warning message is displayed. If 'ignore',
        these cases are silently passed over without raising a warning.

    multioutput : {'raw_values', 'uniform_average'}, optional
        Defines the method for aggregating percentage errors across multiple outputs
        (for multi-output models). 'raw_values' returns an array with an error value
        for each output, while 'uniform_average' averages errors across all outputs
        to provide a single error score.

    Returns
    -------
    float or ndarray
        The Mean Percentage Error value. For `multioutput='raw_values'`, it returns
        an array with the MPE for each output. For `multioutput='uniform_average'`,
        it returns a single float representing the average MPE across all outputs.

    Notes
    -----
    The MPE is calculated using the formula:

    .. math:: 
        MPE = \frac{100}{n} \sum_{i=1}^{n}\\
            \frac{(y_{\text{pred},i} - y_{\text{true},i})}{y_{\text{true},i}}

    where :math:`n` is the number of samples, :math:`y_{\text{pred},i}` is the 
    predicted value, and :math:`y_{\text{true},i}` is the actual value. The MPE can 
    indicate a model's tendency to overestimate or underestimate the values, with 
    positive values indicating overestimation and negative values indicating 
    underestimation [1]_.
    
    The function provides insights into the relative forecasting accuracy of a
    model, making it particularly useful in financial and operational forecasting
    models where percentage differences are more meaningful than absolute 
    differences.

    See Also
    --------
    mean_absolute_error : Compute the mean absolute error.
    mean_squared_error : Compute mean squared error.
    mean_squared_log_error : Compute mean squared logarithmic error.

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B. (2006). Another look at measures of forecast
           accuracy. International Journal of Forecasting, 22(4), 679-688.
           
    Example
    -------
    >>> from gofast.metrics import mean_percentage_error
    >>> y_true = [100, 150, 200, 250]
    >>> y_pred = [110, 140, 210, 240]
    >>> mean_percentage_error(y_true, y_pred)
    0.5
    >>> from gofast.metrics import mean_percentage_error
    >>> y_true = [3, 5, 7.5, 10]
    >>> y_pred = [2.5, 5, 8, 9.5]
    >>> mean_percentage_error(y_true, y_pred)
    -2.5

    """
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True,multi_output =True)
    # Check to ensure non-negativity for division
    ensure_non_negative(y_true, 
        err_msg="y_true must contain non-negative values for MPE calculation.")

    # Determine epsilon value
    epsilon = check_epsilon(epsilon, y_true, y_pred, base_epsilon=1e-10)
    
    # Handle zero division if necessary
    y_true = handle_zero_division(
        y_true, zero_division=zero_division, 
        replace_with =np.nan,
        metric_name ="Mean Percentage Error", 
        )
    # Compute percentage error
    percentage_error = (y_pred - y_true) / np.clip(
        y_true, epsilon, np.max(y_true)) * 100

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        weighted_percentage_error = np.average(percentage_error, weights=sample_weight)
    else:
        weighted_percentage_error = np.mean(percentage_error)

    # Aggregate outputs
    multioutput = validate_multioutput(multioutput)

    if multioutput == 'uniform_average':
        weighted_percentage_error=  np.mean(weighted_percentage_error) if np.ndim(
            weighted_percentage_error) > 0 else weighted_percentage_error
        
    return weighted_percentage_error 

def spearmans_rank_score(
    y_true,  y_pred, *, 
    sample_weight=None,  
    multioutput='uniform_average',
    tie_method='average',  
    nan_policy='propagate',  
    control_vars=None, 
):
    """
    Calculate Spearman's rank correlation coefficient score, with options for 
    handling ties, NaNs, control variables, and multi-output data. This 
    non-parametric measure assesses the monotonic relationship between 
    two datasets without making any assumptions about their frequency 
    distribution.

    Parameters
    ----------
    y_true : array-like
        True values to compare against predictions.
    y_pred : array-like
        Predicted values to be compared to the true values.
    sample_weight : array-like, optional
        Weights for each sample, allowing for weighted correlation calculation.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Determines how to aggregate results for multiple outputs. 'raw_values'
        returns an array of correlations for each output, while 'uniform_average'
        returns the average of all correlations.
    tie_method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        Specifies how to assign ranks to tied elements in the data. The default
        'average' assigns the average of the ranks that would have been assigned
        to all tied values.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle NaNs in the input data. 'propagate' returns NaN if
        NaNs are detected, 'raise' throws an error, and 'omit' ignores elements
        with NaNs.
    control_vars : array-like or list of array-likes, optional
        Control variables for adjusting the calculation of Spearman's rank 
        correlation, used for partial correlation analysis.

    Returns
    -------
    float or ndarray
        Spearman's rank correlation coefficient. If `multioutput` is 'raw_values',
        an array with a correlation coefficient for each output is returned.
        
    Notes
    -----
    Spearman's rank correlation coefficient (\(\rho\)) is a non-parametric measure
    of rank correlation, meaning it assesses how well the relationship between two
    variables can be described using a monotonic function. It evaluates the 
    monotonic relationship by comparing the ranked values for each variable rather
    than the raw data. This method makes it suitable for use with non-linear data
    and data that does not follow a normal distribution, differentiating it from
    Pearson's correlation coefficient which assumes a linear relationship and
    normally distributed data.
    
    The coefficient ranges from -1 to 1, inclusive. A \(\rho\) value of 1 indicates
    a perfect positive monotonic relationship between the two datasets, meaning as
    one variable increases, the other also increases. Conversely, a \(\rho\) value
    of -1 indicates a perfect negative monotonic relationship, signifying that as
    one variable increases, the other decreases. A \(\rho\) value of 0 suggests no
    monotonic relationship between the variables.
    
    The Spearman's rank correlation is especially useful when examining relationships
    involving ordinal variables where the precise differences between ranks are not
    of primary concern but the order of those ranks is important. It is also robust
    to outliers, as it depends on the rank order of values rather than their
    specific magnitudes.
    
    The mathematical expression for Spearman's rank correlation coefficient is:
    
    .. math::
        \\rho = 1 - \\frac{6 \\sum d_i^2}{n(n^2 - 1)}
    
    where \(d_i\) represents the difference between the ranks of corresponding values
    in the two datasets, and \(n\) is the number of observations. The term \(d_i^2\)
    highlights that the correlation is based on the squared differences in ranks,
    ensuring that the direction of the difference does not affect the calculation,
    only its magnitude.
    
    Handling ties in the data (where two or more values receive the same rank) and
    how missing values are treated can affect the computation of Spearman's \(\rho\).
    The `tie_method` parameter allows for different strategies in ranking ties,
    while the `nan_policy` parameter defines the approach towards handling missing
    values. Control variables, if present, are considered through partial
    correlation techniques to isolate the effect of the primary variables of interest.

    References
    ----------
    - Spearman, C. (1904). "The proof and measurement of association between two things".
    - Myer, K., & Waller, N. (2009). "Applied Spearman's rank correlation". 
      Statistics in Medicine.

    See Also
    --------
    scipy.stats.spearmanr : 
        Spearman rank-order correlation coefficient calculation in SciPy.
    pandas.DataFrame.corr:
        Compute pairwise correlation of columns, excluding NA/null values.
    gofast.tools.mathex.optimized_spearmanr: 
        Compute Spearman's rank correlation coefficient with support for 
        sample weights, custom tie handling, and NaN policies.

    Examples
    --------
    >>> from gofast.metrics import spearmans_rank_coeff
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [5, 6, 7, 8, 7]
    >>> spearmans_rank_coeff(y_true, y_pred)
    0.8208

    >>> y_true = np.array([1, 2, np.nan, 4])
    >>> y_pred = np.array([4, np.nan, 3, 1])
    >>> spearmans_rank_coeff(y_true, y_pred, nan_policy='omit')
    -0.9999999999999999
    """

    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True, allow_nan=True,
        multi_output =True 
        )
    # Check for non-negativity and other conditions as needed
    ensure_non_negative(y_true, y_pred )
    # If control_vars is specified, compute partial correlation
    result= optimized_spearmanr(
        y_true, y_pred, 
        sample_weight= sample_weight, 
        tie_method=tie_method, 
        nan_policy=nan_policy, 
        control_vars= control_vars, 
        multioutput= multioutput 
        ) 

    return result

def precision_at_k(
    y_true, y_pred, k, *,
    sample_weight=None,
    multioutput='uniform_average'
):
    """
    Compute the precision at rank k between lists of true relevant items
    and predicted items. This metric evaluates the proportion of relevant items
    found in the top-k positions of the ranked prediction list, which is 
    particularly useful in recommendation systems and information retrieval.
    
    See more in :ref:`User Guide`. 

    Parameters
    ----------
    y_true : list of list of int
        True labels of each sample. Each inner list contains the labels of
        the relevant items for the corresponding sample.
    y_pred : list of list of int
        Predicted labels for each sample. Each inner list is the ranked list
        of predicted items for the corresponding sample.
    k : int
        The rank at which the precision is evaluated. Precision considers only
        the top-k items in the predicted list.
    sample_weight : array-like of shape (n_samples,), optional
        Weights applied to each sample in the calculation of the average precision.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Specifies how to aggregate results across multiple outputs. If 
        'raw_values', returns a score for each sample. If 'uniform_average',
        returns the average precision score.

    Returns
    -------
    float or np.ndarray
        The precision score(s). If `multioutput` is 'raw_values', returns an array
        with a precision score for each sample. If 'uniform_average', returns the
        overall average precision.

    Notes
    -----
    Precision at k (P@k) is defined as:

    .. math::
        P@k = \\frac{1}{|U|} \\sum_{u=1}^{|U|} \\frac{|\\{
            \\text{relevant items at k for user } u\\} \\cap \\{
                \\text{recommended items at k for user } u\\}|}{k}

    where \(|U|\) is the number of users, \(k\) is the rank threshold, and
    the intersection considers the number of true relevant items that appear in
    the top-k predicted items. This metric emphasizes the importance of having
    relevant items at the top of the recommendation list.

    Examples
    --------
    >>> from gofast.metrics import precision_at_k
    >>> y_true = [[1, 2, 3], [1, 2]]
    >>> y_pred = [[2, 1, 4], [1, 2, 3]]
    >>> k = 2
    >>> precision_at_k(y_true, y_pred, k)
    0.75

    References
    ----------
    .. [1] Manning, C.D., Raghavan, P., Schütze, H. (2008). Introduction to 
           Information Retrieval. Cambridge University Press.

    See Also
    --------
    recall_at_k : Calculate recall at rank k.
    f1_score_at_k : Calculate the F1 score at rank k.
    """
    # Ensure y_true and y_pred are standardized 
    y_true, y_pred = standardize_input(y_true, y_pred)

    # Convert k to int if it's float or a NumPy numeric type
    if not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError("k must be an integer or floating point number.")
    k = int(k)

    # Check that k is a positive integer after conversion
    if k <= 0:
        raise ValueError("k must be a positive integer after conversion.")

    precision_scores = []
    weights = []

    # Calculate precision scores
    for idx, (true_items, pred_items) in enumerate(zip(y_true, y_pred)):
        true_items, pred_items = list(true_items) , list(pred_items)
        if len(pred_items) < k:
            raise ValueError(f"Predicted items for instance {idx} has fewer than k items.")
        num_relevant = len(set(true_items) & set(pred_items[:k]))
        precision_score = num_relevant / k
        precision_scores.append(precision_score)
    
        # Append corresponding weight or 1 if no sample_weight is provided
        if sample_weight is not None and len(sample_weight) > idx:
            weights.append(sample_weight[idx])
        else:
            # default weight is 1 when sample_weight is None or shorter than index
            weights.append(1)  
    
    # Normalize weights if provided
    weights = np.array(weights)
    if sample_weight is not None:
        # normalize only if weights were explicitly provided
        weights = weights / np.sum(weights)  
    
    # Validate multioutput parameter
    multioutput = validate_multioutput(multioutput)
    
    # Return the appropriate output based on multioutput parameter
    if multioutput == 'raw_values':
        return np.array(precision_scores)
    elif multioutput == 'uniform_average':
        return np.average(precision_scores, weights=weights)

def ndcg_at_k(
    y_true, y_pred, k, *, 
    sample_weight=None, 
    multioutput='uniform_average', 
    nan_policy='omit'
    ):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at rank K, a widely
    recognized measure in information retrieval and ranking problems. This 
    metric assesses the quality of the ranking by considering the order of
    relevance scores assigned to the set of items up to position K. It's
    particularly useful in evaluating the performance of search engines and
    recommender systems.

    Parameters
    ----------
    y_true :  list of list of int
        List of lists containing the true relevant items with their grades.
    y_pred : list of list of int
        List of lists containing the predicted items up to position K.
    k : int
        The rank at which the NDCG is calculated.
    sample_weight : array-like, optional
        Weights for each sample in y_true and y_pred. Default is None.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Determines the type of averaging performed on the data:
        'raw_values' returns an array with NDCG score per each set of predictions;
        'uniform_average' calculates an average of all NDCG scores.
        Default is 'uniform_average'.
    nan_policy : {'omit', 'raise', 'propagate'}, optional
        Defines how to handle when input contains NaN. Currently, only 'omit'
        is supported, which excludes NaNs from the calculation. 
        Default is 'omit'.

    Returns
    -------
    float or np.ndarray
        The NDCG score at K. Returns a single float value if 'uniform_average' 
        is selected for `multioutput`, or an array of scores if 'raw_values'.

    Notes
    -----
    The NDCG at position K is calculated as the ratio of the Discounted 
    Cumulative Gain (DCG) at K to the Ideal DCG (IDCG) at K, ensuring the 
    score is normalized to the range [0, 1]:

    .. math::
        \mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}

    where:

    .. math::
        \mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}

    and `rel_i` is the graded relevance of the result at position `i`.

    Examples
    --------
    >>> from gofast.metrics import ndcg_at_k
    >>> y_true = [[3, 2, 3], [2, 1, 2]]
    >>> y_pred = [[1, 2, 3], [1, 2, 3]]
    >>> k = 3
    >>> ndcg_at_k(y_true, y_pred, k)
    0.8388113303189411

    References
    ----------
    .. [1] Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation 
           of IR techniques. ACM Transactions on Information Systems (TOIS), 
           20(4), 422-446.

    See Also
    --------
    precision_at_k : Precision at rank K for ranking problems.
    """
    def dcg_at_k(r, k):
        """ Calculate Discounted Cumulative Gain (DCG) at rank k """
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0
    
    # Check that k is a positive integer after conversion
    k=validate_positive_integer(k, "k")
    # Convert lists to numpy arrays for easier manipulation
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric =True, multi_output=True)
    # Handling NaNs based on policy
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights=sample_weight) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
     # If NaN exists
    if nan_policy == 'propagate' and (
            np.isnan(y_true).any() or np.isnan(y_pred).any()):
        return np.nan
    
    # Initialize the list to store NDCG scores for each sample
    ndcg_scores = []

    # reshape arrays if not 2D yet. This is useful for handling 
    # multi-output and 1D target array.
    y_true, y_pred = convert_array_dimensions(y_true, y_pred, target_dim=2 )
    
    # Compute NDCG for each set of predictions and true values
    for i in range(y_true.shape[0]):
        # Sorting by predicted scores for the current set
        order = np.argsort(y_pred[i])[::-1]
        y_true_sorted = y_true[i, order]

        # Calculate DCG and IDCG for the current set
        dcg = dcg_at_k(y_true_sorted, k)
        idcg = dcg_at_k(sorted(y_true_sorted, reverse=True), k)
        
        # Calculate NDCG for the current set
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    # Handle multioutput options
    multioutput = validate_multioutput(multioutput)
    if multioutput == 'uniform_average':
        # Return the average of all NDCG scores
        return np.mean(ndcg_scores)
    elif multioutput == 'raw_values':
        # Return the array of NDCG scores
        return np.array(ndcg_scores)
 
def _ndcg_at_k(
    y_true, y_pred, k, *, 
    sample_weight=None, 
    multioutput='uniform_average', 
    nan_policy='omit'
):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at rank K, a widely
    recognized measure in information retrieval and ranking problems. This 
    metric assesses the quality of the ranking by considering the order of
    relevance scores assigned to the set of items up to position K. It's
    particularly useful in evaluating the performance of search engines and
    recommender systems.

    Parameters
    ----------
    y_true :  list of list of int
        List of lists containing the true relevant items with their grades.
    y_pred : list of list of int
        List of lists containing the predicted items up to position K.
    k : int
        The rank at which the NDCG is calculated.
    sample_weight : array-like, optional
        Weights for each sample in y_true and y_pred. Default is None.
    multioutput : {'raw_values', 'uniform_average'}, optional
        Determines the type of averaging performed on the data:
        'raw_values' returns an array with NDCG score per each set of predictions;
        'uniform_average' calculates an average of all NDCG scores.
        Default is 'uniform_average'.
    nan_policy : {'omit', 'raise', 'propagate'}, optional
        Defines how to handle when input contains NaN. Currently, only 'omit'
        is supported, which excludes NaNs from the calculation. 
        Default is 'omit'.

    Returns
    -------
    float or np.ndarray
        The NDCG score at K. Returns a single float value if 'uniform_average' 
        is selected for `multioutput`, or an array of scores if 'raw_values'.

    Notes
    -----
    The NDCG at position K is calculated as the ratio of the Discounted 
    Cumulative Gain (DCG) at K to the Ideal DCG (IDCG) at K, ensuring the 
    score is normalized to the range [0, 1]:

    .. math::
        \mathrm{NDCG@K} = \frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}}

    where:

    .. math::
        \mathrm{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}

    and `rel_i` is the graded relevance of the result at position `i`.

    Examples
    --------
    >>> from gofast.metrics import ndcg_at_k
    >>> y_true = [[3, 2, 3], [2, 1, 2]]
    >>> y_pred = [[1, 2, 3], [1, 2, 3]]
    >>> k = 3
    >>> ndcg_at_k(y_true, y_pred, k)
    0.7454516132114052

    References
    ----------
    .. [1] Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation 
           of IR techniques. ACM Transactions on Information Systems (TOIS), 
           20(4), 422-446.

    See Also
    --------
    precision_at_k : Precision at rank K for ranking problems.
    """
    # Ensure y_true and y_pred are standardized 
    y_true, y_pred = standardize_input(y_true, y_pred)
    # Handle NaN values according to nan_policy
    (y_true, y_pred), *opt_weights = filter_nan_from(
         y_true, y_pred, sample_weights =sample_weight ) 
    sample_weight = opt_weights[0] if opt_weights else sample_weight 

    # Calculate DCG@k and IDCG@k for each pair of true and predicted rankings
    def dcg_at_k(scores, k):
        return np.sum([(2**scores[i] - 1) / np.log2(i + 2) for i in range(
            min(len(scores), k))])

    # Convert k to int if it's float or a NumPy numeric type
    if not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError("k must be an integer or floating point number.")
    k = int(k)

    # Check that k is a positive integer after conversion
    if k <= 0:
        raise ValueError("k must be a positive integer.")
        
    ndcg_scores = []
    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        actual_scores = np.array([dict(true).get(item, 0) for item in list(pred)[:k]])
        ideal_scores = np.sort(list(true.values()))[::-1][:k]

        dcg_score = dcg_at_k(actual_scores, k)
        idcg_score = dcg_at_k(ideal_scores, k)

        ndcg_score = dcg_score / idcg_score if idcg_score > 0 else 0
        ndcg_scores.append(ndcg_score)

    if sample_weight is not None:
        ndcg_scores = np.average(ndcg_scores, weights=sample_weight)

    multioutput = validate_multioutput(multioutput)
    return np.array(ndcg_scores) if multioutput == 'raw_values' else np.mean(ndcg_scores)

def ndcg_at_k_with_controls(
    y_true, y_pred, k, *,
    sample_weight=None,
    multioutput='uniform_average',
    nan_policy='propagate',
    epsilon=1e-10,
    clip_value=0
):
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) at rank K, incorporating
    advanced controls such as handling missing values, adjusting for sample weights,
    and clipping predictions. NDCG quantifies the effectiveness of a ranking model 
    based on the graded relevance of the recommended items.

    Parameters
    ----------
    y_true : array-like
        True relevance scores for each item, indicating the ideal order.
    y_pred : array-like
        Predicted relevance scores for each item, given by the ranking model.
    k : int
        Rank position at which NDCG is evaluated, considering only the top-k items.
    sample_weight : array-like, optional
        Weights for each pair of values, emphasizing some over others.
    multioutput : {'uniform_average', 'raw_values'}, optional
        Aggregation method for multiple outputs. 'uniform_average' returns 
        the average NDCG across all outputs, while 'raw_values' returns NDCG 
        scores for each output.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Strategy for handling NaN values in `y_true` and `y_pred`. 'propagate' 
        allows NaNs to affect the score, 'omit' ignores them, and 'raise' throws 
        an error if NaNs are detected.
    epsilon : float, optional
        Small constant added to prevent division by zero in logarithmic calculations.
    clip_value : float, optional
        Minimum threshold to clip predicted relevance scores, ensuring they 
        remain non-negative.

    Returns
    -------
    float or ndarray
        The calculated NDCG score at rank K. Returns a single float if 
        `multioutput` is 'uniform_average' or an array of scores if 
        `multioutput` is 'raw_values'.

    Notes
    -----
    NDCG is essential in evaluating ranking algorithms where the order of 
    items is crucial, such as search engines and recommender systems. It 
    provides insight into the model's ability to rank items in order of 
    their relevance, which is particularly valuable when the relevance 
    is graded rather than binary.

    The NDCG score is derived by comparing the DCG of the predicted ranking 
    to the ideal DCG (IDCG), ensuring that the score is normalized to a 
    range of 0 (worst) to 1 (best):

    .. math::
        \mathrm{DCG@K} &= \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)} \\
        \mathrm{IDCG@K} &= \sum_{i=1}^{k} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)} \\
        \mathrm{NDCG@K} &= \frac{DCG@K}{IDCG@K}

    where :math:`rel_i` is the relevance score of the item at position :math:`i`
    in the predicted ranking, and :math:`rel_i^*` is the relevance score of the 
    item at position :math:`i` in the ideal (perfect) ranking.

    This metric is particularly useful for assessing the performance of ranking
    models in situations where the position of an item in the list (and its 
    relative relevance) significantly impacts the user experience or the 
    effectiveness of the model.

    Examples
    --------
    >>> from gofast.metrics import ndcg_at_k_with_controls
    >>> y_true = [[3, 2, 3], [2, 1, 2]]
    >>> y_pred = [[1, 2, 3], [1, 2, 3]]
    >>> k = 3
    >>> ndcg_at_k_with_controls(y_true, y_pred, k)
    0.8192

    References
    ----------
    .. [1] Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation 
           of IR techniques.
    .. [2] Wang, Y., et al. (2013). Theoretical analysis of normalized 
           discounted cumulative gain (NDCG) ranking measures.
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True, 
                                        allow_nan=True, multi_output =True)
    epsilon = check_epsilon(epsilon, y_true, scale_factor=1e-10)

    def dcg_at_k(scores, true_relevances, k, epsilon):
        """
        Calculate Discounted Cumulative Gain at rank K.

        Parameters:
        - scores: list or ndarray, predicted relevance scores (clipped).
        - true_relevances: list or ndarray, true relevance scores.
        - k: int, rank at which to calculate DCG.
        - epsilon: float, small value to prevent division by zero.

        Returns:
        - float, the DCG score at K.
        """
        scores = np.asfarray(scores)[:k]
        discounts = np.log2(np.arange(2, len(scores) + 2)) + epsilon
        return np.sum(scores / discounts)
    
    # Handle NaN values according to nan_policy
    # Implementation would involve filtering or warnings based on nan_policy
    # Handle NaN values according to nan_policy
    y_true, y_pred, *opt_weights = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights =sample_weight ) 
    sample_weight = opt_weights[0] if opt_weights else sample_weight 
    
    # Convert k to int if it's float or a NumPy numeric type
    if not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError("k must be an integer or floating point number.")
    k = int(k)

    # Check that k is a positive integer after conversion
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    # Initialize NDCG scores list
    ndcg_scores = []

    for true_items, pred_items in zip(y_true, y_pred):
        # Clip predicted relevance scores
        pred_items_clipped = np.maximum(pred_items, clip_value)

        # Compute DCG@k
        dcg_k = dcg_at_k(pred_items_clipped, true_items, k, epsilon)

        # Compute IDCG@k
        idcg_k = dcg_at_k(sorted(true_items, reverse=True), true_items, k, epsilon)

        # Calculate NDCG@k and append to scores list
        ndcg_score = dcg_k / idcg_k if idcg_k > 0 else 0
        ndcg_scores.append(ndcg_score)

    # Aggregate NDCG scores. Apply weights if provided and adjust based 
    # on `multioutput` setting.
    if sample_weight is not None:
        # Calculate a weighted average if sample weights are provided.
        aggregated_ndcg_score = np.average(ndcg_scores, weights=sample_weight)
    else:
        # Use a simple mean for aggregation when no weights are specified.
        aggregated_ndcg_score = np.mean(ndcg_scores)
    
    # Return the aggregated score or raw scores depending on 
    # `multioutput` preference.
    multioutput= validate_multioutput(multioutput)
    if multioutput == 'uniform_average':
        return aggregated_ndcg_score
    else:  # multioutput == 'raw_values'
        return np.array(ndcg_scores)

def mean_reciprocal_score(
    y_true, y_pred, *, 
    sample_weight=None,
    nan_policy='propagate',
    epsilon=1e-10,
    zero_division='warn',
    multioutput='uniform_average'
):
    """
    Compute the Mean Reciprocal Rank (MRR) for ranking predictions, with support 
    for handling missing data, applying sample weights, and controlling zero division 
    behavior.

    The MRR is a statistic measure for evaluating any process that produces a list 
    of possible responses to a sample of queries, ordered by the probability of 
    correctness. The reciprocal rank of a query response is the multiplicative 
    inverse of the rank of the first correct answer.

    Parameters
    ----------
    y_true : array-like
        True labels or relevant items. Must be a 1D array of the same length as y_pred.
    y_pred : array-like of list
        Predicted rankings for each query. Each element must be a list of items ranked 
        by predicted relevance.
    sample_weight : array-like of shape (n_samples,), optional
        Weights for each sample in the calculation of the mean.
    nan_policy : {'propagate', 'raise', 'omit'}, default='propagate'
        Policy to handle NaNs in the input. If 'omit', ignores samples with NaNs.
    epsilon : float, default=1e-10
        Small value to prevent division by zero in reciprocal rank computation.
    zero_division : {'warn', 'ignore'}, default='warn'
        How to handle scenarios where the correct label is not among the predictions.
    multioutput : {'uniform_average', 'raw_values'}, default='uniform_average'
        Aggregation method for multiple outputs or ranks.

    Returns
    -------
    float or np.ndarray
        The mean reciprocal rank. If `multioutput` is 'raw_values', returns an array
        of reciprocal ranks for each sample.

    Notes
    -----
    The MRR is defined as:

    .. math:: \text{MRR} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\text{rank}_i}

    where \(Q\) is the number of queries, \(\text{rank}_i\) is the rank of the first 
    correct answer for the \(i\)-th query. The rank is 1-indexed. If the correct answer 
    does not appear in the predictions, \(\text{rank}_i\) is set to \(\infty\) (or handled 
    according to `zero_division`).

    This metric is particularly useful in search engine optimization and recommendation 
    systems where the position of the relevant item (e.g., document, product) within 
    the search results or recommendations list is critical [2]_.

    Examples
    --------
    >>> from gofast.metrics import mean_reciprocal_rank
    >>> y_true = [2, 1]
    >>> y_pred = [[1, 2, 3], [1, 2, 3]]
    >>> mean_reciprocal_rank(y_true, y_pred)
    0.75

    See Also
    --------
    average_precision_score : Compute average precision (AP) from prediction scores.
    ndcg_score : Compute Normalized Discounted Cumulative Gain.

    References
    ----------
    .. [1] Voorhees, E. M. (1999). The TREC-8 Question Answering Track Report. In 
           TREC.
    .. [2] Craswell, N. (2009). Mean Reciprocal Rank. In Encyclopedia of Database 
           Systems.
    """

    # Ensure y_true and y_pred are numpy arrays for consistency
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True, allow_nan=True, multi_output =True
    )
    # check epsilon 
    epsilon = check_epsilon(epsilon, y_true)
    # Apply nan_policy
    y_true, y_pred, *opt_sample_weight= validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights= sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight 

    # Compute reciprocal ranks with epsilon to prevent zero division
    reciprocal_ranks = []
    for idx, true_label in enumerate(y_true):
        match_indices = np.where(y_pred[idx] == true_label)[0]
        if match_indices.size > 0:
            rank = 1 / (match_indices[0] + 1 + epsilon)
        else:
            rank = 0
            if zero_division == 'warn':
                warnings.warn("Zero division encountered in"
                              " reciprocal rank computation.")
        reciprocal_ranks.append(rank)

    # Apply sample weights
    if sample_weight is not None:
        weighted_reciprocal_ranks = np.average(
            reciprocal_ranks, weights=sample_weight)
    else:
        weighted_reciprocal_ranks = np.mean(reciprocal_ranks)

    # Handle multioutput
    multioutput = validate_multioutput(multioutput)
    return ( 
        np.array(reciprocal_ranks) if  multioutput == 'raw_values' 
        else weighted_reciprocal_ranks
        )

def jaccard_flex(
    y_true, y_pred, *, 
    labels=None,
    pos_label=1, 
    average='binary', 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn', 
    multioutput='average_uniform'
):
    """
    Compute the Jaccard Similarity Coefficient, extending functionality to 
    support detailed output and handling of multi-class/multi-label cases.

    This metric measures the similarity and diversity between sample sets, 
    making it useful in various fields including ecology, gene sequencing 
    analysis, and machine learning for clustering accuracy evaluation [1]_.

    Parameters
    ----------
    y_true : array-like
        True labels or binary indicators of membership.
    y_pred : array-like
        Predicted labels or binary indicators of membership.
    labels : array-like, optional
        The set of labels to include when `average` is not 'binary', 
        and their order if `average` is None.
    pos_label : int or str, default=1
        The label of the positive class.
    average : {'binary', 'micro', 'macro', 'samples', 'weighted'}, default='binary'
        Determines the type of averaging performed on the data.
    detailed : bool, default=False
        If True, return detailed metrics in the output.
    scale_errors : bool, default=False
        If True, scale intersection and union by the mean value of `y_true`.
    epsilon : float or 'auto', default='auto'
        A small value to avoid division by zero. Automatically determined if 'auto'.
    zero_division : str or int, default='warn'
        Sets the value to return when there is a zero division.
    multioutput : {'average_uniform', 'raw_values'}, default='average_uniform'
        Defines how to aggregate scores for multiple outputs.

    Returns
    -------
    Bunch
        A bunch containing the Jaccard score, and if `detailed` is True, additional
        metrics like intersection and union sizes, potentially scaled by `scale_errors`.

    Notes
    -----
    The Jaccard Similarity Coefficient (J) is defined as:

    .. math::
        J = \\frac{|y_{\\text{true}} \\cap y_{\\text{pred}}|}{|y_{\\text{true}} \\cup y_{\\text{pred}}|}

    This function supports both binary and multi-class/multi-label tasks by binarizing 
    labels in a multi-class/multi-label setting [2]_.

    Examples
    --------
    >>> from gofast.metrics import jaccard_flex 
    >>> y_true = [0, 1, 2, 1, 2, 0]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> jaccard_flex(y_true, y_pred, average='macro')
    Bunch(score=0.333...)

    References
    ----------
    .. [1] Scikit-learn documentation on Jaccard score:
          https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    .. [2] Wikipedia entry for the Jaccard index:
           https://en.wikipedia.org/wiki/Jaccard_index

    See Also
    --------
    sklearn.metrics.jaccard_score : Jaccard similarity coefficient score.
    
    """
    # Ensure inputs are numpy arrays for consistency
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True, allow_nan=True, multi_output =True
    )
    # Initialize Bunch object to store results
    b = MetricFormatter(title="Jaccard Results" )
    # Determine epsilon dynamically if required
    epsilon = check_epsilon(epsilon, y_pred, scale_factor= np.finfo(float).eps)

    # Handle multi-class/multi-label cases
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Binarize labels in a multi-class/multi-label setting
    if average != 'binary' or len(labels) > 2:
        y_true = label_binarize(y_true, classes=labels)
        y_pred = label_binarize(y_pred, classes=labels)
    
    # Calculate Jaccard score
    b.score = jaccard_score(y_true, y_pred, labels=labels, pos_label=pos_label,
                          average=average, zero_division=zero_division)
    
    # Store basic result
    # Calculate detailed statistics if requested
    if detailed:
        intersect = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        b.intersection = np.sum(intersect, axis=0)
        b.union = np.sum(union, axis=0)
        if scale_errors:
            # Scale the intersection and union by the mean value of y_true
            mean_val = np.mean(y_true, axis=0) + epsilon  # Avoid division by zero
            b.scaled_intersection = b.intersection / mean_val
            b.scaled_union = b.union / mean_val
    
    # Handle multioutput
    multioutput = validate_multioutput(multioutput)
    if multioutput == 'raw_values':
        b.raw_scores = jaccard_score(
            y_true, y_pred, labels=labels, pos_label=pos_label, 
            average=None, zero_division=zero_division)
    return b

def balanced_accuracy_score(
    y_true, y_pred, *, 
    sample_weight=None, 
    normalize=False, 
    strategy='ovr', 
    epsilon=1e-15, 
    zero_division=0,
    multioutput='uniform_average', 
    nan_policy='propagate'
):
    """
    Calculate the balanced accuracy score, a metric that accounts for class 
    imbalance by averaging sensitivity and specificity across classes. 
    
    This function supports both binary and multiclass classification tasks 
    and offers strategies for handling multiclass scenarios through 
    One-vs-Rest (OVR) or One-vs-One (OVO) approaches.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values. For binary and multiclass
        classification, `y_true` should be a 1D array of class labels. For
        multioutput and multilabel classification, `y_true` can be a 2D array
        with each column representing a class.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated targets as returned by a classifier. The shape and type
        should match those of `y_true`. For classifications that involve
        ranking or probability estimates, `y_pred` might require preprocessing
        to fit into the expected binary or multiclass form.

    sample_weight : array-like of shape (n_samples,), optional
        Sample weights. If None, all samples are given equal weight. If provided,
        these should be positive values that sum to 1 for proper normalization.
        Sample weights modify the influence of individual samples on the computed
        metric, emphasizing or de-emphasizing their impact.

    normalize : bool, optional, default=False
        If True, each element of the confusion matrix is divided by the sum of
        elements in its corresponding row. This normalization can be helpful
        when comparing results across datasets with different total numbers of
        samples.

    strategy : {'ovr', 'ovo'}, default='ovr'
        Strategy to compute the score for multiclass classification. The 'ovr'
        (One-vs-Rest) considers each class against all other classes to compute
        an average score. The 'ovo' (One-vs-One) strategy computes the average
        score over all pairs of classes. Not applicable for binary classification.

    epsilon : float, default=1e-15
        A small constant added to denominators to avoid division by zero in
        cases where a class is either perfectly predicted or not present at all
        in `y_pred`. This is particularly useful in metrics like specificity or
        sensitivity, where the denominators can be zero due to the absence of
        positive or negative samples.

    zero_division : {0, 1}, default=0
        Determines the value to return when there is a division by zero during
        the computation. If set to 0, a score of 0 is returned whenever division
        by zero occurs. If set to 1, a score of 1 is returned. This parameter
        helps manage undefined metrics in datasets with skewed class distributions.

    multioutput : {'uniform_average', 'raw_values'}, default='uniform_average'
        Defines the method to aggregate scores for multioutput (multilabel or
        multi-class) data. 'uniform_average' calculates the mean of the scores
        for all outputs, treating each output equally. 'raw_values' returns a
        score for each output without averaging.

    nan_policy : {'propagate', 'raise', 'omit'}, default='propagate'
        Determines how to handle the occurrence of NaNs (Not a Number) in the
        input labels `y_true` and `y_pred`. If 'propagate', NaNs are allowed to
        propagate through the metric calculation, possibly resulting in NaN
        scores. If 'raise', an error is raised upon detecting NaN values. If
        'omit', rows with NaNs are excluded from the calculation, potentially
        resulting in a subset of the original data being evaluated.

   Returns
    -------
    float or ndarray
        The Balanced Accuracy score. For binary and multiclass classification
        scenarios without `multioutput='raw_values'`, this will be a single
        float value representing the overall balanced accuracy across all
        classes. If `multioutput='raw_values'`, an array of balanced accuracy
        scores for each output (class or sample, depending on the context)
        is returned.

    Notes
    -----
    Balanced Accuracy (BA) compensates for the imbalance in the distribution
    of the classes. It is calculated as the average of the proportion of
    correct predictions in each class relative to the number of instances
    in that class. In binary classification, it can be interpreted as the
    average of recall obtained on each class:

    .. math::
        BA = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)

    where \(TP\), \(TN\), \(FP\), and \(FN\) represent the number of true
    positives, true negatives, false positives, and false negatives,
    respectively.

    For multiclass scenarios, Balanced Accuracy extends this concept by
    considering each class as the positive class in turn and calculating
    the average recall across all classes:

    .. math::
        BA = \frac{1}{N} \sum_{i=1}^{N} \frac{TP_i}{TP_i + FN_i}

    where \(N\) is the number of classes, \(TP_i\) and \(FN_i\) are the
    number of true positives and false negatives for class \(i\),
    respectively. The `strategy` parameter ('ovr' or 'ovo') influences
    how these metrics are averaged across classes.

    The use of Balanced Accuracy helps to provide a clearer picture of the
    model's performance across all classes, especially in cases where one
    or more classes are underrepresented in the dataset.

    References
    ----------
    .. [1] Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The
           balanced accuracy and its posterior distribution. 2010 20th International
           Conference on Pattern Recognition, 3121-3124.
    .. [2] Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance
           measures for classification tasks. Information Processing & Management,
           45(4), 427-437.

    See Also
    --------
    accuracy_score : Calculate accuracy classification score.
    precision_score : 
        Calculate precision score for binary and multiclass classification tasks.
    recall_score : 
        Calculate recall score to measure the ability of a classifier to
        find all positive samples.
    f1_score :
        Calculate the F1 score, the weighted average of precision and recall.

    Examples
    --------
    >>> from gofast.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.75

    Multiclass example with One-vs-Rest strategy:

    >>> y_true = [0, 1, 2, 2, 1]
    >>> y_pred = [0, 2, 2, 1, 1]
    >>> balanced_accuracy_score(y_true, y_pred, strategy='ovr')
    0.6666666666666666

    Multiclass example with One-vs-One strategy:

    >>> balanced_accuracy_score(y_true, y_pred, strategy='ovo')
    0.6666666666666666

    Using sample weights:

    >>> weights = [0.5, 1, 1, 1, 1]
    >>> balanced_accuracy_score(y_true, y_pred, sample_weight=weights)
    0.75

    Note: The output value may differ due to the calculation method and rounding.

    """
    # validation 
    # --
    # Ensure y_true and y_pred are valid and
    # have consistent lengths
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, 
        allow_nan=True, multi_output=True )
    # Check that y_true and y_pred are suitable for classification
    y_true, y_pred = check_classification_targets(
        y_true, y_pred, strategy="custom logic")
    # validate epsilon 'auto' or float
    epsilon = check_epsilon(epsilon , y_true, scale_factor=1e-15)
    multioutput = validate_multioutput( multioutput)
    # validate strategy input 
    strategy =normalize_string(
       strategy, target_strs=['ovr', 'ovo'], match_method='contains', 
       return_target_only= True, raise_exception= True, 
       error_msg=("strategy parameter must be either 'ovr' or 'ovo'") 
       ) 
    # validate nan_policy 
    y_true, y_pred, *opt_sample_weight = validate_nan_policy(
        nan_policy, y_true, y_pred, sample_weights = sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight 
    
    # compute balanced score 
    score = compute_balance_accuracy(
        y_true= y_true, y_pred=y_pred, 
        epsilon = epsilon, 
        zero_division= zero_division, 
        strategy =strategy, 
        normalize=normalize, 
        sample_weight= sample_weight
        ) 

    if multioutput == 'uniform_average':
        # Average the scores if multioutput is 'uniform_average'
        return np.average(score) 
  
    return score

def fetch_sklearn_scorers(scorer):
    """
    Fetches a scoring function from scikit-learn's predefined scorers or a custom
    callable scorer based on the provided identifier. If the identifier matches
    a predefined scorer name, that scorer is returned. If the identifier is a
    custom callable scorer, it is returned directly. If no match is found, a
    `ValueError` is raised.

    Parameters
    ----------
    scorer : str or callable
        The name of the predefined scikit-learn scorer or a custom scorer function.
        If a string is provided, it should match one of scikit-learn's predefined
        scoring function names. If a callable is provided, it is used directly as the scorer.

    Returns
    -------
    scorer_function : callable
        The scoring function corresponding to the provided `scorer` identifier.
        This function can be used directly to evaluate the performance of models.

    Raises
    ------
    ValueError
        If the provided `scorer` identifier does not match any predefined scorers
        in scikit-learn and is not a callable function.

    Examples
    --------
    Fetch a predefined scoring function by name:

    >>> from gofast.metrics import fetch_sklearn_scorers
    >>> accuracy_scorer = fetch_sklearn_scorers('accuracy')
    >>> print(accuracy_scorer)
    make_scorer(accuracy_score)

    Fetch a custom callable scorer:

    >>> custom_scorer = lambda estimator, X, y: np.mean(estimator.predict(X) == y)
    >>> fetched_scorer = fetch_sklearn_scorers(custom_scorer)
    >>> print(fetched_scorer)
    <function <lambda> at 0x...>

    Attempt to fetch a non-existing scorer:

    >>> fetch_sklearn_scorers('non_existing_scorer')
    ValueError: Scorer 'non_existing_scorer' not recognized. Must be one of [...]
    """

    predefined_scorers = {
        "classification_report": classification_report,
        "precision_recall": precision_recall_curve,
        "confusion_matrix": confusion_matrix,
        "precision": precision_score,
        "accuracy": accuracy_score,
        "mse": mean_squared_error,
        "recall": recall_score,
        "auc": roc_auc_score,
        "roc": roc_curve,
        "f1": f1_score,
    }

    if isinstance(scorer, str):
        scorer_name = scorer.lower()
        if scorer_name in predefined_scorers:
            return predefined_scorers[scorer_name]
        try:
            return get_scorer(scorer_name)
        except ValueError:
            raise ValueError(f"Scorer '{scorer}' not recognized. Must be one"
                             f" of {list(predefined_scorers.keys())} or a valid"
                             " scorer name available in scikit-learn.")

    elif callable(scorer):
        return scorer
    else:
        raise ValueError("Scorer must be a string name of a predefined scorer or a callable.")

# Mapping of metric names to their respective functions
SCORERS = {
    "precision_recall_tradeoff": precision_recall_tradeoff,
    "roc_tradeoff": roc_tradeoff,
    "mean_squared_log_error": mean_squared_log_error,
    "balanced_accuracy": balanced_accuracy,
    "balanced_accuracy_score": balanced_accuracy_score,
    "information_value": information_value,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "explained_variance_score": explained_variance_score,
    "median_absolute_error": median_absolute_error,
    "max_error_score": max_error_score,
    "mean_poisson_deviance": mean_poisson_deviance,
    "mean_gamma_deviance": mean_gamma_deviance,
    "mean_absolute_deviation": mean_absolute_deviation,
    "dice_similarity_score": dice_similarity_score,
    "gini_score": gini_score,
    "hamming_loss": hamming_loss,
    "fowlkes_mallows_score": fowlkes_mallows_score,
    "root_mean_squared_log_error": root_mean_squared_log_error,
    "mean_percentage_error": mean_percentage_error,
    "percentage_bias": percentage_bias,
    "spearmans_rank_score": spearmans_rank_score,
    "precision_at_k": precision_at_k,
    "ndcg_at_k": ndcg_at_k,
    "mean_reciprocal_score": mean_reciprocal_score,
    "geo_information_value": geo_information_value,
    "adjusted_r2_score": adjusted_r2_score,
    "likelihood_score": likelihood_score
}

def get_scorer(scoring,  include_sklearn=True):
    """
    Fetches a scoring function from gofast's predefined scorers or, optionally,
    scikit-learn's scoring functions, based on the `scoring` argument. It allows
    the use of both named scoring strategies and custom callable scoring functions.
    
    Parameters
    ----------
    scoring : str or callable
        The name of the predefined scoring function as a string, or a custom
        callable scoring function.
    include_sklearn : bool, default=True
        If True, includes scikit-learn's predefined scoring functions in the
        search scope in addition to gofast's predefined scorers.
    
    Returns
    -------
    scorer : callable
        The scoring function.
    
    Raises
    ------
    ValueError
        If `scoring` is a string not found among the predefined scorers or
        if a callable `scoring` function is not recognized as a valid scorer.
    
    Examples
    --------
    Using a predefined scorer name:
    
    >>> from gofast.metrics import get_scorer 
    >>> scorer = get_scorer('accuracy')
    >>> print(scorer)
    <function accuracy_score at ...>
    
    Using a custom callable scoring function:
    
    >>> def custom_scorer(y_true, y_pred):
    ...     return np.mean(y_true == y_pred)
    >>> scorer = get_scorer(custom_scorer)
    >>> print(scorer)
    <function custom_scorer at ...>
    
    Including scikit-learn scorers:
    
    >>> scorer = get_scorer('neg_mean_squared_error', include_sklearn=True)
    >>> print(scorer)
    make_scorer(mean_squared_error, greater_is_better=False, ...)
    
    Notes
    -----
    If `include_sklearn` is True and `scoring` is a named scoring strategy not
    found among gofast's predefined scorers, this function will attempt to fetch
    the scorer from scikit-learn's scoring functions.
    """
    # Validate if the callable is a function and not a class or method
    import types 
       
    if callable(scoring):
        if not isinstance(scoring, types.FunctionType):
            raise ValueError("Scoring must be a function. Methods or classes"
                             " are not supported.")
        # Validate if the callable is among predefined scorers or sklearn scorers
        scorer_name = scoring.__name__
        if scorer_name in SCORERS:
            return scoring
        if include_sklearn:
            from sklearn.metrics import get_scorer_names as sklearn_get_scorer_names
            if scorer_name in sklearn_get_scorer_names():
                return scoring
        raise ValueError(f"The callable scorer '{scorer_name}' is not recognized"
                         " among gofast or sklearn scorers.")

    # Attempt to fetch scorer from gofast's predefined list
    scorer = SCORERS.get(scoring)
    if scorer is not None:
        return scorer

    # Optionally include sklearn scorers in the search
    if include_sklearn:
        from sklearn.metrics import get_scorer as sklearn_get_scorer
        try:
            return sklearn_get_scorer(scoring)
        except ValueError:
            pass

    # Compile a list of all valid scorers for error message
    valid_scorers = list(SCORERS.keys())
    if include_sklearn:
        from sklearn.metrics import get_scorer_names as sklearn_get_scorer_names
        valid_scorers += sklearn_get_scorer_names()

    raise ValueError(f"Scorer '{scoring}' is not recognized."
                     f" Available scorers are: {valid_scorers}.")


def get_scorer_names(include_sklearn=True):
    """
    Retrieves a list of the names of all predefined scoring functions from both
    gofast and, optionally, scikit-learn.

    This function provides a convenient way to explore available scoring functions,
    facilitating the selection of appropriate metrics for evaluating machine learning
    models. By integrating scoring functions from both gofast and scikit-learn, it
    offers a comprehensive view of the metrics that can be used for model assessment.

    Parameters
    ----------
    include_sklearn : bool, optional
        If True, includes scikit-learn's predefined scorers in the returned list.
        Defaults to False.

    Returns
    -------
    list of str
        A sorted list of unique scoring function names from gofast and, optionally,
        scikit-learn.

    Examples
    --------
    >>> from gofast.metrics import get_scorer_names 
    >>> gofast_scorers = get_scorer_names()
    >>> print(gofast_scorers)
    ['accuracy', 'f1_score', 'precision', ...]

    >>> all_scorers = get_scorer_names(include_sklearn=True)
    >>> print(all_scorers)
    ['accuracy', 'adjusted_rand_score', 'auc', 'f1_score', ...]

    Note
    ----
    Including scikit-learn scorers requires scikit-learn to be installed in your
    environment. This function dynamically aggregates scorer names, reflecting
    the current capabilities of both libraries.

    See Also
    --------
    sklearn.metrics.get_scorer_names : 
        Function to list all scorer names available in scikit-learn.
    gofast.metrics : Module containing custom scoring functions.
    """
    scorers = list(SCORERS.keys())
    if include_sklearn:
        from sklearn.metrics import get_scorer_names as sklearn_get_scorer_names
        scorers.extend(sklearn_get_scorer_names())
    # Use sorted and set to remove duplicates and sort the list.   
    return sorted(set(scorers))

def fetch_scorer_functions():
    """
    Retrieves a dictionary of scorer functions from both scikit-learn and gofast.
    
    Function scans through all scoring and error functions defined in
    sklearn.metrics, filtering out private functions (those starting with '_') and
    selecting those with 'score' or 'error' in their names. It combines these with
    the predefined scoring functions from gofast, providing a comprehensive 
    dictionary of available scoring methods.

    Returns
    -------
    dict
        A dictionary where keys are the names of the scoring functions and values
        are the scoring function objects themselves.

    Examples
    --------
    >>> from gofast.metrics import fetch_scorer_functions
    >>> scorer_functions = fetch_scorer_functions()
    >>> print(list(scorer_functions.keys()))
    ['accuracy_score', 'adjusted_rand_score', 'mean_squared_error', ...]

    Note
    ----
    This function is particularly useful for dynamically accessing and utilizing
    scoring functions across different modules, facilitating flexible evaluation
    strategies in machine learning workflows.

    See Also
    --------
    sklearn.metrics : Module in scikit-learn containing scoring and error functions.
    gofast.metrics : Module in gofast containing custom scoring functions.
    """
    import inspect
    import sklearn.metrics as sklearn_metrics  
    scorer_functions = {}
    for name, obj in inspect.getmembers(sklearn_metrics):
        if inspect.isfunction(obj) and not name.startswith('_') and (
                "score" in name or "error" in name):
            scorer_functions[name] = obj
    # Include gofast metrics gofast's custom scoring functions
    scorer_functions.update(SCORERS)  
                
    return scorer_functions

def fetch_scorers(metric_name, /):
    """
    Retrieves a scorer function based on a flexible metric name, considering 
    potential suffixes. The function handles specific metrics like 
   'balanced_accuracy'  strictly to avoid mismatches with similar names.

    Parameters
    ----------
    metric_name : str
        The base name of the metric, which may or may not include suffixes such as
        '_score', '_error', '_deviance', or '_deviation'.

    Returns
    -------
    function
        The scorer function corresponding to the metric name, if available.

    Raises
    ------
    ValueError
        If no matching scorer is found.

    Examples
    --------
    >>> from gofast.metrics import fetch_scorers
    >>> fetch_scorers('accuracy')
    <function accuracy_score at ...>

    >>> fetch_scorers('mean_poisson')
    <function mean_poisson_deviance at ...>

    Notes
    -----
    The function is designed to be flexible by using regular expressions to match 
    the metric name against various possible suffixes. This design ensures the 
    function can retrieve scorer functions accurately even if only partial 
    metric names are provided. The specific case of 'balanced_accuracy' is 
    handled to strictly match without confusion from 'balanced_accuracy_score',
    demonstrating an example of customized matching logic.
    
    See also 
    --------
    gofast.metrics.fetch_scorer_functions: 
        Retrieves a dictionary of scorer functions from both scikit-learn and 
        gofast
    gofast.make_scorers: 
        Creates a scorer callable for gofast.metrics that can be used in model 
        evaluation.
    gofast.metrics.get_scorer_names: 
        Retrieves a list of the names of all predefined scoring functions from both
        gofast and, optionally, scikit-learn.
    """
    import re
    import inspect
    import sklearn.metrics as sklearn_metrics
    
    # Special handling for 'balanced_accuracy' to strictly avoid confusion
    if metric_name == "balanced_accuracy":
        if "balanced_accuracy" in SCORERS:
            return SCORERS["balanced_accuracy"]
        else:
            raise ValueError("No scorer found for 'balanced_accuracy'")
    
    # Build a regex pattern to match the metric name with optional suffixes
    pattern = re.compile(rf"^{re.escape(metric_name)}(_score|_error|_deviance|_deviation)?$")

    # Fetch all scorers from sklearn and gofast custom scorers
    scorer_functions = {name: obj for name, obj in inspect.getmembers(sklearn_metrics)
                        if inspect.isfunction(obj) and not name.startswith('_')
                        and ("score" in name or "error" in name)}
    scorer_functions.update(SCORERS)

    # Match and return the appropriate scorer function
    for name, func in scorer_functions.items():
        if pattern.match(name):
            return func

    # If no scorer is found, raise an exception
    raise ValueError(f"No scorer found for '{metric_name}'")

def make_scorer(
    score_func, *, 
    greater_is_better=True, 
    needs_proba=False, 
    needs_threshold=False, 
    **kwargs
    ):
    """
    Creates a scorer callable for gofast.metrics that can be used in model evaluation.

    This function wraps scoring functions from the gofast.metrics module 
    (or any custom scoring function) to make them compatible with scikit-learn's
    model evaluation tools, such as cross-validation and grid search.

    Parameters
    ----------
    score_func : callable
        A scoring function with signature `score_func(y_true, y_pred, **kwargs)` where:
            - `y_true` is an array-like of true labels,
            - `y_pred` is an array-like of predicted labels or probabilities 
               (depending on `needs_proba`),
            - `**kwargs` are additional arguments to the scoring function.
    greater_is_better : bool, optional
        Whether score_func is a score function, meaning high is good, or a 
        loss function, meaning low is good. By default, it's assumed to be a
        score function (True).
    needs_proba : bool, optional
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
        Set to True if `score_func` requires probability estimates instead of 
        just labels.
    needs_threshold : bool, optional
        Whether the score function requires a decision threshold 
        (only meaningful for binary classification).
    **kwargs : additional arguments
        Additional parameters to be passed to the scoring function.

    Returns
    -------
    scorer : callable
        A callable scorer that takes two arguments `estimator, X` where `X` 
        is the data to be passed to `estimator.predict` or 
        `estimator.predict_proba`, and returns a float representing the score.

    Examples
    --------
    >>> from gofast.metrics import make_scorer
    >>> pbias_scorer = make_scorer(percentage_bias)
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> scores = cross_val_score(clf, X, y, scoring=pbias_scorer)
    >>> print(scores)
    
    >>> from gofast.metrics import information_value 
    >>> scores = cross_val_score(clf, X, y, scoring=iv_scorer)
    >>> print(scores)
    Note
    ----
    This is a wrapper around `sklearn.metrics.make_scorer`, designed to facilitate the use of custom
    scoring functions from the gofast.metrics module in scikit-learn's model evaluation process.
    """
    from sklearn.metrics import make_scorer as sklearn_make_scorer 

    return sklearn_make_scorer(
        score_func, 
        greater_is_better=greater_is_better, 
        needs_proba=needs_proba, 
        needs_threshold=needs_threshold,
        **kwargs
    )
  

            
