# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
`util` module provides utility functions for machine learning estimators. 
It includes tools for scaling, problem detection, weight determination, 
memory depth estimation, hyperparameter optimization, model selection, and 
estimator validation.
"""

from __future__ import annotations
import warnings 
import re
from scipy.sparse import issparse
import numpy as np 

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

try:from sklearn.utils import type_of_target
except: from ..tools.coreutils import type_of_target 
from ..api.types import DataFrame, Series, Tuple, Union, Dict, Optional 
from ..tools.validator import check_array, check_X_y
from ..tools.validator import validate_fit_weights, get_estimator_name
from ..tools.validator import validate_positive_integer 

__all__=[
     'activator','apply_scaling','build_named_estimators', 'detect_problem_type',
     'determine_weights','estimate_memory_depth','fit_with_estimator',
     'get_default_meta_estimator','normalize_sum','optimize_hyperparams',
     'select_best_classification_model','select_best_model',
     'select_best_regression_model','select_default_estimator',
     'validate_memory_depth'
 ]
    
def activator(z, activation='sigmoid', alpha=1.0, clipping_threshold=250):
    """
    Apply the specified activation function to the input array `z`.

    Parameters
    ----------
    z : array-like
        Input array to which the activation function is applied.
    
    activation : str or callable, default='sigmoid'
        The activation function to apply. Supported activation functions are:
        'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 'tanh', 'softmax'.
        If a callable is provided, it should take `z` as input and return the
        transformed output.

    alpha : float, default=1.0
        The alpha value for activation functions that use it (e.g., ELU).

    clipping_threshold : int, default=250
        Threshold value to clip the input `z` to avoid overflow in activation
        functions like 'sigmoid' and 'softmax'.

    Returns
    -------
    activation_output : array-like
        The output array after applying the activation function.

    Notes
    -----
    The available activation functions are defined as follows:

    - Sigmoid: :math:`\sigma(z) = \frac{1}{1 + \exp(-z)}`
    - ReLU: :math:`\text{ReLU}(z) = \max(0, z)`
    - Leaky ReLU: :math:`\text{Leaky ReLU}(z) = \max(0.01z, z)`
    - Identity: :math:`\text{Identity}(z) = z`
    - ELU: :math:`\text{ELU}(z) = \begin{cases}
                  z & \text{if } z > 0 \\
                  \alpha (\exp(z) - 1) & \text{if } z \leq 0
                \end{cases}`
    - Tanh: :math:`\tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}`
    - Softmax: :math:`\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}`

    Examples
    --------
    >>> from gofast.estimators.util import activator
    >>> z = np.array([1.0, 2.0, -1.0, -2.0])
    >>> activator(z, activation='relu')
    array([1.0, 2.0, 0.0, 0.0])
    
    >>> activator(z, activation='tanh')
    array([ 0.76159416,  0.96402758, -0.76159416, -0.96402758])
    
    >>> activator(z, activation='softmax')
    array([[0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986],
           [0.25949646, 0.70682242, 0.02817125, 0.00550986]])

    See Also
    --------
    GradientDescentBase : Base class for gradient descent-based algorithms.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
           MIT Press. http://www.deeplearningbook.org
    """
    clipping_threshold = validate_positive_integer(
        clipping_threshold, "clipping_threshold"
    )
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == 'sigmoid':
            z = np.clip(z, -clipping_threshold, clipping_threshold)
            return 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'leaky_relu':
            return np.where(z > 0, z, z * 0.01)
        elif activation == 'identity':
            return z
        elif activation == 'elu':
            return np.where(z > 0, z, alpha * (np.exp(z) - 1))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'softmax':
            z = np.clip(z, -clipping_threshold, clipping_threshold)
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    elif callable(activation):
        return activation(z)
    else:
        raise ValueError("Activation must be a string or a callable function")
 
def detect_problem_type(problem, y, multi_output=False, estimator=None):
    """
    Detect and set the type of machine learning problem based on the target `y`.

    This function determines whether the given machine learning problem is a 
    classification, regression, or multi-label task. When `problem` is set to 
    'auto', it utilizes scikit-learn's `type_of_target` function to automatically 
    identify the nature of the problem. The function normalizes the problem type 
    string, validates it, and ensures it is one of the supported types. It also 
    handles multi-output scenarios and raises appropriate errors when necessary.

    Parameters
    ----------
    problem : str
        The initial problem type. Accepted values are 'auto', 'binary', 
        'multiclass', 'classification', 'continuous', 'regression', or 
        'multilabel-indicator'. If set to 'auto', the function will automatically 
        detect the problem type based on `y`.

    y : array-like
        The target values. This can be a list, numpy array, pandas Series, or 
        any array-like structure containing the target labels or values.

    multi_output : bool, default=False
        Whether the task is a multi-output (multi-label) problem. If set to 
        `True`, the function will handle multi-label classification problems 
        and allow the problem type 'multilabel-indicator'.

    estimator : object, optional
        The estimator object. If provided, it will be used in the error message 
        when a multi-label indicator is detected but `multi_output` is not 
        supported.

    Returns
    -------
    detected_problem : str
        The detected and validated problem type. Possible values are 
        'classification', 'regression', or 'multilabel-indicator'.

    Raises
    ------
    ValueError
        If the detected or provided problem type is not supported, or if 
        multi-output is not tolerated but a multi-label indicator is detected.

    Examples
    --------
    >>> from sklearn.datasets import load_iris, fetch_california_housing
    >>> from sklearn.preprocessing import MultiLabelBinarizer
    >>> from gofast.estimators.util import detect_problem_type
    
    # For a classification problem
    >>> iris = load_iris()
    >>> X_iris, y_iris = iris.data, iris.target
    >>> problem = detect_problem_type('auto', y_iris)
    >>> print(problem)
    'classification'
    
    # For a regression problem
    >>> housing = fetch_california_housing()
    >>> X_housing, y_housing = housing.data, housing.target
    >>> problem = detect_problem_type('auto', y_housing)
    >>> print(problem)
    'regression'
    
    # For a multi-label classification problem
    >>> mlb = MultiLabelBinarizer()
    >>> y_multi = mlb.fit_transform([(1, 2), (3,), (1, 2, 3)])
    >>> problem = detect_problem_type('auto', y_multi, multi_output=True)
    >>> print(problem)
    'classification'

    Notes
    -----
    The `detect_problem_type` function is essential for determining the nature 
    of the machine learning task based on the target variable `y`. It uses 
    scikit-learn's `type_of_target` function to identify whether the problem 
    is a binary classification, multiclass classification, regression, or 
    multi-label classification.

    The function normalizes the `problem` parameter to ensure consistency and 
    checks if `multi_output` is `True` to handle multi-label classification 
    scenarios. If the problem type is not supported or if there is a 
    contradiction between `multi_output` and the detected problem type, a 
    `ValueError` is raised with an appropriate message.

    See Also
    --------
    sklearn.utils.multiclass.type_of_target : Determine the type of data indicated 
        by the target.
    
    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """
    # Define the set of classification problem types
    classification_problems = {"binary", "multiclass", "classification"}
    
    # Convert the problem type to lowercase to ensure consistency
    problem = str(problem).lower()
    
    # Automatically detect the problem type if set to 'auto'
    if problem == 'auto':
        problem = type_of_target(y)
    
    # Check for multi-label indicator with multi-output set to False
    if problem == "multilabel-indicator" and not multi_output:
        error_message = "Detected 'multilabel-indicator' while multi-output is not supported"
        if estimator is not None:
            error_message += f" by {get_estimator_name(estimator)}"
        raise ValueError(f"{error_message}.")
    
    # Add 'multilabel-indicator' to classification problems if multi-output is True
    if multi_output:
        classification_problems.add("multilabel-indicator")
    
    # Determine the problem type
    if problem in classification_problems:
        detected_problem = 'classification'
    elif problem in {"continuous", "regression"}:
        detected_problem = 'regression'
    else:
        # Raise an error if the problem type is not supported
        raise ValueError(f"Unsupported task type: {problem}")
    
    return detected_problem

def get_default_meta_estimator(
        meta_estimator: BaseEstimator = None, problem: str = 'regression'
        ) -> BaseEstimator:
    """
    Select and return a default meta estimator if none is provided.

    Parameters
    ----------
    meta_estimator : BaseEstimator, optional
        The meta estimator provided by the user. If None, a default estimator
        will be selected based on the problem type.
    problem : str, default='regression'
        The type of problem. Options are 'regression' for regression problems
        and 'classification' for classification problems.

    Returns
    -------
    meta_estimator : BaseEstimator
        The selected meta estimator. Default is LinearRegression for regression
        problems and LogisticRegression for classification problems.

    Raises
    ------
    ValueError
        If an invalid problem type is provided.

    Examples
    --------
    >>> from gofast.estimators._base import get_default_meta_estimator
    >>> regressor = get_default_meta_estimator(problem='regression')
    >>> classifier = get_default_meta_estimator(problem='classification')
    """
    if meta_estimator is not None:
        if not isinstance(meta_estimator, BaseEstimator):
            raise ValueError("The meta_estimator must be an instance of sklearn.base.BaseEstimator")
        return meta_estimator

    if problem.lower() in ('regression', 'reg'):
        return LinearRegression()
    elif problem.lower() in ('classification', 'class', 'clf'):
        return LogisticRegression()
    else:
        raise ValueError("Invalid problem type. Expected 'regression' or 'classification'.")
        
def fit_with_estimator(
        estimator, X, y, sample_weight=None, apply_weighted_y=False, 
        return_weighted_y=False):
    """
    Fit an estimator with the given data and handle sample weights appropriately.

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Individual weights for each sample.
    apply_weighted_y : bool, default=False
        Whether to adjust `y` by `sample_weight` before fitting if the estimator
        does not accept `sample_weight` directly.
    return_weighted_y : bool, default=False
        Whether to return the modified `y` values along with the fitted estimator.

    Returns
    -------
    estimator : estimator object
        The fitted estimator.
    y : array-like of shape (n_samples,), optional
        The modified target values if `return_weighted_y` is True.

    Notes
    -----
    This function checks if the estimator accepts `sample_weight`. If not, and
    `apply_weighted_y` is True, it adjusts the target values `y` using the
    sample weights before fitting the estimator. This is useful for estimators
    that do not natively support sample weights.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> from gofast.estimators._base import fit_with_estimator
    >>> X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    >>> estimator = LinearRegression()
    >>> fitted_estimator = fit_with_estimator(estimator, X, y)
    """
    # Validate inputs
    X, y = check_X_y(X, y)
    # Validate and process sample weights
    sample_weight = validate_fit_weights(y, sample_weight)

    # Check if the estimator accepts sample_weight parameter
    from inspect import signature
    sig = signature(estimator.fit)
    if 'sample_weight' in sig.parameters:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        if apply_weighted_y:
            _, y_weighted = validate_fit_weights(y, sample_weight, weighted_y=True)
            estimator.fit(X, y_weighted)
        else:
            estimator.fit(X, y)

    return (estimator, y) if return_weighted_y else estimator

def determine_weights(
    base_estimators, 
    X, y, 
    cv=None,
    scoring='neg_mean_squared_error', 
    problem='regression'
    ):
    """
    Determine weights based on the performance of base models.

    Parameters
    ----------
    base_estimators : list of estimators
        List of base estimators to be evaluated. Each estimator should 
        support `fit` and `predict` methods. These estimators will be 
        combined through weighted averaging to form the ensemble model.

    X : array-like of shape (n_samples, n_features)
        The training input samples. These represent the features of the 
        dataset used to train the model. Each row corresponds to a sample,
        and each column corresponds to a feature.

    y : array-like of shape (n_samples,)
        The target values for each sample. These are the true values that
        the model aims to predict. Each element corresponds to the target
        value for a respective sample in `X`.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy. It specifies 
        the number of folds in cross-validation or provides an iterable 
        yielding train/test splits. If None, 5-fold cross-validation is used.

    scoring : str, callable, list/tuple, or dict, default='neg_mean_squared_error'
        A string (see model evaluation documentation) or a scorer callable 
        object / function with signature `scorer(estimator, X, y)`. It defines
        the metric used to evaluate the performance of the base estimators.
        Default is 'neg_mean_squared_error', suitable for regression tasks.
        For classification tasks, it is recommended to use 'accuracy' or 
        another appropriate classification metric.

    problem : str, {'regression', 'classification'}, default='regression'
        Defines the type of problem being solved. If 'regression', it uses 
        the default scoring for regression problems. If 'classification', 
        it changes the default scoring to 'accuracy'.

    Returns
    -------
    weights : array-like of shape (n_estimators,)
        Weights for each base estimator, determined based on performance.
        The weights are proportional to the mean cross-validation scores
        of the estimators. They sum to 1.

    Notes
    -----
    This function evaluates each base estimator using cross-validation and
    assigns weights proportional to their performance (e.g., mean cross-
    validation score). The resulting weights are normalized to sum to 1.

    The weighting strategy can be mathematically represented as:

    .. math::
        w_i = \frac{\bar{s}_i}{\sum_{j=1}^{N} \bar{s}_j}

    where:
    - :math:`w_i` is the weight assigned to the \(i\)-th base estimator.
    - :math:`\bar{s}_i` is the mean cross-validation score of the \(i\)-th 
      base estimator.
    - \(N\) is the number of base estimators in the ensemble.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.estimators._base import determine_weights
    >>> import numpy as np
    >>> base_estimators = [LogisticRegression(), DecisionTreeClassifier()]
    >>> X = np.random.rand(100, 4)
    >>> y = np.random.randint(0, 2, size=100)
    >>> weights = determine_weights(base_estimators, X, y, problem='classification')
    >>> print(weights)

    See Also
    --------
    - cross_val_score : Evaluate a score by cross-validation.
    - make_scorer : Make a scorer from a performance metric or loss function.
    - WeightedAverageRegressor : Ensemble regressor using weighted average.

    """
    scores = []
    cv = cv or 5 
    if isinstance(scoring, str) and problem.lower() == 'classification':
        scoring = 'accuracy'
        
    for estimator in base_estimators:
        # Perform cross-validation and get the mean score
        score = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        scores.append(np.mean(score))
    
    # Convert scores to weights by normalizing them
    scores = np.array(scores)
    # Ensure no negative weights (e.g., converting scores if using negative error metrics)
    if np.any(scores < 0):
        scores = scores - np.min(scores)
    weights = scores / np.sum(scores)
    
    return weights

def apply_scaling(scaler, X, return_keys=False):
    """
    Apply the specified scaler to the data.

    Parameters
    ----------
    scaler : str or object
        The scaler to use for scaling the input data. Supported scalers include:
        - 'minmax' or '01': Uses `MinMaxScaler` to scale features to [0, 1].
        - 'standard' or 'zscore': Uses `StandardScaler` to scale features to have
          zero mean and unit variance.
        - 'robust': Uses `RobustScaler` to scale features using statistics that
          are robust to outliers.
        - 'sum': Normalizes features by dividing each element by the sum of its
          respective row.
        If `scaler` is already a fitted scaler object, it will be used directly.

    X : array-like of shape (n_samples, n_features)
        The input data to scale.

    return_keys : bool, default=False
        If True, returns  the scaler and the key of the scaler used along 
        with the scaled data.

    Returns
    -------
    X_scaled : array-like of shape (n_samples, n_features)
        The scaled input data.

    Notes
    -----
    This method applies the scaler specified by the `scaler` attribute to the
    input data `X`. If `scaler` is a string, it maps to a corresponding scaler
    class and applies it to `X`. If `scaler` is already a fitted scaler object,
    it is used directly to transform `X`.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from gofast.estimators._base import apply_scaling
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> X_scaled = apply_scaling('standard', X)
    >>> print(X_scaled)

    >>> scaler = StandardScaler()
    >>> X_scaled = apply_scaling(scaler, X)
    >>> print(X_scaled)

    See Also
    --------
    - sklearn.preprocessing.MinMaxScaler
    - sklearn.preprocessing.StandardScaler
    - sklearn.preprocessing.RobustScaler

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.

    """
    # Validate input array
    X = check_array(X, accept_sparse=True)

    # Define a mapping from string identifiers to scaler classes
    scaler_map = {
        'minmax': MinMaxScaler,
        '01': MinMaxScaler,
        'standard': StandardScaler,
        'zscore': StandardScaler,
        'robust': RobustScaler
    }

    # If scaler is a string, map to corresponding scaler class
    if isinstance(scaler, str):
        scaler_key = scaler.lower()
        if scaler_key in scaler_map:
            scaler = scaler_map[scaler_key]()
        elif scaler_key == 'sum':
            X = normalize_sum(X)
        else:
            raise ValueError(f"Unknown scaler type: {scaler}")

    # Ensure the scaler has a fit_transform method
    if callable(getattr(scaler, "fit_transform", None)):
        X = scaler.fit_transform(X)
    else:
        raise TypeError("Scaler must be a string or a fitted scaler object"
                        f" with fit_transform method, got {type(scaler)}")

    # Return scaled data, and optionally the scaler keys used
    return (X, scaler, scaler_key) if return_keys else X

def normalize_sum(X, axis=0):
    """
    Normalize the data by dividing each element by the sum along the specified axis.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to normalize. Can be a dense array or sparse matrix.
        
    axis : int or None, default=0
        Axis along which the sum is applied.
        - If 0, normalize along columns.
        - If 1, normalize along rows.
        - If None, normalize the entire array.

    Returns
    -------
    X_normalized : array-like of shape (n_samples, n_features)
        The sum-normalized input data.

    Notes
    -----
    This method normalizes the input data `X` by dividing each element by the 
    sum along the specified axis. This ensures that the sum of the elements along 
    the specified axis is equal to 1. This is useful for various machine learning 
    algorithms where normalized inputs are required.

    For a dense matrix, each element is divided by the sum of its respective 
    column or row directly. For a sparse matrix, the non-zero elements are divided 
    by the sum of their respective columns or rows.

    Mathematically, the normalization for each element :math:`x_{ij}` in row 
    :math:`i` and column :math:`j` of matrix :math:`X` is given by:

    .. math::
        x_{ij}^{\text{normalized}} = \frac{x_{ij}}{\sum_{k=1}^{n} x_{ik}}

    where:
    - :math:`x_{ij}^{\text{normalized}}` is the normalized value.
    - :math:`x_{ij}` is the original value.
    - :math:`n` is the number of columns (features) if `axis=1` or rows (samples) 
      if `axis=0`.
    - :math:`\sum_{k=1}^{n} x_{ik}` is the sum of all elements in row :math:`i`
      if `axis=1` or column :math:`j` if `axis=0`.

    If `axis` is None, the normalization is performed over the entire array:

    .. math::
        x_{ij}^{\text{normalized}} = \frac{x_{ij}}{\sum_{k=1}^{n} \sum_{l=1}^{m} x_{kl}}

    Examples
    --------
    >>> from gofast.estimators._base import normalize_sum
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> 
    >>> # Example with dense matrix
    >>> X_dense = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized_dense = normalize_sum(X_dense, axis=1)
    >>> print("Normalized Dense Matrix:\n", X_normalized_dense)
    >>> 
    >>> # Example with sparse matrix
    >>> X_sparse = csr_matrix([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized_sparse = normalize_sum(X_sparse, axis=1)
    >>> print("Normalized Sparse Matrix:\n", X_normalized_sparse.toarray())

    See Also
    --------
    sklearn.preprocessing.normalize : Utility function to normalize samples.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    # Validate axis parameter
    if axis not in {None, 0, 1}: 
        raise ValueError("axis must be one of None, 0, or 1")
    
    X = check_array( X, accept_large_sparse= True, accept_sparse= True, 
                    input_name="X", to_frame= False )
    if issparse(X):
        return _normalize_sparse(X, axis)
    else:
        return _normalize_dense(X, axis)

def _normalize_sparse(X, axis):
    # Sum each row or columns for sparse matrix
    if axis is None:
        total_sum = X.sum()
        normalized_data = X.data / total_sum
        return X.__class__((normalized_data, X.indices, X.indptr), shape=X.shape)
    else:
        row_col_sums = np.array(X.sum(axis=axis)).flatten()
        row_indices, col_indices = X.nonzero()
        if axis == 0:
            normalized_data = X.data / row_col_sums[col_indices]
        else:
            normalized_data = X.data / row_col_sums[row_indices]
        return X.__class__((normalized_data, (row_indices, col_indices)), shape=X.shape)

def _normalize_dense(X, axis):
    # Sum each row  or columns for dense matrix
    if axis is None:
        total_sum = X.sum()
        return X / total_sum
    else:
        row_col_sums = X.sum(axis=axis)
        if axis == 0:
            return X / row_col_sums[np.newaxis, :]
        else:
            return X / row_col_sums[:, np.newaxis]

def optimize_hyperparams(
        estimator, X, y, param_grid=None, optimizer='RSCV', cv=None):
    """
    Optimize hyperparameters for the given estimator using the specified
    optimizer and cross-validation strategy.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to optimize. The estimator should have methods 
        `fit` and `predict` to be compatible with the optimization process.

    X : array-like of shape (n_samples, n_features)
        Training data input. This represents the features of the dataset 
        used to train the model. Each row corresponds to a sample, and each 
        column corresponds to a feature.

    y : array-like of shape (n_samples,)
        Target values for each sample. These are the true values that the 
        model aims to predict. Each element corresponds to the target value 
        for a respective sample in `X`.

    optimizer : str, default='RSCV'
        The optimization method to use for hyperparameter tuning. Supported 
        options include:
        - 'RSCV' : RandomizedSearchCV for random search over hyperparameter 
          distributions.
        - 'GSCV' : GridSearchCV for exhaustive search over specified 
          hyperparameter values.
    param_grid: dict, 
       List of parameters for estimator fine-tuning. If ``None``, none 
       grid param is passed and return default tunning. 
       
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. It specifies the 
        number of folds in cross-validation or provides an iterable yielding 
        train/test splits. If None, 5-fold cross-validation is used.

    Returns
    -------
    estimator : estimator object
        The optimized estimator. If the optimizer was used, the returned 
        estimator will be the best estimator found during the optimization 
        process. Otherwise, the original estimator is returned.

    Examples
    --------
    >>> from gofast.estimators.util import optimize_hyperparams
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> estimator = RandomForestRegressor()
    >>> optimized_estimator = optimize_hyperparams(
        estimator, X, y, optimizer='GSCV')

    Notes
    -----
    Hyperparameter optimization is a critical step in the machine learning 
    pipeline to improve the performance of a model. It involves searching for 
    the best combination of hyperparameters that yield the highest performance 
    on a given dataset.

    The choice of optimizer (`optimizer` parameter) can significantly impact 
    the efficiency and effectiveness of the hyperparameter search:
    - 'RSCV' (RandomizedSearchCV) is suitable for large hyperparameter spaces 
      as it samples a fixed number of hyperparameter settings from specified 
      distributions.
    - 'GSCV' (GridSearchCV) performs an exhaustive search over a specified 
      parameter grid, which can be computationally expensive but thorough.

    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyperparameters.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified parameter values.

    References
    ----------
    .. [1] Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). 
           Algorithms for Hyper-Parameter Optimization. In Advances in Neural 
           Information Processing Systems (pp. 2546-2554).
    .. [2] Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
    """
    from ..models.utils import get_strategy_method
    
    param_grid = param_grid or {}
    # Validate and retrieve the optimizer method
    optimizer = get_strategy_method(optimizer)
    
    # Initialize the search object (RandomizedSearchCV or GridSearchCV)
    search = optimizer(estimator, param_grid, cv=cv) if optimizer else estimator

    # Fit the search object to the data
    search.fit(X, y)
    # Return the best estimator if available, otherwise return the search object
    return search.best_estimator_ if hasattr(search, 'best_estimator_') else search

def estimate_memory_depth(X, default_depth=5):
    """
    Estimates the memory depth for a HammersteinWienerRegressor when none 
    is specified.

    Args:
    X (np.array): The input dataset, assumed to be time-series data.
    default_depth (int): Default memory depth if a smart estimate cannot be determined.

    Returns:
    int: An estimated memory depth.
    """
    if X.ndim != 2:
        raise ValueError("Input data X must be a 2D array where rows are"
                         " observations and columns are features.")

    # Check the autocorrelation of the first feature as a simple example
    feature_autocorr = np.correlate(X[:, 0], X[:, 0], mode='full')
    max_lag = len(feature_autocorr) // 2  # Middle of the correlation array
    feature_autocorr = feature_autocorr[max_lag:]  # Take the second half (positive lags)

    # Find the first lag where autocorrelation drops below 0.5
    # This could be a simplistic criterion for determining the memory
    significant_lags = np.where(feature_autocorr < 0.5 * np.max(feature_autocorr))[0]
    if significant_lags.size > 0:
        estimated_depth = significant_lags[0]
        if estimated_depth > 0:
            return estimated_depth
    # Return the default depth if no significant autocorrelation drop is found
    return default_depth 

def validate_memory_depth(X, memory_depth=None, default_depth=5):
    """
    Validates the provided memory depth or estimates it if None. This function 
    ensures that the memory depth is a positive integer and does not exceed the 
    number of available samples in the dataset. If no memory depth is provided, 
    it estimates a sensible default based on the autocorrelation of the input 
    dataset or a predefined heuristic.

    Parameters
    ----------
    X : np.array
        The input dataset, assumed to be time-series data. It should be a 2D NumPy 
        array where rows represent time points and columns represent variables.
    memory_depth : int, optional
        The memory depth to validate. It must be a positive integer and cannot
        exceed the length of the dataset. The default is None, which triggers
        estimation of memory depth.
    default_depth : int or str, optional
        Specifies the default depth to use if `memory_depth` is None. If set to 
        ``"auto"``, the function will calculate the default as half the number of
        samples. The default is 5.

    Returns
    -------
    int
        The validated or estimated memory depth.

    Raises
    ------
    ValueError
        If `memory_depth` is not a positive integer, or if it exceeds the number 
        of samples, or if it is not a whole number when expected.

    Notes
    -----
    The function utilizes a heuristic based on the autocorrelation of the first 
    feature to estimate memory depth if none is provided. This estimation process 
    is crucial for dynamic modeling in systems where the appropriate memory depth 
    is not straightforward to determine.

    Examples
    --------
    >>> from gofast.estimators.util import validate_memory_depth
    >>> import numpy as np
    >>> X = np.random.rand(100, 10)  # A dataset with 100 samples and 10 features
    >>> validated_depth = validate_memory_depth(X)
    >>> print(validated_depth)

    See Also
    --------
    estimate_memory_depth : The function used for estimating memory depth based on 
                            autocorrelation if no memory depth is provided.

    References
    ----------
    .. [1] Ljung, L. (1999). System Identification - Theory for the User. Prentice Hall, 
           Upper Saddle River, N.J.
    
    """
    num_samples = X.shape[0]
    if memory_depth is not None:
        memory_depth = validate_positive_integer(memory_depth, "memory_depth",  
                                                 round_float="ceil") 
        if memory_depth > num_samples:
            raise ValueError("Memory depth cannot exceed the number"
                             f" of samples ({num_samples}).")
        return memory_depth
    else:
        if default_depth == "auto":
            default_depth = num_samples // 2  

        estimated_depth = estimate_memory_depth(X, default_depth)
        if estimated_depth > num_samples:
            raise ValueError("Estimated memory depth exceeds the number"
                             f" of samples ({num_samples}). Consider"
                             " reducing the default depth or using a"
                             " different estimation technique.")
            
        return estimated_depth

def select_default_estimator(
        estimator: str|BaseEstimator, problem: str = 'regression'):
    """
    Select a default estimator based on the given problem type.

    This function selects a default estimator based on the provided 
    estimator name and problem type. It maps common names and aliases 
    for estimators to their corresponding classes in Scikit-learn. 

    Parameters
    ----------
    estimator : str
        The name or alias of the estimator to be selected. The function 
        recognizes common names and aliases for logistic regression, 
        decision tree classifier, and linear regression.

        Supported classifiers:
        - "logisticregression", "logit", "MaxEnt", "logistic"
        - "decisiontreeclassifier", "dt", "dtc", "tree"

        Supported regressors:
        - "linearregression", "linreg", "decisiontreeregressor"

    BaseEstimator : sklearn.base.BaseEstimator
        The base estimator class from Scikit-learn. This is required to 
        check if the provided estimator has `fit` and `predict` methods.

    problem : str, optional
        The type of problem for which the estimator is needed. Supported 
        values are 'regression' and 'classification'. The default is 
        'regression'.

    Returns
    -------
    estimator : object
        The selected default estimator instance. For classification 
        problems, it returns `LogisticRegression` or `DecisionTreeClassifier`. 
        For regression problems, it returns `LinearRegression`.

    Raises
    ------
    ValueError
        If the provided `estimator` string does not match any recognized 
        aliases or if the `estimator` does not have `fit` and `predict` 
        methods.

    Examples
    --------
    >>> from gofast.estimators.util import select_default_estimator

    Select a default logistic regression estimator:
    
    >>> estimator = select_default_estimator('logit', 'classification')
    >>> print(estimator)
    LogisticRegression()

    Select a default decision tree classifier:
    
    >>> estimator = select_default_estimator('dtc',  'classification')
    >>> print(estimator)
    DecisionTreeClassifier()

    Select a default linear regression estimator:
    
    >>> estimator = select_default_estimator('linreg', 'regression')
    >>> print(estimator)
    LinearRegression()

    Notes
    -----
    The function supports common aliases for logistic regression and 
    decision tree classifiers. It ensures that the returned estimator 
    has `fit` and `predict` methods.

    See Also
    --------
    LogisticRegression : Logistic regression classifier from Scikit-learn.
    DecisionTreeClassifier : Decision tree classifier from Scikit-learn.
    LinearRegression : Linear regression model from Scikit-learn.

    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", 
           Journal of Machine Learning Research, 12, pp. 2825-2830, 2011.
    """
    # Define the mapping of regex patterns to the corresponding estimators
    estimator_mapping = {
        r"logistic(regression)?|logit|max(ent|imum[-_ ]?entropy)": LogisticRegression,
        r"decision[-_ ]?tree(classifier|regressor)?|dtc?|tree": {
            'classification': DecisionTreeClassifier,
            'regression': DecisionTreeRegressor
        },
        r"linear(regression)?|lreg|lin(reg|ear[-_ ]model)?": LinearRegression
    }

    if isinstance(estimator, str):
        estimator_lower = estimator.lower()
        problem_lower = problem.lower()
        for pattern, est in estimator_mapping.items():
            if re.match(pattern, estimator_lower, re.IGNORECASE):
                if isinstance(est, dict):
                    if problem_lower in est:
                        return est[problem_lower]()
                    else:
                        raise ValueError(
                            "Invalid problem type for decision tree. Supported"
                            " types are 'regression' and 'classification'.")
                return est()
        
        raise ValueError(
            "Invalid estimator name. Supported estimators: logistic regression,"
            " decision tree, and linear regression.")
    
    if not hasattr(estimator, 'fit') or not hasattr(estimator, 'predict'):
        raise ValueError("The provided estimator must have fit and predict methods.")
    
    return estimator

def select_best_regression_model(
    X: Union[DataFrame, np.ndarray], 
    y: Union[Series, np.ndarray]
) -> Tuple[object, str, float]:
    """
    Analyze the data and provide the best linear regression model 
    based on R^2 score.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Features data. Each row represents an observation, and each 
        column represents a feature.

    y : Union[pd.Series, np.ndarray]
        Target data. A one-dimensional array-like structure containing 
        the target variable.

    Returns
    -------
    Tuple[object, str, float]
        A tuple containing:
        
        - The best model fitted to the data.
        - The name of the best model as a string.
        - The R^2 score of the best model on the test set.

    Notes
    -----
    The function fits multiple linear regression models to the data 
    and selects the best one based on the R^2 score, which is defined 
    as:

    .. math:: R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}` is the sum of squares of residuals and 
    :math:`SS_{tot}` is the total sum of squares. The R^2 score 
    indicates the proportion of the variance in the dependent variable 
    that is predictable from the independent variables.

    The function supports the following regression models:

    - Ordinary Least Squares (LinearRegression)
    - Ridge Regression (Ridge)
    - Lasso Regression (Lasso)
    - Elastic Net Regression (ElasticNet)
    - Bayesian Ridge Regression (BayesianRidge)
    - Stochastic Gradient Descent Regressor (SGDRegressor)
    - Huber Regressor (HuberRegressor)
    - RANSAC Regressor (RANSACRegressor)
    - Theil-Sen Regressor (TheilSenRegressor)
    
    User should:
    - Ensure that the input data `X` and `y` are in compatible shapes 
      and formats.
    - The function standardizes the features before fitting the models.
    - The R^2 score is used as the primary metric for model evaluation 
      and selection.

    Examples
    --------
    >>> from gofast.estimators.util import select_best_regression_model
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
    >>> best_model, best_model_name, best_r2_score = select_best_regression_model(X, y)
    >>> print(f"Best Model: {best_model_name}")
    >>> print(f"Best R^2 Score: {best_r2_score}")

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary Least Squares 
                                            Linear Regression.
    sklearn.linear_model.Ridge : Ridge regression with L2 
                                 regularization.
    sklearn.linear_model.Lasso : Lasso regression with L1 
                                 regularization.
    sklearn.linear_model.ElasticNet : ElasticNet regression combining 
                                      L1 and L2 regularization.
    sklearn.linear_model.BayesianRidge : Bayesian Ridge regression.
    sklearn.linear_model.SGDRegressor : Stochastic Gradient Descent 
                                        regressor.
    sklearn.linear_model.HuberRegressor : Huber regressor for robust 
                                          linear regression.
    sklearn.linear_model.RANSACRegressor : RANSAC regressor for robust 
                                           linear regression.
    sklearn.linear_model.TheilSenRegressor : Theil-Sen estimator for 
                                             robust linear regression.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., 
           Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., 
           Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., 
           Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: 
           Machine Learning in Python. Journal of Machine Learning 
           Research, 12, 2825-2830.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
        SGDRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor
    )
    from sklearn.metrics import r2_score

    def standardize_data(X_train: np.ndarray, X_test: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize the training and test data."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def initialize_models() -> Dict[str, object]:
        """Initialize a dictionary of regression models."""
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.7),
            "BayesianRidge": BayesianRidge(),
            "SGDRegressor": SGDRegressor(),
            "HuberRegressor": HuberRegressor(),
            "RANSACRegressor": RANSACRegressor(),
            "TheilSenRegressor": TheilSenRegressor()
        }

    def evaluate_models(
        models: Dict[str, object], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[object, str, float]:
        """Fit and evaluate models, returning the best 
        one based on R^2 score."""
        best_model = None
        best_model_name = ""
        best_r2_score = -np.inf

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2_score:
                best_r2_score = r2
                best_model = model
                best_model_name = model_name

        return best_model, best_model_name, best_r2_score

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Standardize the data
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Initialize models
    models = initialize_models()
    # Evaluate models and select the best one
    best_model, best_model_name, best_r2_score = evaluate_models(
        models, X_train_scaled, y_train, X_test_scaled, y_test)

    return best_model, best_model_name, best_r2_score

def select_best_classification_model(
    X: Union[DataFrame, np.ndarray], 
    y: Union[Series, np.ndarray]
) -> Tuple[object, str, float]:
    """
    Analyze the data and provide the best classification model based 
    on accuracy score.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Features data. Each row represents an observation, and each 
        column represents a feature.

    y : Union[pd.Series, np.ndarray]
        Target data. A one-dimensional array-like structure containing 
        the target variable.

    Returns
    -------
    Tuple[object, str, float]
        A tuple containing:
        
        - The best classification model fitted to the data.
        - The name of the best model as a string.
        - The accuracy score of the best model on the test set.

    Notes
    -----
    The function fits multiple classification models to the data and 
    selects the best one based on the accuracy score, which is defined 
    as:

    .. math:: Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}

    where :math:`TP` is the number of true positives, :math:`TN` is the 
    number of true negatives, :math:`FP` is the number of false 
    positives, and :math:`FN` is the number of false negatives. The 
    accuracy score indicates the proportion of correctly classified 
    instances.

    The function supports the following classification models:

    - Logistic Regression (LogisticRegression)
    - Support Vector Classifier (SVC)
    - K-Nearest Neighbors (KNeighborsClassifier)
    - Decision Tree Classifier (DecisionTreeClassifier)
    - Random Forest Classifier (RandomForestClassifier)
    - Gradient Boosting Classifier (GradientBoostingClassifier)
    - Gaussian Naive Bayes (GaussianNB)
    - Multinomial Naive Bayes (MultinomialNB)
    - Bernoulli Naive Bayes (BernoulliNB)
    
    User should: 
    - Ensure that the input data `X` and `y` are in compatible shapes 
      and formats.
    - The function standardizes the features before fitting the models 
      that are not sensitive to negative values.
    - The accuracy score is used as the primary metric for model 
      evaluation and selection.

    Examples
    --------
    >>> from gofast.estimators.util import select_best_classification_model
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=20, 
                                   n_classes=2, random_state=42)
    >>> best_model, best_model_name, best_accuracy_score = select_best_classification_model(X, y)
    >>> print(f"Best Model: {best_model_name}")
    >>> print(f"Best Accuracy Score: {best_accuracy_score}")

    See Also
    --------
    sklearn.linear_model.LogisticRegression : Logistic Regression classifier.
    sklearn.svm.SVC : Support Vector Classifier.
    sklearn.neighbors.KNeighborsClassifier : K-Nearest Neighbors classifier.
    sklearn.tree.DecisionTreeClassifier : Decision Tree classifier.
    sklearn.ensemble.RandomForestClassifier : Random Forest classifier.
    sklearn.ensemble.GradientBoostingClassifier : Gradient Boosting classifier.
    sklearn.naive_bayes.GaussianNB : Gaussian Naive Bayes classifier.
    sklearn.naive_bayes.MultinomialNB : Multinomial Naive Bayes classifier.
    sklearn.naive_bayes.BernoulliNB : Bernoulli Naive Bayes classifier.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., 
           Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., 
           Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., 
           Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. 
           (2011). Scikit-learn: Machine Learning in Python. Journal of 
           Machine Learning Research, 12, 2825-2830.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.metrics import accuracy_score

    def standardize_data(
            X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Standardize the training and test data."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def normalize_data(
            X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the training and test data to range [0, 1]."""
        scaler = MinMaxScaler()
        X_train_normalized = scaler.fit_transform(X_train)
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, X_test_normalized

    def initialize_models() -> Dict[str, object]:
        """Initialize a dictionary of classification models."""
        return {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "SVC": SVC(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "GaussianNB": GaussianNB(),
            "MultinomialNB": MultinomialNB(),
            "BernoulliNB": BernoulliNB()
        }

    def evaluate_models(
        models: Dict[str, object], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Tuple[object, str, float]:
        """Fit and evaluate models, returning the best one based on accuracy score."""
        best_model = None
        best_model_name = ""
        best_accuracy_score = -np.inf

        for model_name, model in models.items():
            # Use StandardScaler for models that are not sensitive to negative values
            if model_name in ["LogisticRegression", "SVC", "KNeighborsClassifier",
                              "GradientBoostingClassifier"]:
                X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)
            else:
                X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > best_accuracy_score:
                best_accuracy_score = accuracy
                best_model = model
                best_model_name = model_name

        return best_model, best_model_name, best_accuracy_score

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = initialize_models()

    # Evaluate models and select the best one
    best_model, best_model_name, best_accuracy_score = evaluate_models(
        models, X_train, y_train, X_test, y_test
    )

    return best_model, best_model_name, best_accuracy_score

def select_best_model(
    X: Optional[Union[DataFrame, np.ndarray]] = None, 
    y: Optional[Union[Series, np.ndarray]] = None, 
    estimator: Optional[str] = None, 
    problem: str = "regression"
) -> object:
    """
    Select the best model for a given dataset and problem type.

    Parameters
    ----------
    X : Optional[Union[pd.DataFrame, np.ndarray]], optional
        Features data. Each row represents an observation, and each 
        column represents a feature. If `X` is not provided, the 
        function returns the default estimator.

    y : Optional[Union[pd.Series, np.ndarray]], optional
        Target data. A one-dimensional array-like structure containing 
        the target variable. If `y` is not provided, the function 
        returns the default estimator.

    estimator : Optional[str], optional
        The estimator type to use. If not provided, defaults to "dt" 
        (Decision Tree) for classification and "linreg" (Linear 
        Regression) for regression.

    problem : str, default="regression"
        The type of problem, either "regression" or "classification". 
        This parameter determines the default estimator type and the 
        model selection process.

    Returns
    -------
    object
        The best estimator model fitted to the data if provided, 
        otherwise the default estimator.

    Notes
    -----
    - Ensure that the input data `X` and `y` are in compatible shapes 
      and formats.
    - The function standardizes the features before fitting the models 
      that are not sensitive to negative values.
    - The accuracy score is used as the primary metric for model 
      evaluation and selection for classification problems.
    - The R^2 score is used as the primary metric for model evaluation 
      and selection for regression problems.
      
    The function selects the best model based on the problem type:
    
    - For regression problems, the function optimizes the R^2 score, 
      defined as:
      
      .. math:: R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}
      
      where :math:`SS_{res}` is the sum of squares of residuals and 
      :math:`SS_{tot}` is the total sum of squares. The R^2 score 
      indicates the proportion of the variance in the dependent variable 
      that is predictable from the independent variables.
    
    - For classification problems, the function optimizes the accuracy 
      score, defined as:
      
      .. math:: Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}
      
      where :math:`TP` is the number of true positives, :math:`TN` is 
      the number of true negatives, :math:`FP` is the number of false 
      positives, and :math:`FN` is the number of false negatives. The 
      accuracy score indicates the proportion of correctly classified 
      instances.

    Examples
    --------
    >>> from gofast.estimators.util import select_best_model
    >>> from sklearn.datasets import make_classification, make_regression

    >>> # Example for classification
    >>> X_class, y_class = make_classification(n_samples=1000, n_features=20, 
    ...                                        random_state=42)
    >>> best_class_model = select_best_model(
    ...        X_class, y_class, problem="classification")
    >>> print(f"Best Classification Model: {best_class_model}")

    >>> # Example for regression
    >>> X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1,
    ...                               random_state=42)
    >>> best_reg_model = select_best_model(X_reg, y_reg, problem="regression")
    >>> print(f"Best Regression Model: {best_reg_model}")

    See Also
    --------
    sklearn.linear_model.LogisticRegression : Logistic Regression classifier.
    sklearn.svm.SVC : Support Vector Classifier.
    sklearn.neighbors.KNeighborsClassifier : K-Nearest Neighbors classifier.
    sklearn.tree.DecisionTreeClassifier : Decision Tree classifier.
    sklearn.ensemble.RandomForestClassifier : Random Forest classifier.
    sklearn.ensemble.GradientBoostingClassifier : Gradient Boosting classifier.
    sklearn.naive_bayes.GaussianNB : Gaussian Naive Bayes classifier.
    sklearn.naive_bayes.MultinomialNB : Multinomial Naive Bayes classifier.
    sklearn.naive_bayes.BernoulliNB : Bernoulli Naive Bayes classifier.
    sklearn.linear_model.LinearRegression : 
        Ordinary Least Squares Linear Regression.
    sklearn.linear_model.Ridge : Ridge regression with L2 regularization.
    sklearn.linear_model.Lasso : Lasso regression with L1 regularization.
    sklearn.linear_model.ElasticNet : 
        ElasticNet regression combining L1 and L2 regularization.
    sklearn.linear_model.BayesianRidge :
        Bayesian Ridge regression.
    sklearn.linear_model.SGDRegressor : 
        Stochastic Gradient Descent regressor.
    sklearn.linear_model.HuberRegressor :
        Huber regressor for robust linear regression.
    sklearn.linear_model.RANSACRegressor : 
        RANSAC regressor for robust linear regression.
    sklearn.linear_model.TheilSenRegressor : 
        Theil-Sen estimator for robust linear regression.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., 
           Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., 
           Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., 
           Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. 
           (2011). Scikit-learn: Machine Learning in Python. Journal of 
           Machine Learning Research, 12, 2825-2830.
    """
    # Determine the default estimator based on the problem type
    if estimator is None:
        estimator = "dt" if problem == "classification" else "linreg"
    elif estimator is not None: 
        return select_default_estimator(estimator, problem=problem)
    # Select the default estimator
    best_estimator = select_default_estimator(estimator, problem=problem)

    # If X or y is not provided, return the default estimator
    if X is None or y is None:
        return best_estimator
    # Attempt to find the best model using provided data
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if problem == 'regression':
                best_estimator, *_ = select_best_regression_model(X, y)
            elif problem == 'classification':
                best_estimator, *_ = select_best_classification_model(X, y)
                
    except Exception as e:
        # Log or handle the exception as needed
        print(f"An error occurred: {e}")

    return best_estimator

def validate_estimators(estimators):
    # Ensure each estimator has a fit method and is a scikit-learn estimator
    for name, estimator in estimators:
        if not hasattr(estimator, "fit"):
            raise ValueError(f"Estimator {name} does not implement a 'fit' method.")
        if not isinstance(estimator, BaseEstimator):
            raise ValueError(f"Estimator {name} does not follow the scikit-learn"
                             " API estimator structure.")

def build_named_estimators(
        estimators, perform_deep_check=False, error='raise'):
    """
    Build a list of named estimators.

    Parameters
    ----------
    estimators : list of estimators or list of (str, estimator) tuples
        List of estimators which can either be a list of scikit-learn 
        estimator instances or a list of (name, estimator) tuples.

    perform_deep_check : bool, optional, default=False
        If True, performs a deep check to validate that each estimator 
        implements a `fit` method and is a scikit-learn estimator.

    error : {'raise', 'warn'}, optional, default='raise'
        Specifies how to handle errors if the input format is incorrect.
        - 'raise': Raise a ValueError with an appropriate error message.
        - 'warn': Issue a warning with an appropriate error message.

    Returns
    -------
    list of (str, estimator) tuples
        List of tuples where each tuple contains a string (the name of 
        the estimator) and the estimator instance.

    Examples
    --------
    >>> from gofast.estimators.util import build_named_estimators
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> estimators = [RandomForestClassifier(), SVC()]
    >>> named_estimators = build_named_estimators(estimators)
    >>> print(named_estimators)
    [('RandomForestClassifier', RandomForestClassifier()), ('SVC', SVC())]

    Notes
    -----
    The function converts a list of estimators into a list of named 
    tuples. Each named tuple contains the class name of the estimator 
    and the estimator instance. This is particularly useful for 
    ensemble methods that require a list of named estimators.

    For a single estimator, it will be converted to a list with one 
    tuple. For a single tuple, it will be converted to a list with one 
    tuple.

    If `perform_deep_check` is set to True, the function will validate 
    that each estimator implements a `fit` method and is a scikit-learn 
    estimator.

    The function uses the `error_handling` parameter to determine 
    whether to raise an error or issue a warning if the input format is 
    incorrect.

    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", 
           Journal of Machine Learning Research, 12, pp. 2825-2830, 2011.
    """

    def convert_to_named_estimators(estimators):
        transformed = []
        for estimator in estimators:
            name = estimator.__class__.__name__
            transformed.append((name, estimator))
        return transformed

    # Convert single estimator to list
    if isinstance(estimators, BaseEstimator):
        estimators = [(estimators.__class__.__name__, estimators)]

    # Convert single tuple to list of tuples
    if isinstance(estimators, tuple) and len(estimators) == 2:
        estimators = [estimators]

    # Check if estimators are in the correct format
    if all(isinstance(item, tuple) and len(item) == 2 for item in estimators):
        if perform_deep_check:
            validate_estimators(estimators)
        return estimators

    # Convert list of estimators to list of named tuples
    if all(isinstance(item, BaseEstimator) for item in estimators):
        named_estimators = convert_to_named_estimators(estimators)
        if perform_deep_check:
            validate_estimators(named_estimators)
        return named_estimators

    # Handle incorrect format
    error_message = ("Estimators list format is incorrect. Expected structure:"
                     " [(name, estimator), ...] or [estimator, ...].")
    
    if error == 'raise':
        raise ValueError(error_message)
    elif error == 'warn':
        warnings.warn(error_message)
    return []
