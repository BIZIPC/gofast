# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import inspect 
from collections import defaultdict 
import numpy as np 
from tqdm import tqdm

from sklearn.metrics import r2_score , accuracy_score
from ..api.property import BaseClass 
from ..tools.validator import check_X_y, get_estimator_name
from ..tools.validator import check_is_fitted, validate_fit_weights 


__all__=["StandardEstimator", "DecisionStumpRegressor", "DecisionStumpClassifier"]

class StandardEstimator:
    """Base class for all classes in gofast for parameters retrievals

    Notes
    -----
    All class defined should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "gofast classes should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this class and
            contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple classes as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

class DecisionStumpRegressor(BaseClass, StandardEstimator):
    r"""
    A simple decision stump regressor for use in gradient boosting.

    This class implements a basic decision stump, which is a decision tree 
    with only one decision node and two leaves. The stump splits the data 
    based on the feature and threshold that minimizes the error.
    
    For each feature, the stump examines all possible thresholds
    (unique values of the feature). The MSE for a split at a given 
    threshold t is calculated as follows:

    .. math::
        MSE = \sum_{i \in \text{left}(t)} (y_i - \overline{y}_{\text{left}})^2 +
              \sum_{i \in \text{right}(t)} (y_i - \overline{y}_{\text{right}})^2

    where :math:`\text{left}(t)` and :math:`\text{right}(t)` are the sets of indices 
    of samples that fall to the left and right of the threshold t, respectively. 
    :math:`\overline{y}_{\text{left}}` and :math:`\overline{y}_{\text{right}}` 
    are the mean values of the target variable for the samples in each of these 
    two sets.

    The algorithm selects the feature and threshold that yield the lowest MSE.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to consider a split at a node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node. A split point 
        at any depth will only be considered if it leaves at least this many training 
        samples in each of the left and right branches.
        
    verbose : int, default=False
        Controls the verbosity when fitting and predicting.
        
    Attributes
    ----------
    split_feature_ : int
        Index of the feature used for the split.

    split_value_ : float
        Threshold value used for the split.

    left_value_ : float
        The value predicted for samples where the feature value is less than or 
        equal to the split value.

    right_value_ : float
        The value predicted for samples where the feature value is greater than 
        the split value.

    feature_importances_ : ndarray
        The feature importances (if calculated), reflecting the reduction in MSE 
        contributed by each feature.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the decision stump to the data.

    predict(X)
        Predict target values for the given input data using the trained 
        decision stump.

    decision_function(X)
        Compute the raw decision scores for the given input data.

    score(X, y, sample_weight=None)
        Return the coefficient of determination R^2 of the prediction.
       
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from gofast.estimators.tree import DecisionStumpRegressor
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> stump = DecisionStumpRegressor()
    >>> stump.fit(X, y)
    >>> predictions = stump.predict(X)
    """

    def __init__(self, min_samples_split=2, min_samples_leaf=1, verbose=False):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose=verbose
        
        self.split_feature_ = None
        self.split_value_ = None
        self.left_value_ = None
        self.right_value_ = None
        self.feature_importances_ = None
        

    def fit(self, X, y, sample_weight=None):
        """
        Fits the decision stump to the data.

        The method iterates over all features and their unique values to 
        find the split 
        that minimizes the mean squared error.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            The target values.

        Return 
        -------
        self: object 
           Return self.
        Notes
        -----
        The decision stump is a weak learner and is primarily used in
        ensemble methods like AdaBoost and Gradient Boosting.
        """
        X, y = check_X_y(X, y )
        sample_weight = validate_fit_weights(np.ones(X.shape[0]), sample_weight)
        self.feature_importances_ = np.zeros(X.shape[1])
        min_error = float('inf')
        n_samples, n_features = X.shape
        
        if self.verbose:
            progress_bar = tqdm(
                range(n_features), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_sorted, y_sorted, weights_sorted = ( 
                X[sorted_idx, feature], y[sorted_idx], sample_weight[sorted_idx]
                )
            for i in range(self.min_samples_leaf, n_samples - self.min_samples_leaf):
                if i < self.min_samples_split or X_sorted[i] == X_sorted[i - 1]:
                    continue
                left_mask = sorted_idx[:i]
                right_mask = sorted_idx[i:]
                if len(left_mask) < self.min_samples_leaf or len(
                        right_mask) < self.min_samples_leaf:
                    continue

                left_value = np.average(y_sorted[:i], weights=weights_sorted[:i])
                right_value = np.average(y_sorted[i:], weights=weights_sorted[i:])
                error = (np.sum(weights_sorted[:i] * ((y_sorted[:i] - left_value) ** 2)) +
                         np.sum(weights_sorted[i:] * ((y_sorted[i:] - right_value) ** 2)))

                if error < min_error:
                    min_error = error
                    self.split_feature_ = feature
                    self.split_value_ = X_sorted[i - 1]
                    self.left_value_ = left_value
                    self.right_value_ = right_value
                    self.feature_importances_[feature] += min_error - error
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
                
        self.fitted_ = True

        return self 
       
    def predict(self, X):
        """
        Predict target values for the given input data using the trained 
        decision stump.

        The prediction is based on the split learned during the fitting 
        process. Each sample in X is assigned a value based on which side 
        of the split it falls on.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples for which predictions are to be made.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values for each sample in X.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> predictions = stump.predict(X)
        >>> print(predictions[:5])

        Notes
        -----
        The `predict` method should be called only after the `fit` method has 
        been called. It uses the `split_feature`, `split_value`, `left_value`,
        and `right_value` attributes set by the `fit` method to make predictions.
        """
        check_is_fitted(self, "fitted_")
        left_mask = X[:, self.split_feature_] <= self.split_value_
        y_pred = np.where(left_mask, self.left_value_, self.right_value_)
        return y_pred
        
    def decision_function(self, X):
        """
        Compute the raw decision scores for the given input data.

        The decision function calculates the continuous value for each sample in X
        based on the trained decision stump. It reflects the model's degree of 
        certainty about the classification.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples for which decision scores are to be computed.

        Returns
        -------
        decision_scores : ndarray of shape (n_samples,)
            The raw decision scores for each sample in X.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> decision_scores = stump.decision_function(X)
        >>> print(decision_scores[:5])

        Notes
        -----
        This method should be called after the model has been fitted. It uses the
        attributes set during the `fit` method (i.e., `split_feature`, `split_value`,
        `left_value`, `right_value`) to compute the decision scores.
        """
        check_is_fitted(self, "fitted_") 
        # # Calculate the decision scores based on the split
        decision_scores = np.where(
            X[:, self.split_feature_] <= self.split_value_,
            self.left_value_, self.right_value_
            )
        
        return decision_scores

    def score(self, X, y, sample_weight=None):
        """
        Calculate the coefficient of determination :math:`R^2` of the 
        prediction.
    
        The :math:`R^2` score is a statistical measure of how well the predictions 
        approximate the true data points. An :math:`R^2` of 1 indicates that 
        the model perfectly predicts the target values, whereas an :math:`R^2` 
        of 0 indicates that the model predicts as well as a model that always 
        predicts the mean of the target  values, regardless of the input 
        features.
    
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples. For each feature in the dataset, the method 
            predicts a response based on the decision stump model.
        
        y : ndarray of shape (n_samples,)
            True values for the target variable. These are the values that 
            the model attempts to predict.
    
        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If provided, the :math:`R^2` score 
            will be calculated with these weights, emphasizing the importance of 
            certain samples more than others.
    
        Returns
        -------
        score : float
            The :math:`R^2` score that indicates the proportion of variance in the 
            dependent variable that is predictable from the independent variables. 
            Values range from -∞ (a model that performs infinitely worse than 
            the mean model) to 1. A model that always predicts the exact true 
            value would achieve a score of 1.
    
        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> score = stump.score(X, y)
        >>> print(f"R^2 score: {score:.2f}")
        """
        check_is_fitted(self, "fitted_") 
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

class DecisionStumpClassifier(BaseClass, StandardEstimator):
    r"""
    A simple decision stump classifier that uses a single-level decision tree for
    binary classification. This classifier identifies the best feature and
    threshold to split the data into two groups, aiming to minimize impurity in 
    each node.

    The decision stump makes a binary decision: it assigns every sample in one
    subset to one class and all samples in the other subset to another class. The
    choice of subset is based on a threshold applied to one feature.

    .. math::
        I(node) = 1 - \max(p, 1-p)

    Where \( I(node) \) is the impurity of the node, and \( p \) is the proportion
    of the samples in the node that belong to the most frequent class.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to consider a split at a node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node. This parameter
        ensures that each leaf has at least `min_samples_leaf` samples, which
        helps prevent the tree from overfitting.
        
    verbose : int, default=False
        Controls the verbosity when fitting.

    Attributes
    ----------
    split_feature_ : int
        The index of the feature used for the best split.

    split_value_ : float
        The threshold value used for the split at `split_feature_`.

    left_class_ : int
        The class label assigned to samples where the feature value is less than
        or equal to the threshold.

    right_class_ : int
        The class label assigned to samples where the feature value is greater
        than the threshold.

    See Also
    --------
    DecisionTreeClassifier : A classifier that uses multiple levels of decision
                             nodes, providing more complex classification boundaries.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.estimators.tree import DecisionStumpClassifier
    >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
    >>> stump = DecisionStumpClassifier(min_samples_split=4, min_samples_leaf=2)
    >>> stump.fit(X, y)
    >>> print(stump.predict([[0, 0], [1, 1]]))
    [0 1]

    Notes
    -----
    This implementation is simplified and intended for binary classification only.
    It does not handle multi-class classification and does not support more advanced
    tree-building methods that consider information gain or Gini impurity.
    """
    def __init__(self, min_samples_split=2, min_samples_leaf=1, verbose=False):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose=verbose
        self.split_feature_ = None
        self.split_value_ = None
        self.left_class_ = None
        self.right_class_ = None
        
        
    def fit(self, X, y, sample_weight=None):
        r"""
        Fit the decision stump model to the binary classification data.
    
        The fitting process involves finding the feature and threshold that 
        minimize the impurity of the resulting binary split. This is achieved by 
        calculating the error as the negative of the weighted sum of correct 
        classifications in both left and right groups formed by the threshold. 
        The goal is to maximize the number of correct predictions by selecting 
        the optimal split.
    
        .. math::
            error = - \left( \sum_{i \in \text{{left}}}w_i(y_i = \text{{left\_class}}) + 
            \sum_{j \in \text{{right}}}w_j(y_j = \text{{right\_class}}) \right)
    
        Where `left_class` and `right_class` are the classes predicted for the 
        left and right groups, respectively, \( y_i \) is the actual class of the 
        ith sample, and \( w_i \) is the sample weight of the ith sample.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (class labels in classification).
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples.
    
        Returns
        -------
        self : object
            Returns self with the fitted model. The following attributes are set after 
            fitting:
                - `split_feature_` : Index of the feature used for the best split.
                - `split_value_` : Threshold value at the best split.
                - `left_class_` : Class label for samples on the left of the threshold.
                - `right_class_` : Class and data handling, this method modifies internal
                                   state and prepares the model for prediction.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> stump.predict([[0.1, -0.1], [0.3, 0.3]])
        array([0, 1])
    
        Notes
        -----
        The method checks the input data using `check_X_y` from sklearn, which verifies
        that the data is correctly formatted and that the number of samples and labels
        matches. It also requires the `get_estimator_name` function to properly identify
        the estimator in error messages and validations.
        """
        X, y = check_X_y(X, y, estimator=self)
        min_error = float('inf')
        n_samples, n_features = X.shape
    
        sample_weight = validate_fit_weights(y, sample_weight ) 
        if self.verbose:
            progress_bar = tqdm(range(n_features), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}' )
            
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
    
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(
                        right_mask) < self.min_samples_leaf:
                    continue
    
                left_class = np.bincount(y[left_mask], weights=sample_weight[left_mask]).argmax()
                right_class = np.bincount(y[right_mask], weights=sample_weight[right_mask]).argmax()
    
                error = - (
                    np.sum(sample_weight[left_mask] * (y[left_mask] == left_class)) + 
                    np.sum(sample_weight[right_mask] * (y[right_mask] == right_class))
                )
    
                if error < min_error:
                    min_error = error
                    self.split_feature_ = feature
                    self.split_value_ = threshold
                    self.left_class_ = left_class
                    self.right_class_ = right_class
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
            
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
    
        The predictions are based on the simple rule using the `split_feature_` and 
        `split_value_` determined during the fitting process. Depending on whether 
        the feature value for a given sample is less than or equal to `split_value_`, 
        the prediction is either `left_class_` or `right_class_`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> print(stump.predict([[0.1, -0.1], [0.3, 0.3]]))
        [0 1]
        """
        check_is_fitted(self, "fitted_") 
        left_mask = X[:, self.split_feature_] <= self.split_value_
        y_pred = np.where(left_mask, self.left_class_, self.right_class_)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
    
        The probabilities are computed as the frequencies of the `left_class_` and 
        `right_class_` in the training data. For each sample in X, this method 
        returns the probability of the sample belonging to the `left_class_` or 
        `right_cap` class based on the position relative to the `split_value_`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            The probability of the sample belonging to each class.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> print(stump.predict_proba([[0.1, -0.1], [0.3, 0.3]]))
        [[0.7 0.3]
         [0.2 0.8]]
        """
        check_is_fitted(self, "fitted_") 
        left_mask = X[:, self.split_feature_] <= self.split_value_
        proba_left = np.mean(self.left_class_ == 1)
        proba_right = np.mean(self.right_class_ == 1)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 0] = np.where(left_mask, 1 - proba_left, 1 - proba_right)
        proba[:, 1] = np.where(left_mask, proba_left, proba_right)
        return proba
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
    
        This method computes the accuracy, which is the fraction of correctly predicted
        labels to the total number of observations. It uses the `predict` method to obtain 
        the class labels for the input samples and compares them with the true labels.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
    
        y : array-like of shape (n_samples,)
            True labels for X.
    
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y. The score is a float in the range [0, 1]
            where 1 indicates perfect accuracy.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> accuracy = stump.score(X, y)
        >>> print(f"Accuracy: {accuracy:.2f}")
        
        Notes
        -----
        This method checks if the estimator is fitted by using `check_is_fitted` before
        proceeding with predictions and scoring. If the model is not fitted, it will raise
        a `NotFittedError`. The inputs are validated using `check_X_y` from sklearn, which
        ensures that the dimensions and types of X and y are compatible and appropriate for
        the estimator.
        """
        check_is_fitted(self, "fitted_") 
        X, y = check_X_y(X, y, estimator=get_estimator_name(self))
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class _GradientBoostingClassifier:
    r"""
    A simple gradient boosting classifier for binary classification.

    Gradient boosting is a machine learning technique for regression and 
    classification problems, which produces a prediction model in the form 
    of an ensemble of weak prediction models, typically decision trees. It 
    builds the model in a stage-wise fashion like other boosting methods do, 
    and it generalizes them by allowing optimization of an arbitrary 
    differentiable loss function.

    Attributes
    ----------
    n_estimators : int
        The number of boosting stages to be run.
    learning_rate : float
        Learning rate shrinks the contribution of each tree.
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Build the gradient boosting model from the training set (X, y).
    predict(X)
        Predict class labels for samples in X.
    predict_proba(X)
        Predict class probabilities for X.

    Mathematical Formula
    --------------------
    The model is built in a stage-wise fashion as follows:
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.predict_proba(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical
    Learning," Springer, 2009.

    Applications
    ------------
    Gradient Boosting can be used for both regression and classification problems. 
    It's particularly effective in scenarios where the relationship between 
    the input features and target variable is complex and non-linear. It's 
    widely used in applications like risk modeling, classification of objects,
    and ranking problems.
    """

    def __init__(self, n_estimators=100, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        X, y = check_X_y(X, y, estimator= self )
        # Convert labels to 0 and 1
        y = np.where(y == np.unique(y)[0], -1, 1)

        F_m = np.zeros(len(y))

        for m in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = -1 * y * self._sigmoid(-y * F_m)

            # Fit a decision stump to the pseudo-residuals
            stump = DecisionStumpRegressor()
            stump.fit(X, residuals)

            # Update the model
            F_m += self.learning_rate * stump.predict(X)
            self.estimators_.append(stump)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities are the model's confidence
        scores for the positive class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities for the input samples. The columns 
            correspond to the negative and positive classes, respectively.
            
        Examples 
        ----------
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        print(model.predict_proba(X)[:5])
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        proba_positive_class = self._sigmoid(F_m)
        return np.vstack((1 - proba_positive_class, proba_positive_class)).T

    def _sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        return np.where(self._sigmoid(F_m) > 0.5, 1, 0)
    
class _GradientBoostingRegressor:
    r"""
    A simple gradient boosting regressor for regression tasks.

    Gradient Boosting builds an additive model in a forward stage-wise fashion. 
    At each stage, regression trees are fit on the negative gradient of the loss function.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of boosting stages to be run. This is essentially the 
        number of decision trees in the ensemble.

    eta0 : float, optional (default=1.0)
        Learning rate shrinks the contribution of each tree. There is a 
        trade-off between eta0 and n_estimators.
        
    max_depth : int, default=1
        The maximum depth of the individual regression estimators.
        
    Attributes
    ----------
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Fit the gradient boosting model to the training data.
    predict(X)
        Predict continuous target values for samples in X.
    decision_function(X)
        Compute the raw decision scores for the input data.

    Mathematical Formula
    --------------------
    Given a differentiable loss function L(y, F(x)), the model is 
    constructed as follows:
    
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> model = GradientBoostingRegressor(n_estimators=100, eta0=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.decision_function(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.

    See Also
    --------
    DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor

    Applications
    ------------
    Gradient Boosting Regressor is commonly used in various regression tasks where the relationship 
    between features and target variable is complex and non-linear. It is particularly effective 
    in predictive modeling and risk assessment applications.
    """

    def __init__(self, n_estimators=100, eta0=1.0, max_depth=1):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth=max_depth

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor to the training data.

        The method sequentially adds decision stumps to the ensemble, each one 
        correcting its predecessor. The fitting process involves finding the best 
        stump at each stage that reduces the overall prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for training. Each row in X is a sample and each 
            column is a feature.

        y : array-like of shape (n_samples,)
            The target values (continuous). The regression targets are continuous 
            values which the model will attempt to predict.

        Raises
        ------
        ValueError
            If input arrays X and y have incompatible shapes.

        Notes
        -----
        - The fit process involves computing pseudo-residuals which are the gradients 
          of the loss function with respect to the model's predictions. These are used 
          as targets for the subsequent weak learner.
        - The model complexity increases with each stage, controlled by the learning rate.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> reg = GradientBoostingRegressor(n_estimators=50, eta0=0.1)
        >>> reg.fit(X, y)
        """

        X, y = check_X_y(X, y, estimator= self )
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between X and y")

        # Initialize the prediction to zero
        F_m = np.zeros(y.shape)

        for m in range(self.n_estimators):
            # Compute residuals
            residuals = y - F_m

            # # Fit a regression tree to the negative gradient
            tree = DecisionStumpRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the model predictions
            F_m += self.eta0 * tree.predict(X)

            # Store the fitted estimator
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict continuous target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        F_m = sum(self.eta0 * estimator.predict(X)
                  for estimator in self.estimators_)
        return F_m

    def decision_function(self, X):
        """
        Compute the raw decision scores for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision_scores : array-like of shape (n_samples,)
            The raw decision scores for each sample.
        """
        return sum(self.eta0 * estimator.predict(X)
                   for estimator in self.estimators_)