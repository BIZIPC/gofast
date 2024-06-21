# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from tqdm import tqdm
from scipy.sparse import issparse
from sklearn.base import BaseEstimator

from ..tools.validator import check_array, check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
#from ..tools._param_validation import Interval, StrOptions
from .util  import validate_fit_weights, validate_positive_integer

class BaseWeightedTree(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Weighted Tree models.
    
    The `BaseWeightedTree` class serves as the foundation for building
    ensemble models that combine multiple decision trees using gradient
    boosting techniques. This class handles the common functionality for
    both classification and regression tasks.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble. Increasing the number
        of estimators generally improves the performance but also increases
        the training time.
    
    eta0 : float, default=0.1
        The learning rate for gradient boosting, controlling how much each
        tree influences the overall prediction. Smaller values require more
        trees to achieve the same performance.
    
    max_depth : int, default=3
        The maximum depth of each decision tree, determining the complexity
        of the model. Deeper trees can model more complex patterns but may
        also lead to overfitting.
    
    criterion : str, default="gini"
        The function to measure the quality of a split. Supported criteria
        are "gini" for the Gini impurity and "entropy" for the information
        gain (for classification), and "squared_error" for mean squared error,
        "friedman_mse" for mean squared error with improvement score by Friedman,
        "absolute_error" for mean absolute error, and "poisson" for Poisson
        deviance (for regression).
    
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to
        choose the best random split.
    
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and 
          `ceil(min_samples_split * n_samples)` are the minimum number of 
          samples for each split.
    
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` are the minimum number of 
          samples for each node.
    
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all 
        the input samples) required to be at a leaf node. Samples have 
        equal weight when sample_weight is not provided.
    
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and 
          `int(max_features * n_features)` features are considered at each 
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always 
        randomly permuted at each split. When `max_features` < n_features, 
        the algorithm will select `max_features` at random at each split 
        before finding the best split among them. 
        - Pass an int for reproducible output across multiple function calls.
    
    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes 
        are defined as relative reduction in impurity. If None, then 
        unlimited number of leaf nodes.
    
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.
    
    verbose : bool, default=False
        Controls the verbosity of the fitting process. If True, the progress
        of the fitting process is displayed.
    
    Attributes
    ----------
    base_estimators_ : list of DecisionTreeClassifier or DecisionTreeRegressor
        List of base learners, each a decision tree.
    
    weights_ : list
        Weights associated with each base learner, influencing their
        contribution to the final prediction.
    
    Notes
    -----
    - The performance of the ensemble model depends on the quality of the base
      estimators and the hyperparameters.
    - Proper tuning of the hyperparameters is essential for achieving good
      performance.
     
    The weighted tree ensemble model follows a boosting approach where each
    subsequent tree is trained to correct the errors of the previous trees.
    The model can be mathematically formulated as follows:
    
    1. **Initialization**:
       .. math::
           F_0(x) = 0
    
    2. **Iteration for each tree**:
       - For classifier:
         .. math::
             y_{\text{pred}_i} = \sum_{m=1}^{M} \alpha_m h_m(x_i)
         where :math:`\alpha_m` is the weight of the `m`-th tree, and 
         :math:`h_m(x_i)` is the prediction of the `m`-th tree for input `x_i`.
       - For regressor:
         .. math::
             r_i = y_i - \eta \cdot f_m(X_i)
         where :math:`r_i` is the residual for sample :math:`i`, :math:`y_i` is
         the true value for sample :math:`i`, :math:`\eta` is the learning rate,
         and :math:`f_m` is the prediction of the `m`-th tree.
    
    Examples
    --------
    >>> from gofast.estimators._base_tree import BaseWeightedTree
    >>> from gofast.estimators.tree import WeightedTreeClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> clf = WeightedTreeClassifier(n_estimators=10, eta0=0.1, max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    
    
    See Also
    --------
    - `sklearn.ensemble.GradientBoostingClassifier`: Scikit-learn's Gradient
      Boosting Classifier for comparison.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.
    - `sklearn.metrics.accuracy_score`: A common metric for evaluating
      classification models.
    
    References
    ----------
    .. [1] Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting
           Machine," The Annals of Statistics, 2001.
    .. [2] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
           Python," Journal of Machine Learning Research, 12:2825-2830.
    """
    _parameter_constraints: dict = {
        "eta0":  [Interval(Real, 0., 1.0, closed="both")],
        "splitter": [StrOptions({"best", "random"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"auto", "sqrt", "log2"}, deprecated={"auto"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
    }
    
    @abstractmethod
    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        verbose=False
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.criterion = criterion 
        self.splitter = splitter 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf 
        self.max_features = max_features 
        self.random_state = random_state 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.verbose = verbose

    @abstractmethod
    def _make_estimator(self):
        pass
    
    def fit(self, X, y, sample_weight=None, check_input=True ):
        """
        Fit the ensemble of weighted decision trees to the training data.
    
        This method trains the ensemble of decision trees on the provided 
        training data `X` and target values `y`. For classification tasks, it 
        adjusts the weights of the samples to focus on the ones that are 
        misclassified. For regression tasks, it updates the residuals at each 
        iteration to minimize the prediction error.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification, these are the class labels. For 
            regression, these are the continuous values to predict.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If None, all samples are given 
            equal weight. This parameter is used only for regression tasks.
            
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you're doing.
            
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes 
        -------
        The fitting process involves different steps for classification and 
        regression tasks:
    
        **For Classification:**
        1. **Initialization of sample weights**:
           .. math::
               w_i = \frac{1}{n}, \quad \forall i \in \{1, \ldots, n\}
           where `n` is the number of samples.
    
        2. **Iteration for each tree**:
           a. Train a decision tree on the weighted dataset.
           b. Predict the labels and calculate the weighted error:
              .. math::
                  \text{Weighted Error} = \sum_{i=1}^{n} (w_i \cdot (y_i \neq y_{\text{pred}_i}))
           c. Compute the weight for the current tree:
              .. math::
                  \alpha = \eta_0 \cdot \log\left(\frac{1 - \text{Weighted Error}}{\text{Weighted Error}}\right)
           d. Update the sample weights for the next iteration:
              .. math::
                  w_i = w_i \cdot \exp(\alpha \cdot (y_i \neq y_{\text{pred}_i}))
    
        **For Regression:**
        1. **Initialization of residuals**:
           .. math::
               r_i = y_i, \quad \forall i \in \{1, \ldots, n\}
    
        2. **Iteration for each tree**:
           a. Train a decision tree on the residuals.
           b. Predict the residuals and update them:
              .. math::
                  r_i = r_i - \eta_0 \cdot f_m(X_i)
           where `f_m(X_i)` is the prediction of the `m`-th tree for input `X_i`.
      
        - This method must be called before `predict` or `predict_proba`.
        - For classification, the sample weights are adjusted at each iteration 
          to focus on the samples that are misclassified.
        - For regression, the residuals are updated at each iteration to 
          minimize the prediction error.  
           
    
        Examples
        --------
        >>> from gofast.estimators._base_tree import BaseWeightedTree
        >>> from gofast.estimators.tree import WeightedTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> clf = WeightedTreeClassifier(n_estimators=10, eta0=0.1, max_depth=3)
        >>> clf.fit(X_train, y_train)
 
    
        See Also
        --------
        BaseWeightedTree.predict : Predict using the trained ensemble of trees.
    
        References
        ----------
        .. [1] Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting
               Machine," The Annals of Statistics, 2001.
        .. [2] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
               Python," Journal of Machine Learning Research, 12:2825-2830.
        """
        self._validate_params() 
        if check_input:
            # Need to validate separately here.
            # We can't pass multi_output=True because that would allow y to be
            # csr.
            check_X_params = dict(accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based sparse matrices"
                    )

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )
            
        self.base_estimators_ = []
        self.weights_ = []
        if self._is_classifier():
            sample_weights = self._compute_sample_weights(y)
        else:
            residuals = y

        if self.verbose:
            progress_bar = tqdm(range(self.n_estimators), ascii=True, ncols=100,
                                desc=f'Fitting {self.__class__.__name__}')

        for _ in range(self.n_estimators):
            base_estimator = self._make_estimator()

            if self._is_classifier():
                base_estimator.fit(X, y, sample_weight=sample_weights)

                y_pred = base_estimator.predict(X)
                errors = (y != y_pred)
                weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)

                if weighted_error == 0:
                    continue

                weight = self.eta0 * np.log((1 - weighted_error) / weighted_error)
                sample_weights = self._update_sample_weights(y, y_pred, weight)
            else:
                base_estimator.fit(X, residuals, sample_weight=sample_weight)
                predictions = base_estimator.predict(X)
                residuals -= self.eta0 * predictions
                weight = self.eta0

            self.base_estimators_.append(base_estimator)
            self.weights_.append(weight)

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self
    
    def predict(self, X):
        """
        Predict class labels or target values for samples in `X`.
    
        The `predict` method aggregates the predictions from each decision tree
        in the ensemble, weighted by their respective weights, to produce the 
        final prediction. For classification tasks, it outputs the class labels 
        based on the sign of the aggregated prediction. For regression tasks, 
        it outputs the continuous values.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row corresponds to a single sample, and each 
            column corresponds to a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for classification tasks or predicted target 
            values for regression tasks.
    
        Notes
        -----
        - Ensure that the `fit` method has been called before invoking `predict`.
        - The performance of the predictions depends on both the quality of the 
          base estimators and the accuracy of the weights assigned to them.
        - For classification tasks, the method outputs the class labels, while 
          for regression tasks, it outputs the continuous target values.
          
        The prediction process involves aggregating the predictions from each 
        decision tree in the ensemble:
    
        **For Classification:**
        1. Compute the weighted sum of predictions from all trees:
           .. math::
               \hat{y} = \sum_{m=1}^{M} \alpha_m h_m(x)
           where :math:`\alpha_m` is the weight of the `m`-th tree, and 
           :math:`h_m(x)` is the prediction of the `m`-th tree for input `x`.
    
        2. Normalize the aggregated predictions:
           .. math::
               \hat{y} = \frac{\hat{y}}{\sum_{m=1}^{M} \alpha_m}
           if weights are provided, otherwise:
           .. math::
               \hat{y} = \hat{y}
    
        3. Determine the class labels based on the sign of the aggregated prediction:
           .. math::
               \hat{y}_{\text{class}} = \text{sign}(\hat{y})
    
        **For Regression:**
        1. Compute the weighted sum of predictions from all trees:
           .. math::
               \hat{y} = \sum_{m=1}^{M} \alpha_m f_m(x)
           where :math:`\alpha_m` is the weight of the `m`-th tree, and 
           :math:`f_m(x)` is the prediction of the `m`-th tree for input `x`.
    
        Examples
        --------
        >>> from gofast.estimators._base_tree import BaseWeightedTree
        >>> from gofast.estimators.tree import WeightedTreeClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> clf = WeightedTreeClassifier(n_estimators=10, eta0=0.1, max_depth=3)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    
        See Also
        --------
        BaseWeightedTree.fit : Fit the ensemble of weighted decision trees.
        BaseWeightedTree.predict_proba : Predict class probabilities for samples in `X`.
    
        References
        ----------
        .. [1] Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting
               Machine," The Annals of Statistics, 2001.
        .. [2] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
               Python," Journal of Machine Learning Research, 12:2825-2830.
        """
        check_is_fitted(self, 'base_estimators_')
        X = check_array(X, accept_sparse=True)
        y_pred = np.zeros(X.shape[0])
    
        for base_estimator, weight in zip(self.base_estimators_, self.weights_):
            y_pred += weight * base_estimator.predict(X)
    
        if self._is_classifier():
            y_pred = y_pred / np.sum(self.weights_) if self.weights_ else y_pred
            return np.sign(y_pred)
        else:
            return y_pred
   
    @abstractmethod
    def _is_classifier(self):
        pass

    def _compute_sample_weights(self, y):
        """Compute sample weights."""
        return np.ones_like(y) / len(y)

    def _update_sample_weights(self, y, y_pred, weight):
        """Update sample weights."""
        return np.exp(-weight * y * y_pred)
    
class BaseDTB(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Decision Tree-Based Ensembles.

    This class serves as a foundational base for creating ensemble
    models using decision trees. It encapsulates common functionality
    and parameters for both classification and regression tasks.
    This class is intended to be inherited by specific ensemble
    classes such as `DTBClassifier` and `DTBRegressor`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble. This parameter controls
        how many decision trees will be fitted and aggregated to form
        the final model.

    max_depth : int, default=3
        The maximum depth of each decision tree in the ensemble. Limiting
        the depth of the trees helps prevent overfitting by restricting
        the complexity of the model.

    criterion : str, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        for regression are:
        - "squared_error": mean squared error, which is equal to variance
          reduction as feature selection criterion.
        - "friedman_mse": mean squared error with improvement score by Friedman.
        - "absolute_error": mean absolute error.
        - "poisson": Poisson deviance.

        For classification, supported criteria are:
        - "gini": Gini impurity.
        - "entropy": Information gain (entropy).

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are:
        - "best": Choose the best split.
        - "random": Choose the best random split.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum number of
          samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum number of
          samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When `max_features` < n_features,
        the algorithm will select `max_features` at random at each split
        before finding the best split among them. Pass an int for reproducible
        output across multiple function calls.

    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes
        are defined as relative reduction in impurity. If None, then unlimited
        number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        `ccp_alpha` will be chosen.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values
        indicate more messages.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the ensemble model to the data.

    predict(X)
        Predict using the fitted ensemble model.

    Notes
    -----
    The `BaseDTB` class combines the predictive power of multiple decision
    trees by averaging their predictions (for regression) or using majority
    voting (for classification). This reduces variance and improves accuracy
    over a single decision tree. The ensemble prediction is computed by
    aggregating the predictions from each individual tree within the ensemble.

    The mathematical formulation for regression can be described as follows:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from
      all trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the
      :math:`i`-th tree.

    For classification, the final class label is determined by majority voting:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input :math:`x`.
    - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the
      occurrence of the most frequent class label predicted by the :math:`i`-th tree.
    - :math:`n` is the number of decision trees in the ensemble.
    - :math:`C_i(x)` is the class label predicted by the :math:`i`-th decision tree.

    Examples
    --------
    Here's an example of how to use the `DTBClassifier` and `DTBRegressor`:

    >>> from gofast.estimators.tree import DTBClassifier, DTBRegressor
    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.model_selection import train_test_split

    Classification Example:

    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print("Predicted class labels:", y_pred)

    Regression Example:

    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print("Predicted values:", y_pred)

    See Also
    --------
    sklearn.ensemble.RandomForestClassifier : A popular ensemble method
        based on decision trees for classification tasks.
    sklearn.ensemble.RandomForestRegressor : A popular ensemble method
        based on decision trees for regression tasks.
    sklearn.tree.DecisionTreeClassifier : Decision tree classifier used as
        base learners in ensemble methods.
    sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as
        base learners in ensemble methods.

    References
    ----------
    .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
           Learning." Springer Series in Statistics. (2001).
    """
    _parameter_constraints: dict = {
        "splitter": [StrOptions({"best", "random"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="neither"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(Real, 0.0, 1.0, closed="right"),
            StrOptions({"auto", "sqrt", "log2"}, deprecated={"auto"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
    }
    
    @abstractmethod
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        ccp_alpha=0., 
        verbose=0
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None, check_input=True):
        """
        Fit the ensemble model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are given
            equal weight. This parameter is used to account for different
            importance of samples during the training process.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The `fit` method trains the ensemble model on the input data `X` and
        target labels `y`. The process involves the following steps:
    
        1. **Input Validation**: The input data `X` and target values `y` are
           checked for validity and conformance to the expected format and type
           using `check_X_y`.
    
        2. **Estimator Initialization**: An empty list `estimators_` is created
           to store the individual decision trees that will be trained.
    
        3. **Sample Indices Generation**: An array of sample indices is created
           to facilitate the selection of samples for training each tree.
    
        4. **Progress Bar Setup**: If `verbose` is greater than 0, a progress bar
           is initialized to provide visual feedback during the fitting process.
    
        5. **Bootstrap Sampling**: For each estimator, a subset of samples is
           randomly selected with replacement (if `bootstrap` is `True`) or without
           replacement (if `bootstrap` is `False`).
    
        6. **Tree Training**: A new decision tree is created and trained on the
           selected subset of samples.
    
        7. **Out-of-Bag Score Calculation**: If `bootstrap` is `True` and
           `subsample` is less than 1.0, the out-of-bag (OOB) score is calculated
           as the mean squared error on the samples not used for training (for
           regression tasks).
    
        The mathematical formulation of the ensemble prediction process is as follows:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
        where:
        - :math:`N` is the number of trees in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all trees.
        - :math:`y_{\text{tree}_i}` represents the prediction made by the
          :math:`i`-th tree.
    
        Examples
        --------
        Here's an example of how to use the `fit` method with `DTBClassifier` and
        `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBClassifier, DTBRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predicted class labels:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predicted values:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_X_y : Utility function to check the input
            data and target values.
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        self._validate_params() 
        if check_input: 
            check_X_params = dict(accept_sparse="csc")
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )
            if issparse(X):
                X.sort_indices()

                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError(
                        "No support for np.int64 index based sparse matrices"
                    )

            if self.criterion == "poisson":
                if np.any(y < 0):
                    raise ValueError(
                        "Some value(s) of y are negative which is"
                        " not allowed for Poisson regression."
                    )
                if np.sum(y) <= 0:
                    raise ValueError(
                        "Sum of y is not positive which is "
                        "necessary for Poisson regression."
                    )
        
        self.estimators_ = []
        n_samples = X.shape[0]
        sample_indices = np.arange(n_samples)
        sample_weight = validate_fit_weights(y, sample_weight=sample_weight)
        if self.verbose > 0:
            progress_bar = tqdm(
                range(self.n_estimators), ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )
        # TODO 
        self.n_estimators = validate_positive_integer(
            self.n_estimators, "n_estimators", round_float="ceil") 
        for i in range(self.n_estimators):
            if self.bootstrap:
                subsample_indices = np.random.choice(
                    sample_indices, size=int(self.subsample * n_samples), replace=True
                )
            else:
                subsample_indices = sample_indices[:int(self.subsample * n_samples)]
    
            tree = self._create_tree()
            tree.fit(X[subsample_indices], y[subsample_indices], 
                     sample_weight=sample_weight[subsample_indices])
            self.estimators_.append(tree)
    
            if self.verbose > 0:
                progress_bar.update(1)
    
        if self.verbose > 0:
            progress_bar.close()
    
        if self.bootstrap and self.subsample < 1.0:
            oob_indices = np.setdiff1d(sample_indices, subsample_indices)
            if len(oob_indices) > 0:
                oob_predictions = np.mean(
                    [tree.predict(X[oob_indices]) for tree in self.estimators_], axis=0
                )
                self.oob_score_ = np.mean((y[oob_indices] - oob_predictions) ** 2)
    
        return self
        
    def predict(self, X):
        """
        Predict using the ensemble model.
    
        This method generates predictions for the input samples `X`
        using the fitted ensemble model. It checks if the model is fitted
        and then uses the base estimators within the ensemble to make
        predictions.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. It can be a dense or sparse matrix.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values (class labels or continuous values).
    
        Notes
        -----
        The `predict` method performs the following steps:
    
        1. **Check if Fitted**: The method checks if the model has been fitted
           by verifying the presence of the `estimators_` attribute. If the model
           is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated using
           `check_array` to ensure they conform to the expected format and type.
           This includes handling both dense and sparse matrices.
    
        3. **Prediction**: The method collects predictions from each individual
           base estimator in the ensemble. These predictions are then aggregated
           to form the final output. For regression tasks, the predictions are
           averaged. For classification tasks, majority voting is used to determine
           the final class label.
    
        The mathematical formulation of the prediction process is as follows:
    
        For regression:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
        where:
        - :math:`N` is the number of trees in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all trees.
        - :math:`y_{\text{tree}_i}` represents the prediction made by the
          :math:`i`-th tree.
    
        For classification:
    
        .. math::
            C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)
    
        where:
        - :math:`C_{\text{final}}(x)` is the final predicted class label for input :math:`x`.
        - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the
          occurrence of the most frequent class label predicted by the :math:`i`-th tree.
        - :math:`n` is the number of decision trees in the ensemble.
        - :math:`C_i(x)` is the class label predicted by the :math:`i`-th decision tree.
    
        Examples
        --------
        Here's an example of how to use the `predict` method with `DTBClassifier` and
        `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBClassifier, DTBRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predicted class labels:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predicted values:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.utils.validation.check_is_fitted : Utility function to check if the
            estimator is fitted.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return self._aggregate_predictions(predictions)


    @abstractmethod
    def _create_tree(self):
        """
        Create a new decision tree instance.
    
        This abstract method should be implemented by subclasses to return
        an instance of a decision tree model (`DecisionTreeClassifier` or
        `DecisionTreeRegressor`) configured with the appropriate parameters.
    
        Returns
        -------
        tree : DecisionTreeClassifier or DecisionTreeRegressor
            A new instance of a decision tree model.
        """
        pass
    
    @abstractmethod
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from multiple trees.
    
        This abstract method should be implemented by subclasses to combine
        the predictions from all individual trees in the ensemble. For
        regression, this might involve averaging the predictions. For
        classification, this might involve majority voting.
    
        Parameters
        ----------
        predictions : array-like of shape (n_estimators, n_samples)
            The predictions from all individual trees.
    
        Returns
        -------
        aggregated_predictions : array-like of shape (n_samples,)
            The aggregated predictions.
        """
        pass
