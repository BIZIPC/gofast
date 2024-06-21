# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator

from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from ..tools.validator import check_array
from ..tools.validator import check_is_fitted, get_estimator_name

from sklearn.utils._param_validation import Interval, StrOptions, HasMethods
from numbers import Integral, Real

class BaseEnsemble(BaseEstimator, metaclass=ABCMeta):
    """
    BaseEnsemble

    The `BaseEnsemble` class serves as a foundational abstract base
    class for creating ensemble learning models. This class
    encapsulates the common functionality and parameters required
    for implementing both ensemble classifiers and regressors.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of base estimators in the ensemble. For the 
        hybrid strategy, this is the total number of base estimators 
        combined across bagging and boosting.
        
    eta0 : float, default=0.1
        Learning rate that shrinks the contribution of each base 
        estimator by `eta0`. This parameter is only used for the 
        boosting and hybrid strategies.
        
    max_depth : int, default=3
        The maximum depth of the individual estimators. This controls 
        the complexity of each base estimator.
        
    strategy : {'hybrid', 'bagging', 'boosting'}, default='hybrid'
        The strategy to use for the ensemble. Options are:
        - 'bagging': Use Bagging strategy.
        - 'boosting': Use Boosting strategy.
        - 'hybrid': Combine Bagging and Boosting strategies.
        
    random_state : int or RandomState, default=None
        Controls the randomness of the estimator for reproducibility. 
        Pass an int for reproducible output across multiple function 
        calls.
        
    max_samples : float or int, default=1.0
        The number of samples to draw from `X` to train each base 
        estimator. If float, then draw `max_samples * n_samples` samples.
        
    max_features : int or float, default=1.0
        The number of features to draw from `X` to train each base 
        estimator. If float, then draw `max_features * n_features` features.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling 
        without replacement is performed.
        
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
        
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the 
        generalization error.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to 
        fit and add more estimators to the ensemble.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`. 
        `None` means 1 unless in a `joblib.parallel_backend` context.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the 
        impurity greater than or equal to this value. Used to control 
        tree growth.
        
    init : estimator object, default=None
        An estimator object that is used to compute the initial predictions. 
        Used only for boosting.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        - If int, consider `min_samples_split` as the minimum number.
        - If float, `min_samples_split` is a fraction and 
          `ceil(min_samples_split * n_samples)` is the minimum number 
          of samples for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. 
        A split point at any depth will only be considered if it leaves 
        at least `min_samples_leaf` training samples in each of the left 
        and right branches.
        - If int, consider `min_samples_leaf` as the minimum number.
        - If float, `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` is the minimum number 
          of samples for each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights 
        required to be at a leaf node.
        
    max_leaf_nodes : int, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion. Best 
        nodes are defined as relative reduction in impurity. If None, 
        unlimited number of leaf nodes.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set 
        for early stopping. Used only for boosting.
        
    n_iter_no_change : int, default=None
        Used to decide if early stopping will be used to terminate 
        training when validation score is not improving. Used only for 
        boosting.
        
    tol : float, default=1e-4
        Tolerance for the early stopping. Used only for boosting.
        
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    
    estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset. 
        If `None`, the default base estimator will be used, which is 
        defined in the subclasses.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values 
        indicate more messages.
        
    Attributes
    ----------
    model_ : object
        The fitted ensemble model.
    
    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the ensemble model to the data.
        
    predict(X)
        Predict using the fitted ensemble model.
    
    Notes
    -----
    This model combines the predictive power of multiple base estimators 
    through bagging, boosting, or a hybrid approach, effectively reducing 
    variance and improving accuracy over a single estimator.
    
    The hybrid strategy uses a combination of bagging and boosting, 
    where the bagging model contains boosting models as its base estimators. 
    This leverages the strengths of both approaches to achieve better 
    performance.
    
    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
    where:
    - :math:`N` is the number of base estimators in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from 
      all base estimators.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the 
      :math:`i`-th base estimator.
    
    Examples
    --------
    Here's an example of how to use the `BaseEnsemble` class on a dataset:
    
    >>> from gofast.estimators._base import BaseEnsemble
    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score, mean_squared_error
    
    Classification Example:
    
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
    ...                          random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print("Classification accuracy:", accuracy_score(y_test, y_pred))
    
    Regression Example:
    
    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
    ...                         random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print("Regression MSE:", mean_squared_error(y_test, y_pred))
    
    See Also
    --------
    sklearn.ensemble.BaggingClassifier : A bagging classifier.
    sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
    sklearn.ensemble.BaggingRegressor : A bagging regressor.
    sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    sklearn.metrics.accuracy_score : A common metric for evaluating classification models.
    sklearn.metrics.mean_squared_error : A common metric for evaluating regression models.
    
    References
    ----------
    .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
           icml. Vol. 96. 1996.
    .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
           Learning." Springer Series in Statistics. (2001).
    """
    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict"]), None],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "eta0": [Interval(Real, 0, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "strategy": [StrOptions({"hybrid", "bagging", "boosting"})],
        "random_state": ["random_state"],
        "max_samples": [Interval(Integral, 1, None, closed="left"), 
                        Interval(Real, 0.0, 1.0, closed="both")],
        "max_features": [Interval(Integral, 1, None, closed="left"), 
                         Interval(Real, 0.0, 1.0, closed="both")],
        "bootstrap": [bool],
        "bootstrap_features": [bool],
        "oob_score": [bool],
        "warm_start": [bool],
        "n_jobs": [Interval(Integral, None, None, closed="neither"), None],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "init": [HasMethods(["fit", "predict"]), None],
        "min_samples_split": [Interval(Integral, 2, None, closed="left"), 
                              Interval(Real, 0.0, 1.0, closed="right")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left"), 
                             Interval(Real, 0.0, 1.0, closed="right")],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "validation_fraction": [Interval(Real, 0.0, 1.0, closed="both")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), "boolean"],
    }
    
    @abstractmethod
    def __init__(
        self,
        n_estimators=50,
        eta0=0.1,
        max_depth=3,
        strategy='hybrid',
        random_state=None,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        min_impurity_decrease=0.0,
        init=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        estimator=None,
        verbose=0
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.strategy = strategy
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ensemble model to the data.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the specified ensemble strategy.
        The available strategies are `bagging`, `boosting`, and `hybrid`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        Notes
        -----
        The `fit` method initializes the base estimator and then trains
        the ensemble model based on the specified strategy. The process
        can be summarized as follows:
    
        1. **Input Validation**: The input data `X` and target values `y`
           are checked for validity and conformance to the expected format
           and type using `check_X_y`.
    
        2. **Base Estimator Initialization**: If no base estimator is
           provided, the default base estimator is initialized with the
           specified `max_depth`.
    
        3. **Ensemble Strategy**: The chosen ensemble strategy is converted
           to lowercase for consistency. Based on the value of `strategy`,
           the corresponding `_fit_bagging`, `_fit_boosting`, or `_fit_hybrid`
           method is called to train the ensemble model.
    
        The mathematical formulation of the ensemble model can be described
        as follows:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{est}_i}
    
        where:
        - :math:`N` is the number of base estimators in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all base estimators.
        - :math:`y_{\text{est}_i}` represents the prediction made by the
          :math:`i`-th base estimator.
    
        Examples
        --------
        Here's an example of how to use the `fit` method with both
        `EnsembleClassifier` and `EnsembleRegressor`:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier, EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> print("Fitted ensemble classifier model:", clf)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
        ...                         random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> print("Fitted ensemble regressor model:", reg)
    
        See Also
        --------
        sklearn.utils.validation.check_X_y : Utility function to check the input
            data and target values.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        
        self._validate_params() 
        
        check_X_params = dict (accept_sparse=True, )
        check_y_params = dict(ensure_2d=False, dtype=None)
        X, y = self._validate_data(
            X, y, reset =True, 
            validate_separately= (check_X_params, check_y_params)
        )
        if self.estimator is None:
            self.estimator = self.default_estimator(max_depth=self.max_depth)

        self.strategy = str(self.strategy).lower()
        if self.strategy == 'bagging':
            self._fit_bagging(X, y, sample_weight, self.is_classifier)
        elif self.strategy == 'boosting':
            self._fit_boosting(X, y, sample_weight, self.is_classifier)
        elif self.strategy == 'hybrid':
            self._fit_hybrid(X, y, sample_weight, self.is_classifier)
        else:
            raise ValueError(
                "Invalid strategy, choose from 'hybrid', 'bagging', 'boosting'")

        return self
    
    def predict(self, X):
        """
        Predict using the fitted ensemble model.
    
        This method generates predictions for the input samples `X`
        using the fitted ensemble model. It checks if the model
        is fitted and then uses the base estimators within the
        ensemble to make predictions.
    
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
    
        1. **Check if Fitted**: The method checks if the model has been
           fitted by verifying the presence of the `model_` attribute.
           If the model is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated
           using `check_array` to ensure they conform to the expected
           format and type. This includes handling both dense and sparse
           matrices.
    
        3. **Prediction**: The method delegates the prediction task to
           the `predict` method of the fitted `model_` (which can be
           a `BaggingClassifier`, `BaggingRegressor`, `GradientBoostingClassifier`,
           or `GradientBoostingRegressor` depending on the strategy used).
    
        The mathematical formulation of the prediction process can be
        described as follows:
    
        .. math::
            \hat{y}_i = f(X_i)
    
        where:
        - :math:`\hat{y}_i` is the predicted value for the i-th sample.
        - :math:`f` is the ensemble model, which aggregates the predictions
          of the base estimators.
        - :math:`X_i` is the i-th input sample.
    
        Examples
        --------
        Here's an example of how to use the `predict` method with both
        `EnsembleClassifier` and `EnsembleRegressor`:
    
        >>> from gofast.estimators._base import EnsembleClassifier, EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predictions:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
        ...                         random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predictions:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.utils.validation.check_is_fitted : Utility function to check if the
            estimator is fitted.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        check_is_fitted(self, 'model_')
        X = check_array(
            X, accept_sparse=True,
            accept_large_sparse=True,
            estimator=get_estimator_name(self)
        )
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted ensemble model.
    
        This method generates probability estimates for the input samples
        `X` using the fitted ensemble model. This method is only applicable
        for classification tasks. If used with a regressor, it raises a
        `NotImplementedError`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. It can be a dense or sparse matrix.
    
        Returns
        -------
        p : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
    
        Notes
        -----
        The `predict_proba` method performs the following steps:
    
        1. **Check if Fitted**: The method checks if the model has been
           fitted by verifying the presence of the `model_` attribute.
           If the model is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated
           using `check_array` to ensure they conform to the expected
           format and type. This includes handling both dense and sparse
           matrices.
    
        3. **Probability Prediction**: The method delegates the probability
           prediction task to the `predict_proba` method of the fitted
           `model_` (which should be a `BaggingClassifier` or 
           `GradientBoostingClassifier`).
    
        If the ensemble model is a regressor, this method raises a
        `NotImplementedError` with the message "Probability estimates are
        not available for regressors."
    
        The mathematical formulation of the probability prediction process
        can be described as follows:
    
        .. math::
            P_{\text{class}} = \frac{1}{N} \sum_{i=1}^{N} P_{\text{est}_i}
    
        where:
        - :math:`N` is the number of base estimators in the ensemble.
        - :math:`P_{\text{class}}` is the final predicted probability
          aggregated from all base estimators.
        - :math:`P_{\text{est}_i}` represents the predicted probability
          made by the :math:`i`-th base estimator.
    
        Examples
        --------
        Here's an example of how to use the `predict_proba` method with
        `EnsembleClassifier`:
    
        >>> from gofast.estimators._base import EnsembleClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> probas = clf.predict_proba(X_test)
        >>> print("Predicted probabilities:", probas)

        """
        raise NotImplementedError(
            "Probability estimates are not available for regressors.")
        
    def _fit_bagging(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Bagging strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Bagging strategy. Depending
        on the value of `is_classifier`, it uses either a `BaggingClassifier`
        or a `BaggingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `BaggingClassifier` is used.
            - If `False`, a `BaggingRegressor` is used.
    
        Notes
        -----
        The `Bagging` strategy, short for Bootstrap Aggregating, involves
        training multiple base estimators on random subsets of the original
        dataset and aggregating their predictions. This approach reduces
        variance and improves the robustness of the model.
    
        For a given dataset :math:`(X, y)`, bagging works as follows:
        
        1. Randomly draw `n_estimators` subsets from the dataset with
           replacement (bootstrap sampling).
        2. Train a base estimator on each subset.
        3. Aggregate the predictions from all base estimators.
    
        The final prediction for classification is determined by majority
        voting (for discrete labels) or averaging probabilities (for
        probabilistic outputs). For regression, the final prediction is
        the average of all base estimators' predictions.
    
        The base estimator used can be specified by the `estimator`
        parameter. If not provided, a default estimator is used.
    
        Examples
        --------
        Here's an example of how to use the `_fit_bagging` method for
        both classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> estimator = DecisionTreeClassifier(max_depth=3)
        >>> obj = EnsembleClassifier(estimator=estimator)
        >>> obj._fit_bagging(X, y, sample_weight, is_classifier)
        >>> print("Fitted bagging classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> estimator = DecisionTreeRegressor(max_depth=3)
        >>> obj = EnsembleRegressor(estimator=estimator)
        >>> obj._fit_bagging(X, y, sample_weight, is_classifier)
        >>> print("Fitted bagging regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    
        """
        if is_classifier:
            self.model_ = BaggingClassifier(
                estimator=self.estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            self.model_ = BaggingRegressor(
                estimator=self.estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        self.model_.fit(X, y, sample_weight)

    def _fit_boosting(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Boosting strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Boosting strategy. Depending
        on the value of `is_classifier`, it uses either a 
        `GradientBoostingClassifier` or a `GradientBoostingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `GradientBoostingClassifier` is used.
            - If `False`, a `GradientBoostingRegressor` is used.
    
        Notes
        -----
        The `Boosting` strategy involves training multiple base estimators
        sequentially. Each estimator attempts to correct the errors of the
        previous one, making the model robust to overfitting and improving
        its predictive performance.
    
        For a given dataset :math:`(X, y)`, boosting works as follows:
    
        1. Initialize the model with a base estimator and a learning rate.
        2. Train the base estimator on the data.
        3. Adjust the weights of the training samples based on the error
           of the previous estimator.
        4. Train the next base estimator on the adjusted data.
        5. Repeat steps 3-4 for `n_estimators` iterations.
        6. Aggregate the predictions of all base estimators.
    
        The final prediction for classification is determined by majority
        voting (for discrete labels) or averaging probabilities (for
        probabilistic outputs). For regression, the final prediction is
        the weighted sum of all base estimators' predictions.
    
        The base estimator used can be specified by the `estimator`
        parameter. If not provided, a default estimator is used.
    
        Examples
        --------
        Here's an example of how to use the `_fit_boosting` method for
        both classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor 
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> estimator = GradientBoostingClassifier(max_depth=3)
        >>> obj = EnsembleClassifier(estimator=estimator)
        >>> obj._fit_boosting(X, y, sample_weight, is_classifier)
        >>> print("Fitted boosting classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> estimator = GradientBoostingRegressor(max_depth=3)
        >>> obj = EnsembleRegressor(estimator=estimator)
        >>> obj._fit_boosting(X, y, sample_weight, is_classifier)
        >>> print("Fitted boosting regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
               Learning." Springer Series in Statistics. (2001).
    
        """
        if is_classifier:
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        else:
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        self.model_.fit(X, y, sample_weight)
    
    def _fit_hybrid(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Hybrid strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Hybrid strategy, which combines
        bagging and boosting approaches. Depending on the value of
        `is_classifier`, it uses either a `GradientBoostingClassifier`
        or a `GradientBoostingRegressor` as the base estimator within
        a `BaggingClassifier` or `BaggingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples
            are given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `GradientBoostingClassifier` is used within a 
              `BaggingClassifier`.
            - If `False`, a `GradientBoostingRegressor` is used within a 
              `BaggingRegressor`.
    
        Notes
        -----
        The `Hybrid` strategy combines the strengths of both bagging and 
        boosting. Bagging reduces variance by training multiple base estimators 
        on random subsets of the dataset, while boosting reduces bias by 
        sequentially training base estimators, each correcting the errors of 
        the previous ones.
    
        For a given dataset :math:`(X, y)`, the hybrid strategy works as follows:
    
        1. Initialize the model with a bagging estimator and a boosting base 
           estimator.
        2. Use the boosting estimator to train on subsets of the data, with 
           each boosting iteration attempting to correct the errors of the 
           previous iteration.
        3. Use the bagging estimator to aggregate the boosting models.
        4. Repeat the process for `n_estimators` iterations.
    
        The final prediction for classification is determined by majority voting 
        (for discrete labels) or averaging probabilities (for probabilistic outputs). 
        For regression, the final prediction is the average of all base estimators' 
        predictions.
    
        Examples
        --------
        Here's an example of how to use the `_fit_hybrid` method for both 
        classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor 
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> obj = EnsembleClassifier()
        >>> obj._fit_hybrid(X, y, sample_weight, is_classifier)
        >>> print("Fitted hybrid classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> obj = EnsembleRegressor()
        >>> obj._fit_hybrid(X, y, sample_weight, is_classifier)
        >>> print("Fitted hybrid regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
               Learning." Springer Series in Statistics. (2001).
    
        """
        if is_classifier:
            estimator = GradientBoostingClassifier(
                n_estimators=self.n_estimators // 2,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        else:
            estimator = GradientBoostingRegressor(
                n_estimators=self.n_estimators // 2,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
    
        if is_classifier:
            self.model_ = BaggingClassifier(
                estimator=estimator,
                n_estimators=2,  # number of boosting models in the bagging
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            self.model_ = BaggingRegressor(
                estimator=estimator,
                n_estimators=2,  # number of boosting models in the bagging
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        self.model_.fit(X, y, sample_weight)