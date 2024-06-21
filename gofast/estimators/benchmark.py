# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y
from ..tools.validator import check_is_fitted, check_array
from .util import get_default_meta_estimator 

_logger = gofastlog().get_gofast_logger(__name__)

__all__=[ 
            "BenchmarkRegressor", "BenchmarkClassifier"   
    ]
class BenchmarkRegressor(BaseEstimator, RegressorMixin):
    """
    Benchmark Regressor for combining various regression estimators.

    This regressor operates as a stacking model, combining multiple base
    regression estimators to improve prediction accuracy. It benchmarks
    individual regressors and utilizes a meta-regressor for final predictions.
    The `BenchmarkRegressor` fits various base regressors on the dataset and
    uses their predictions as input for the meta-regressor to make final
    predictions.
    
    `BenchmarkRegressor` provides a versatile way to combine different 
    regression.estimators.benchmark using a stacking strategy. It's designed 
    to be flexible, allowing users to experiment with various combinations of  
    base.estimators.benchmark and meta-regressors.
    
    The stacking strategy can be mathematically represented as:

    .. math::
        \hat{y} = f_{\text{meta}}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)

    where :math:`\hat{y}` is the final prediction, :math:`f_{\text{meta}}` 
    represents the meta-regressor's prediction function, and :math:`\hat{y}_i` 
    are the predictions from the \(i\)-th base regressor. Each :math:`\hat{y}_i`
    is obtained by applying the corresponding base regressor to the input 
    features. This method allows for a refined prediction output by leveraging 
    the strengths of each base estimator and effectively combining their 
    individual predictive capabilities.

    Parameters
    ----------
    base_estimators : list of (str, estimator) tuples
        Base regressors to be used in the ensemble. Each tuple should contain
        a name and an estimator object.

    meta_regressor : estimator, default='LinearRegression'
        The meta-regressor to combine the base regressors' predictions.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy for base regressors.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit`.
        `None` means 1 unless in a joblib.parallel_backend context.
        `-1` means using all processors.

    passthrough : bool, default=False
        When False, only the predictions of the base estimators are used as
        training data for the meta-estimator. When True, the original training
        data is added to the meta-estimator training data.

    verbose : int, default=0
        Verbosity level.

    optimize_hyperparams : bool, default=False
        Whether to optimize hyperparameters of base regressors.

    meta_regressor_ensemble : bool, default=False
        Whether to use an ensemble of meta-regressors.

    optimizer : str, default='GSCV'
        The optimization method for hyperparameter tuning. Options include:
        'RSCV' (RandomizedSearchCV), 'GSCV' (GridSearchCV).

    ensemble_size : int, default=3
        The number of meta-regressors to use if `meta_regressor_ensemble` is True.

    param_grid : dict, default=None
        The parameter grid for hyperparameter optimization. If None, a default
        grid will be used.

    Attributes
    ----------
    stacked_model_ : StackingRegressor
        The underlying StackingRegressor model.

    Examples
    --------
    >>> from gofast.estimators.benchmark import BenchmarkRegressor
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> housing = fetch_california_housing()
    >>> X, y = housing.data, housing.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    >>> base_estimators = [
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor()),
            ('knn', KNeighborsRegressor())
        ]
    >>> meta_regressor = LinearRegression()
    >>> benchmark_reg = BenchmarkRegressor(base_estimators=base_estimators,
                                           meta_regressor=meta_regressor)
    >>> benchmark_reg.fit(X_train, y_train)
    >>> y_pred = benchmark_reg.predict(X_test)
    >>> print('R^2 Score:', benchmark_reg.score(X_test, y_test))

    Notes
    -----
    The choice of base regressors and the meta-regressor can significantly
    affect performance. Experiment with different combinations to find the best
    setup. The cross-validation strategy for base regressors can be tuned to
    balance between overfitting and underfitting.

    See Also
    --------
    StackingRegressor : Stacking regressor for combining estimators for regression.
    RandomForestRegressor : A random forest regressor.
    GradientBoostingRegressor : Gradient Boosting for regression.

    References
    ----------
    .. [1] Breiman, L. (1996). Stacked Regressions. Machine Learning.
           24(1):49-64.
    """
    def __init__(
        self, 
        base_estimators, 
        meta_regressor=None, 
        cv=None,
        stack_method='auto',
        n_jobs=None,
        passthrough=False, 
        verbose=0,
        optimize_hyperparams=False,
        meta_regressor_ensemble=False,
        optimizer='GSCV', 
        ensemble_size=3, 
        param_grid=None
    ):
        self.base_estimators = base_estimators
        self.meta_regressor = get_default_meta_estimator(meta_regressor)
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose
        self.optimize_hyperparams = optimize_hyperparams
        self.meta_regressor_ensemble = meta_regressor_ensemble
        self.optimizer = optimizer
        self.ensemble_size = ensemble_size
        self.param_grid = param_grid or {}
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Benchmark Regressor model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It is the observed data at training and
            prediction time, used as independent variables in learning.
        y : array-like of shape (n_samples,)
            The target values. It is the dependent variable in learning, usually
            the target of prediction.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
    
        Returns
        -------
        self : BenchmarkRegressor instance
            Returns self for easy method chaining.
    
        Notes
        -----
        This method fits the base regressors and the meta-regressor to the
        training data. If `optimize_hyperparams` is True, it performs
        hyperparameter optimization for the base regressors using the specified
        optimizer.
    
        See Also
        --------
        StackingRegressor : Stacking regressor for combining estimators for regression.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkRegressor
        >>> from sklearn.datasets import fetch_california_housing
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> from sklearn.neighbors import KNeighborsRegressor
        >>> housing = fetch_california_housing()
        >>> X, y = housing.data, housing.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        >>> base_estimators = [
                ('lr', LinearRegression()),
                ('dt', DecisionTreeRegressor()),
                ('knn', KNeighborsRegressor())
            ]
        >>> meta_regressor = LinearRegression()
        >>> benchmark_reg = BenchmarkRegressor(base_estimators=base_estimators,
                                               meta_regressor=meta_regressor)
        >>> benchmark_reg.fit(X_train, y_train)
    
        References
        ----------
        .. [1] Breiman, L. (1996). Stacked Regressions. Machine Learning.
               24(1):49-64.
        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__, to_frame=True)
        
        if self.verbose:
            print("Starting fit of Benchmark Regressor")
    
        # Optimize hyperparameters of base regressors if enabled
        if self.optimize_hyperparams:
            if self.verbose:
                print("Optimizing hyperparameters of base regressors...")
            self.base_estimators = [
                (name, self._optimize_hyperparams(estimator, X, y))
                for name, estimator in self.base_estimators
            ]
    
        # Create and fit the stacking model
        self.stacked_model_ = StackingRegressor(
            estimators=self.base_estimators,
            final_estimator=self._create_meta_regressor(X, y),
            n_jobs=self.n_jobs,
            passthrough=self.passthrough,
            cv=self.cv
        )
        
        if self.verbose:
            print("\nFitting stacking model...")
    
        self.stacked_model_.fit(X, y, sample_weight=sample_weight)
        
        if self.verbose:
            print("Benchmark Regressor fit completed")
    
        return self
    
    def _optimize_hyperparams(self, estimator, X, y):
        """
        Optimize hyperparameters of the given estimator.
    
        Parameters
        ----------
        estimator : estimator object
            The estimator for which to optimize hyperparameters.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        best_estimator : estimator object
            The estimator with the best found hyperparameters.
    
        Notes
        -----
        This method uses the specified optimizer to perform hyperparameter
        optimization for the given estimator. The parameter grid for the
        optimization is specified by the `param_grid` attribute.
    
        References
        ----------
        .. [1] Bergstra, J. and Bengio, Y. (2012). Random Search for Hyper-Parameter
               Optimization. Journal of Machine Learning Research.
        """
        from ..models.utils import get_optimizer_method 
        optimizer = get_optimizer_method(self.optimizer)
        grid_search = optimizer(
            estimator, self.param_grid,
            cv=self.cv, 
            n_jobs=self.n_jobs
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def _create_meta_regressor(self, X, y):
        """
        Create the meta-regressor for the stacking model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        meta_regressor : estimator object
            The meta-regressor, which can be an ensemble if
            `meta_regressor_ensemble` is True.
    
        Notes
        -----
        This method creates the meta-regressor for the stacking model. If
        `meta_regressor_ensemble` is True, it creates an ensemble of
        meta-regressors and fits them on the meta-features derived from the
        predictions of the base regressors.
    
        See Also
        --------
        StackingRegressor : Stacking regressor for combining estimators for regression.
        """
        if self.meta_regressor_ensemble:
            if self.verbose:
                print("Using ensemble of meta-regressors...")
            # Create an ensemble of meta-regressors
            meta_regressors = [
                clone(self.meta_regressor) 
                for _ in range(self.ensemble_size)
            ]
            meta_predictions = []
            if self.verbose: 
                progress_bar = tqdm(
                    meta_regressors, ascii=True, ncols=100,
                    desc='Fitting meta-regressors'
                )
            for meta_reg in meta_regressors:
                meta_predictions.append(
                    cross_val_predict(meta_reg, X, y, cv=self.cv, 
                                      method='predict'))
                if self.verbose: 
                    progress_bar.update(1) 
            if self.verbose: 
                progress_bar.close()
            
            # Combine meta_predictions into a 2D array
            meta_features = np.column_stack(meta_predictions)
            
            ensemble_meta_regressor = clone(
                self.meta_regressor).fit(meta_features, y)
            return ensemble_meta_regressor
        else:
            return self.meta_regressor
    
    def predict(self, X):
        """
        Predict target values for samples in `X`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
    
        Notes
        -----
        This method uses the fitted stacking model to predict the target values for
        the input samples. It ensures that the model is fitted before making
        predictions.
    
        See Also
        --------
        StackingRegressor : Stacking regressor for combining estimators for regression.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkRegressor
        >>> from sklearn.datasets import load_boston
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> benchmark_reg = BenchmarkRegressor(base_estimators=base_estimators,
                                               meta_regressor=meta_regressor)
        >>> benchmark_reg.fit(X, y)
        >>> y_pred = benchmark_reg.predict(X)
        """
        check_is_fitted(self, 'stacked_model_')
        X = check_array(X, accept_sparse=True)
        return self.stacked_model_.predict(X)
    
    def score(self, X, y):
        """
        Compute the score from the `stacked_model_`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
        y : array-like of shape (n_samples,)
            True values for `X`.
    
        Returns
        -------
        score : float
            The R^2 score of the prediction.
    
        Notes
        -----
        This method uses the fitted stacking model to compute the R^2 score for the
        predictions on the input samples. It ensures that the model is fitted
        before computing the score.
    
        See Also
        --------
        StackingRegressor : Stacking regressor for combining estimators for regression.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkRegressor
        >>> from sklearn.datasets import load_boston
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> benchmark_reg = BenchmarkRegressor(base_estimators=base_estimators,
                                               meta_regressor=meta_regressor)
        >>> benchmark_reg.fit(X, y)
        >>> score = benchmark_reg.score(X, y)
        >>> print('R^2 Score:', score)
        """
        check_is_fitted(self, 'stacked_model_')
        return self.stacked_model_.score(X, y)


class BenchmarkClassifier(BaseEstimator, ClassifierMixin):
    """
    Benchmark Classifier for combining various classification.estimators.

    This classifier operates as a stacking model, combining multiple base
    classifiers to improve prediction accuracy. It benchmarks individual
    classifiers and utilizes a meta-classifier for final predictions. The
    `BenchmarkClassifier` fits various base classifiers on the dataset and
    uses their predictions as input for the meta-classifier to make final
    predictions.
    
    `BenchmarkClassifier` provides a flexible way to combine different 
    classification.estimators.benchmark using a stacking strategy. The included 
    `predict_proba` method allows for the estimation of class probabilities, 
    which is crucial in many classification scenarios.
    
    The stacking strategy can be mathematically represented as:

    .. math::
        \hat{y} = f_{\text{meta}}(C_1(x), C_2(x), \ldots, C_m(x))

    where :math:`\hat{y}` is the final prediction, :math:`f_{\text{meta}}` 
    represents the meta-classifier's prediction function, and 
    :math:`C_i(x)` are the predictions from the \(i\)-th base classifier. 
    Each :math:`C_i(x)` computes the predicted output for the input \(x\), 
    contributing to the overall decision made by the meta-classifier.

    Parameters
    ----------
    base_classifiers : list of (str, estimator) tuples
        Base classifiers to be used in the ensemble.
        Each tuple should contain a name and an estimator object.
        
    meta_classifier : estimator, default=`LogisticRegression`
        The meta-classifier to combine the base classifiers' predictions.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy for base classifiers.
        
    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'},\
        default='auto'
        Methods called for each base estimator. It can be:
        - 'auto' for using 'predict_proba' if available, otherwise 'decision_function'
          if available, otherwise 'predict'.
        - 'predict_proba' to use `predict_proba`.
        - 'decision_function' to use `decision_function`.
        - 'predict' to use `predict`.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit`.
        `None` means 1 unless in a joblib.parallel_backend context.
        `-1` means using all processors.
        
    passthrough : bool, default=False
        When False, only the predictions of the base estimators are used as
        training data for the meta-estimator. When True, the
        `original training data` is added to the meta-estimator training data.
        
    verbose : int, default=0
        Verbosity level.
        
    optimize_hyperparams : bool, default=False
        Whether to optimize hyperparameters of base classifiers.
        
    meta_classifier_ensemble : bool, default=False
        Whether to use an ensemble of meta-classifiers.
        
    optimizer : str, default='RSCV'
        The optimization method for hyperparameter tuning. Options include:
        'RSCV' (RandomizedSearchCV), 'GSCV' (GridSearchCV).
        
    ensemble_size : int, default=3
        The number of meta-classifiers to use if `meta_classifier_ensemble` is True.
        
    param_grid : dict, default=None
        The parameter grid for hyperparameter optimization. If None, a default
        grid will be used.

    Attributes
    ----------
    stacked_model_ : StackingClassifier
        The underlying StackingClassifier model.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from gofast.estimators.benchmark import BenchmarkClassifier
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define base classifiers and meta classifier
    >>> base_classifiers = [
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier())
        ]
    >>> meta_classifier = LogisticRegression()

    # Create and fit the BenchmarkClassifier
    >>> benchmark_clf = BenchmarkClassifier(base_classifiers=base_classifiers,
                                            meta_classifier=meta_classifier)
    >>> benchmark_clf.fit(X_train, y_train)
    >>> y_pred = benchmark_clf.predict(X_test)
    >>> print('Accuracy:', benchmark_clf.score(X_test, y_test))

    Notes
    -----
    - The choice of base classifiers and the meta-classifier can significantly affect
      the performance. It's recommended to experiment with different combinations.
    - The cross-validation strategy for base classifiers can be tuned to balance
      between overfitting and underfitting.
    - Hyperparameter optimization can improve the performance of the base classifiers
      and the meta-classifier.

    See Also
    --------
    StackingClassifier : Stacking classifier for combining estimators for classification.
    RandomForestClassifier : A random forest classifier.
    GradientBoostingClassifier : Gradient Boosting for classification.

    References
    ----------
    .. [1] Breiman, L. (1996). Stacked Regressions. Machine Learning.
           24(1):49-64.
    """
    def __init__(
        self, 
        base_classifiers, 
        meta_classifier=None, 
        cv=None,
        stack_method='auto',
        n_jobs=None,
        passthrough=False, 
        verbose=0,
        optimize_hyperparams=False,
        meta_classifier_ensemble=False,
        optimizer='GSCV', 
        ensemble_size=3, 
        param_grid=None
    ):
        self.base_classifiers = base_classifiers
        self.meta_classifier = get_default_meta_estimator(meta_classifier, 'clf') 
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose
        self.optimize_hyperparams = optimize_hyperparams
        self.meta_classifier_ensemble = meta_classifier_ensemble
        self.optimizer = optimizer
        self.ensemble_size = ensemble_size
        self.param_grid = param_grid or {}
        
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Benchmark Classifier model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It is the observed data at training and
            prediction time, used as independent variables in learning.
        y : array-like of shape (n_samples,)
            The target values. It is the dependent variable in learning, usually 
            the target of prediction.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
    
        Returns
        -------
        self : BenchmarkClassifier instance
            Returns self for easy method chaining.
    
        Notes
        -----
        This method fits the base classifiers and the meta-classifier to the
        training data. If `optimize_hyperparams` is True, it performs
        hyperparameter optimization for the base classifiers using the specified
        optimizer.
    
        See Also
        --------
        StackingClassifier : Stacking classifier for combining estimators for classification.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        >>> base_classifiers = [
                ('lr', LogisticRegression()),
                ('dt', DecisionTreeClassifier()),
                ('knn', KNeighborsClassifier())
            ]
        >>> meta_classifier = LogisticRegression()
        >>> benchmark_clf = BenchmarkClassifier(base_classifiers=base_classifiers,
                                                meta_classifier=meta_classifier)
        >>> benchmark_clf.fit(X_train, y_train)
        """
        X, y = check_X_y(X, y, accept_sparse= True, 
                         estimator=self.__class__.__name__)
        
        if self.verbose:
            print("Starting fit of Benchmark Classifier")
    
        # Optimize hyperparameters of base classifiers if enabled
        if self.optimize_hyperparams:
            if self.verbose:
                print("Optimizing hyperparameters of base classifiers...")
            self.base_classifiers = [
                (name, self._optimize_hyperparams(estimator, X, y))
                for name, estimator in self.base_classifiers
            ]
    
        # Create and fit the stacking model
        self.stacked_model_ = StackingClassifier(
            estimators=self.base_classifiers,
            final_estimator=self._create_meta_classifier(X, y),
            stack_method=self.stack_method,
            n_jobs=self.n_jobs,
            passthrough=self.passthrough,
            cv=self.cv
        )
        
        if self.verbose:
            print("\nFitting stacking model...")
    
        self.stacked_model_.fit(X, y, sample_weight=sample_weight)
        
        if self.verbose:
            print("Benchmark Classifier fit completed")
    
        return self
    
    def _optimize_hyperparams(self, estimator, X, y):
        """
        Optimize hyperparameters of the given estimator.
    
        Parameters
        ----------
        estimator : estimator object
            The estimator for which to optimize hyperparameters.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        best_estimator : estimator object
            The estimator with the best found hyperparameters.
    
        Notes
        -----
        This method uses the specified optimizer to perform hyperparameter
        optimization for the given estimator. The parameter grid for the
        optimization is specified by the `param_grid` attribute.
    
        References
        ----------
        .. [1] Bergstra, J. and Bengio, Y. (2012). Random Search for Hyper-Parameter
               Optimization. Journal of Machine Learning Research.
        """
        from ..models.utils import get_optimizer_method 
        optimizer = get_optimizer_method(self.optimizer)
        grid_search = optimizer(
            estimator, self.param_grid,
            cv=self.cv, 
            n_jobs=self.n_jobs
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def _create_meta_classifier(self, X, y):
        """
        Create the meta-classifier for the stacking model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        meta_classifier : estimator object
            The meta-classifier, which can be an ensemble if
            `meta_classifier_ensemble` is True.
    
        Notes
        -----
        This method creates the meta-classifier for the stacking model. If
        `meta_classifier_ensemble` is True, it creates an ensemble of
        meta-classifiers and fits them on the meta-features derived from the
        predictions of the base classifiers.
    
        See Also
        --------
        StackingClassifier : Stacking classifier for combining estimators for classification.
        """
        if self.meta_classifier_ensemble:
            if self.verbose:
                print("Using ensemble of meta-classifiers...")
            # Create an ensemble of meta-classifiers
            meta_classifiers = [
                clone(self.meta_classifier) 
                for _ in range(self.ensemble_size)
            ]
            meta_predictions = []
            if self.verbose: 
                progress_bar = tqdm(
                    meta_classifiers, ascii=True, ncols=100,
                    desc='Fitting meta-classifiers'
                )
            for meta_clf in meta_classifiers:
                meta_predictions.append(
                    cross_val_predict(meta_clf, X, y, cv=self.cv, 
                                      method='predict_proba'))
                if self.verbose: 
                    progress_bar.update(1) 
            if self.verbose: 
                progress_bar.close()
                
            meta_features = np.mean(meta_predictions, axis=0)
            ensemble_meta_classifier = clone(
                self.meta_classifier).fit(meta_features, y)
            return ensemble_meta_classifier
        else:
            return self.meta_classifier
    
    def predict(self, X):
        """
        Predict the class labels for the input samples.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
    
        Notes
        -----
        This method uses the fitted stacking model to predict the class labels for
        the input samples. It ensures that the model is fitted before making
        predictions.
    
        See Also
        --------
        StackingClassifier : Stacking classifier for combining estimators for classification.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkClassifier
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> benchmark_clf = BenchmarkClassifier(base_classifiers=base_classifiers,
                                                meta_classifier=meta_classifier)
        >>> benchmark_clf.fit(X, y)
        >>> y_pred = benchmark_clf.predict(X)
        """
        check_is_fitted(self, 'stacked_model_')
        X = check_array(X, accept_sparse=True)
        return self.stacked_model_.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
    
        Notes
        -----
        This method uses the fitted stacking model to predict the class
        probabilities for the input samples. It ensures that the model is fitted
        before making predictions.
    
        See Also
        --------
        StackingClassifier : Stacking classifier for combining estimators for classification.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import BenchmarkClassifier
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> benchmark_clf = BenchmarkClassifier(base_classifiers=base_classifiers,
                                                meta_classifier=meta_classifier)
        >>> benchmark_clf.fit(X, y)
        >>> y_proba = benchmark_clf.predict_proba(X)
        """
        check_is_fitted(self, 'stacked_model_')
        X = check_array(X, accept_sparse=True)
        return self.stacked_model_.predict_proba(X)

