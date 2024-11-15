# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Ensure the reliability of machine learning models and pipelines
through robust testing.
"""

import time
import json
from numbers import Integral, Real
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from packaging.version import Version

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils._param_validation import HasMethods, StrOptions

from ..compat.sklearn import validate_params, Interval
from ..decorators import RunReturn, smartFitRun
from ..tools.depsutils import ensure_pkg
from ..tools.validator import (
    check_X_y, check_array,  check_is_fitted, check_is_runned
)
from ._base import BaseTest
from ._config import INSTALL_DEPENDENCIES, USE_CONDA


__all__ = [
    'PipelineTest', 
    'ModelQuality',  
    'OverfittingDetection',  
    'DataIntegrity', 
    'BiasDetection', 
    'ModelVersionCompliance',  
    'PerformanceRegression',  
    'CIIntegration'
]



@smartFitRun
class PipelineTest(BaseTest):
    """
    Tests and evaluates scikit-learn pipelines with built-in metrics tracking
    and parallel execution capabilities. Integrates with scoring functions
    and enables automated logging of events throughout the test lifecycle.

    The test process fits the pipeline to provided data, evaluates it based
    on predefined metrics, and optionally tracks those metrics through
    custom scoring functions.

    .. math::
        M(X, y) = f(\\text{Pipeline}(X), y)

    where :math:`M(X, y)` represents the metrics evaluated on the data
    :math:`X` with the true labels :math:`y` using a pipeline-based
    function :math:`\\text{Pipeline}(X)`.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The scikit-learn pipeline that will be fitted and evaluated.

    scoring : callable or None, default=None
        A scoring function to evaluate the fitted pipeline. The scoring
        function should have the signature ``scoring(estimator, X, y) -> float``.
        If `track_metrics` is ``True``, this function is mandatory.

    track_metrics : bool, default=True
        If ``True``, the test will track and evaluate performance metrics.

    metrics : list of str or None, default=None
        A list of metrics to evaluate during the test. Default metrics
        include ``'accuracy'``, ``'precision'``, and ``'recall'``.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, the test will be executed in parallel, enabling
        faster processing of large datasets.

    store_results : bool, default=True
        If ``True``, test results will be stored.

    logging_level : {'NONE', 'INFO', 'DEBUG'}, default='INFO'
        The logging level. Set to ``'NONE'`` to disable logging.

    **config_params : dict
        Additional configuration parameters passed to the test for
        customization.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to be tested and evaluated.

    scoring : callable or None
        A function for scoring the fitted pipeline.

    track_metrics : bool
        Indicates whether performance metrics should be tracked.

    metrics : list of str
        A list of metrics used for evaluation.

    results_ : dict
        Stores the results from the fitting and evaluation process.

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the pipeline to the provided data and, if tracking is
        enabled, computes the fit score using the scoring function.

    evaluate_metrics(X, y)
        Evaluates the specified metrics on the test data.

    Notes
    -----
    - This class requires scikit-learn's pipeline for execution.
    - The `track_metrics` flag enables custom scoring; you must provide
      a scoring function if this flag is set to ``True``.
    - The `fit` method should be called before `evaluate_metrics`.

    Examples
    --------
    >>> from gofast.mlops.testing import PipelineTest
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import make_scorer, accuracy_score
    >>>
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', LogisticRegression())
    ... ])
    >>> scorer = make_scorer(accuracy_score)
    >>> test = PipelineTest(pipeline=pipeline, scoring=scorer)
    >>> X_train, X_test, y_train, y_test = ...
    >>> test.fit(X_train, y_train)
    >>> metrics = test.evaluate_metrics(X_test, y_test)
    >>> print(metrics)

    See Also
    --------
    sklearn.pipeline.Pipeline : The base scikit-learn pipeline.
    sklearn.metrics : Scoring functions for evaluating pipelines.

    References
    ----------
    .. [1] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in
       Python," Journal of Machine Learning Research, 12, pp. 2825-2830.
    """

    @validate_params({
        'pipeline': [HasMethods(['fit', 'predict'])],
        'scoring': [callable, None],
        'track_metrics': [bool],
        'metrics': [list, None],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'logging_level': [StrOptions({'NONE', 'INFO', 'DEBUG'})],
    })
    def __init__(
        self,
        pipeline,
        scoring: Optional[Callable] = None,
        track_metrics: bool = True,
        metrics: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        logging_level: str = "INFO",
        **config_params
    ):

        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=logging_level != "NONE",
            **config_params
        )
        self.pipeline = pipeline
        self.scoring = scoring
        self.track_metrics = track_metrics
        self.metrics = metrics or ["accuracy", "precision", "recall"]
        self.results_ = {}

        if self.track_metrics and not self.scoring:
            raise ValueError(
                "Scoring function must be provided if track_metrics is True."
            )

    def fit(self, X, y=None, **fit_params):
        """
        Fits the pipeline to the provided data. If metrics tracking is
        enabled, computes the fit score using the provided scoring function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values. If ``None``, `X` is expected to be the entire
            dataset.

        **fit_params : dict
            Additional parameters passed to the pipeline's `fit` method.

        Returns
        -------
        self : PipelineTest
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `track_metrics` is ``True`` but no scoring function is provided.

        Notes
        -----
        If `track_metrics` is enabled and the scoring function is provided,
        the fit score is logged and stored in the `results_` dictionary.
        """
        if y is None:
            X = check_array(X, input_name='X')
        else:
            X, y = check_X_y(X, y, multi_output=True)

        self.pipeline.fit(X, y, **fit_params)
        self._set_fitted()

        if self.track_metrics and self.scoring:
            fit_score = self.scoring(self.pipeline, X, y)
            self.results_["fit_score"] = fit_score
            self.log_event("fit_completed", {"fit_score": fit_score})

        return self

    def evaluate_metrics(self, X, y):
        """
        Evaluates the specified metrics on the provided test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values.

        Returns
        -------
        metrics : dict
            A dictionary containing the computed metrics.

        Raises
        ------
        ValueError
            If the pipeline is not fitted before metrics evaluation.

        Notes
        -----
        By default, accuracy is computed as a metric, but other metrics such
        as precision and recall can also be evaluated if specified.
        """
        check_is_fitted(self.pipeline)
        self.check_is_fitted("Pipeline must be fitted before evaluating metrics.")

        results = {}
        y_pred = self.pipeline.predict(X)

        if "accuracy" in self.metrics:
            results["accuracy"] = accuracy_score(y, y_pred)

        if "precision" in self.metrics:
            results["precision"] = precision_score(y, y_pred, average='weighted')

        if "recall" in self.metrics:
            results["recall"] = recall_score(y, y_pred, average='weighted')

        if "f1" in self.metrics:
            results["f1"] = f1_score(y, y_pred, average='weighted')
            

        self.results_["metrics"] = results
        self.log_event("metrics_evaluated", {"metrics": results})
        return results


@smartFitRun
class ModelQuality(BaseTest):
    """
    Evaluates the performance of machine learning models by calculating
    various metrics on training and testing datasets. Supports cross-validation,
    custom metrics, parallel execution, and logging of results.

    The evaluation process includes splitting the data into training and
    testing sets (if required), fitting the model, calculating specified
    performance metrics, and optionally performing cross-validation.

    The mathematical formulation behind common classification metrics like
    accuracy, precision, recall, and F1-score is as follows:

    .. math::
        \\text{Accuracy} = \\frac{TP + TN}{TP + TN + FP + FN}

    where :math:`TP`, :math:`TN`, :math:`FP`, and :math:`FN` are the true
    positives, true negatives, false positives, and false negatives,
    respectively.

    Parameters
    ----------
    model : estimator object
        The machine learning model to be evaluated. The model should follow
        the scikit-learn API, having `fit` and `predict` methods.

    metrics : dict of str to callable, default=None
        A dictionary of metric names and corresponding scoring functions.
        Default metrics include ``'accuracy'``, ``'precision'``,
        ``'recall'``, and ``'f1_score'``.

    cross_validation : bool, default=False
        If ``True``, performs cross-validation in addition to regular
        evaluation.

    cv_folds : int, default=5
        The number of folds for cross-validation.

    evaluation_split : float, default=0.2
        The proportion of data to be used as the test set. If set to ``1.0``,
        no data splitting occurs.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution for faster processing.

    store_results : bool, default=True
        Whether to store the results of the evaluation.

    logging_level : {'NONE', 'INFO', 'DEBUG'}, default='INFO'
        The logging level. Set to ``'NONE'`` to disable logging.

    **config_params : dict
        Additional configuration parameters passed to the test for
        customization.

    Attributes
    ----------
    results_ : dict
        Stores the results of the evaluation, including metrics and
        cross-validation scores if applicable.

    Methods
    -------
    fit(X, y, **fit_params)
        Fits the model to the training data and evaluates metrics on both
        training and testing datasets.

    Notes
    -----
    - This class assumes the input model follows the scikit-learn API, with
      `fit` and `predict` methods.
    - Custom metrics must be passed as a dictionary, where each key is a
      metric name and each value is a callable (usually a function from
      scikit-learn's metrics module).

    Examples
    --------
    >>> from gofast.mlops.testing import ModelQuality
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> model = RandomForestClassifier(random_state=42)
    >>> evaluator = ModelQuality(model=model, metrics={'accuracy': accuracy_score})
    >>> X, y = ...  # Load or generate data
    >>> evaluator.fit(X, y)
    >>> print(evaluator.results_)

    See Also
    --------
    sklearn.model_selection.train_test_split : Utility function to split the data.
    sklearn.model_selection.cross_val_score : Function to perform cross-validation.
    sklearn.metrics : Module with standard evaluation metrics.

    References
    ----------
    .. [1] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in
       Python," Journal of Machine Learning Research, 12, pp. 2825-2830.
    """

    @validate_params({
        'model': [HasMethods(['fit', 'predict'])],
        'metrics': [dict, None],
        'cross_validation': [bool],
        'cv_folds': [int],
        'evaluation_split': [Interval(Real, 0, 1, closed='right')],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'logging_level': [StrOptions({'NONE', 'INFO', 'DEBUG'})],
    })
    def __init__(
        self,
        model,
        metrics: Optional[Dict[str, Callable]] = None,
        cross_validation: bool = False,
        cv_folds: int = 5,
        evaluation_split: float = 0.2,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        logging_level: str = "INFO",
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=logging_level != "NONE",
            **config_params
        )
        self.model = model
        self.metrics = metrics or {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score
        }
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.evaluation_split = evaluation_split
        self.results_ = {}

    def fit(self, X, y, **fit_params):
        """
        Fits the model to the provided data and evaluates the specified
        metrics on both training and testing datasets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        **fit_params : dict
            Additional parameters passed to the model's `fit` method.

        Returns
        -------
        self : ModelQuality
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `X` or `y` is None.

        Notes
        -----
        This method splits the data into training and testing sets, fits
        the model on the training set, and then evaluates metrics on both
        training and testing data.

        If `cross_validation` is enabled, cross-validation is performed
        using the specified number of folds (`cv_folds`).

        The evaluation results are stored in the `results_` attribute.
        """
        if X is None or y is None:
            raise ValueError("Both X and y are required to fit the model.")

        X, y = check_X_y(X, y, multi_output=True)
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Fit the model on training data
        self.model.fit(X_train, y_train, **fit_params)
        self._set_fitted()

        # Evaluate metrics on training data
        train_metrics = self._evaluate_metrics(X_train, y_train, data_type="train")
        # Evaluate metrics on testing data
        test_metrics = self._evaluate_metrics(X_test, y_test, data_type="test")

        self.results_['train_metrics'] = train_metrics
        self.results_['test_metrics'] = test_metrics

        # Optionally perform cross-validation
        if self.cross_validation:
            cv_scores = self._cross_validate(X_train, y_train)
            self.results_['cross_val_scores'] = cv_scores

        return self

    def _split_data(self, X, y):
        """
        Splits the input data into training and testing sets based on the
        `evaluation_split` ratio.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of arrays
            The split data for training and testing.

        Notes
        -----
        If `evaluation_split` is set to 1.0, no splitting is performed,
        and the entire dataset is used for both training and testing.
        """
        if self.evaluation_split < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.evaluation_split, random_state=self.random_seed
            )
            return X_train, X_test, y_train, y_test
        else:
            # No split; use the entire dataset for training and testing
            return X, X, y, y

    def _evaluate_metrics(self, X, y, data_type="train"):
        """
        Evaluates the specified metrics on the given dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The true target values.

        data_type : {'train', 'test'}, default='train'
            Specifies whether the data is from training or testing.

        Returns
        -------
        metrics_results : dict
            A dictionary containing the evaluated metrics.

        Notes
        -----
        If a metric function expects an additional argument (e.g., `average`
        for multi-class metrics), it is provided as ``'weighted'``.

        Any exceptions during metric computation are caught, and the metric
        value is set to ``None``.
        """
        metrics_results = {}
        y_pred = self.model.predict(X)
        for metric_name, metric_func in self.metrics.items():
            try:
                # Check if the metric function requires 'average' parameter
                if 'average' in metric_func.__code__.co_varnames:
                    score = metric_func(y, y_pred, average='weighted')
                else:
                    score = metric_func(y, y_pred)
                metrics_results[metric_name] = score
            except Exception as e:
                self.log_event(f"metric_error_{metric_name}", {
                    "error": str(e), "data_type": data_type
                })
                metrics_results[metric_name] = None

        self.log_event(f"{data_type}_metrics_evaluation", metrics_results)
        return metrics_results

    def _cross_validate(self, X, y):
        """
        Performs cross-validation on the training data and returns the
        cross-validation scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        scores : array
            Cross-validation scores for each fold.

        Notes
        -----
        Cross-validation is performed using the specified number of folds
        (`cv_folds`). The cross-validation results are stored in the
        `results_` attribute under the key ``'cross_val_scores'``.

        Exceptions during cross-validation are logged, and ``None`` is
        returned.
        """
        try:
            scores = cross_val_score(
                self.model, X, y,
                cv=self.cv_folds,
                n_jobs=-1 if self.parallel_execution else 1
            )
            self.log_event('cross_validation', {"cv_scores": scores})
            return scores
        except Exception as e:
            self.log_event("cross_val_error", {"error": str(e)})
            return None


@smartFitRun
class OverfittingDetection(BaseTest):
    """
    Detects overfitting in a machine learning model by comparing the model's
    performance on training and test data. Supports cross-validation,
    configurable tolerance for detecting overfitting, and logging of results.

    The detection is based on the comparison of training and test scores.
    If the difference between the training score and the test score exceeds
    the specified tolerance, overfitting is considered to be detected.

    .. math::
        \\text{Overfitting} = (S_{\\text{train}} - S_{\\text{test}}) > \\text{tolerance}

    where :math:`S_{\\text{train}}` is the score on the training set,
    :math:`S_{\\text{test}}` is the score on the test set, and `tolerance`
    is a user-defined threshold.

    Parameters
    ----------
    model : estimator object
        The machine learning model to be evaluated for overfitting.
        The model should follow the scikit-learn API, having ``fit`` and
        ``score`` methods.

    tolerance : float, default=0.05
        The acceptable difference between training and test scores before
        overfitting is detected.

    cross_validation : bool, default=False
        If ``True``, cross-validation will be used to evaluate the model.

    cv_folds : int, default=5
        The number of cross-validation folds if `cross_validation` is enabled.

    evaluation_split : float, default=0.2
        The proportion of the data used for testing. If set to ``1.0``,
        no data splitting is performed.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution for cross-validation.

    store_results : bool, default=True
        If ``True``, the overfitting results will be stored.

    enable_logging : bool, default=True
        If ``True``, logs the events related to overfitting detection.

    **config_params : dict
        Additional configuration parameters passed to the test for customization.

    Attributes
    ----------
    results_ : dict
        Stores the results of the overfitting detection process, including
        training and test scores, and whether overfitting is detected.

    Methods
    -------
    fit(X, y, X_test=None, y_test=None, **fit_params)
        Fits the model to the training data and evaluates overfitting by
        comparing train and test scores.

    cross_validate(X, y)
        Performs cross-validation and returns the cross-validation scores.

    save_overfitting_report(path)
        Saves the overfitting report to the specified file.

    Notes
    -----
    - This class assumes the model follows the scikit-learn API, with ``fit``
      and ``score`` methods.
    - Overfitting is detected if the difference between the training and
      test scores exceeds the specified tolerance.

    Examples
    --------
    >>> from gofast.mlops.testing import OverfittingDetection
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>> model = RandomForestClassifier(random_state=42)
    >>> detector = OverfittingDetection(model=model, tolerance=0.05)
    >>> detector.fit(X, y)
    >>> print(detector.results_)

    See Also
    --------
    sklearn.model_selection.cross_val_score : Utility function to perform cross-validation.
    sklearn.metrics : Module containing various scoring functions for models.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009). "The Elements of
       Statistical Learning," Springer Series in Statistics.
    """

    @validate_params({
        'model': [HasMethods(['fit', 'score'])],
        'tolerance': [Interval(Real, 0, None, closed='left')],
        'cross_validation': [bool],
        'cv_folds': [int],
        'evaluation_split': [Interval(Real, 0, 1, closed='right')],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'enable_logging': [bool],
    })
    def __init__(
        self,
        model,
        tolerance: float = 0.05,
        cross_validation: bool = False,
        cv_folds: int = 5,
        evaluation_split: float = 0.2,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        enable_logging: bool = True,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            store_results=store_results,
            enable_logging=enable_logging,
            parallel_execution=parallel_execution,
            random_seed=random_seed,
            **config_params
        )
        self.model = model
        self.tolerance = tolerance
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.evaluation_split = evaluation_split
        self.results_ = None  # Initialize results_ attribute

    def fit(self, X, y, X_test=None, y_test=None, **fit_params):
        """
        Fits the model and detects overfitting by comparing the training
        and test scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The training target values.

        X_test : array-like of shape (n_samples_test, n_features), optional
            The test input samples. If not provided, data is split internally.

        y_test : array-like of shape (n_samples_test,), optional
            The test target values. If not provided, data is split internally.

        **fit_params : dict
            Additional parameters passed to the model's ``fit`` method.

        Returns
        -------
        self : OverfittingDetection
            Returns the instance itself.

        Raises
        ------
        ValueError
            If `X_test` is provided without `y_test`, or vice versa.

        Notes
        -----
        - If `X_test` and `y_test` are not provided, the data will be split
          internally using `evaluation_split`.
        - If `cross_validation` is enabled, cross-validation is performed
          instead of the standard train-test split.
        """
        if X is None or y is None:
            raise ValueError("Both X and y are required to fit the model.")

        if X_test is not None and y_test is None:
            raise ValueError("y_test must be provided if X_test is supplied.")
        if y_test is not None and X_test is None:
            raise ValueError("X_test must be provided if y_test is supplied.")

        X, y = check_X_y(X, y, multi_output=True)

        if self.cross_validation:
            # Perform cross-validation
            self.results_ = self._cross_validate(X, y)
        else:
            if X_test is not None and y_test is not None:
                X_test, y_test = check_X_y(X_test, y_test, multi_output=True)
                X_train, y_train = X, y
            else:
                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = self._split_data(X, y)

            # Fit the model
            self.model.fit(X_train, y_train, **fit_params)
            self._set_fitted()

            # Detect overfitting
            self.results_ = self._detect_overfitting(X_train, y_train, X_test, y_test)

        return self

    def _split_data(self, X, y):
        """
        Splits the input data into training and testing sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of arrays
            The split data for training and testing.

        Notes
        -----
        If `evaluation_split` is set to 1.0, no splitting is performed,
        and the entire dataset is used for both training and testing.
        """
        if self.evaluation_split < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.evaluation_split, random_state=self.random_seed
            )
        else:
            # No split; use the entire dataset for both training and testing
            X_train, X_test, y_train, y_test = X, X, y, y

        return X_train, X_test, y_train, y_test

    def _detect_overfitting(self, X_train, y_train, X_test, y_test):
        """
        Detects overfitting by comparing training and test scores.

        Parameters
        ----------
        X_train : array-like of shape (n_samples_train, n_features)
            The training input samples.

        y_train : array-like of shape (n_samples_train,)
            The training target values.

        X_test : array-like of shape (n_samples_test, n_features)
            The test input samples.

        y_test : array-like of shape (n_samples_test,)
            The test target values.

        Returns
        -------
        results : dict
            A dictionary containing training and test scores, and a boolean
            indicating whether overfitting is detected.

        Notes
        -----
        Overfitting is detected if the difference between the training and
        test scores exceeds the specified tolerance.
        """
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        overfitting_detected = (train_score - test_score) > self.tolerance
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'overfitting_detected': overfitting_detected
        }

        self.log_event("overfitting_detection", results)
        return results

    def _cross_validate(self, X, y):
        """
        Performs cross-validation on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        results : dict
            A dictionary containing the cross-validation scores and their mean.

        Notes
        -----
        Cross-validation is performed using the specified number of folds (`cv_folds`).
        """
        scores = cross_val_score(
            self.model, X, y,
            cv=self.cv_folds,
            n_jobs=-1 if self.parallel_execution else 1
        )
        mean_score = scores.mean()
        results = {
            'cross_val_scores': scores,
            'mean_cross_val_score': mean_score
        }

        self.log_event("cross_validation", results)
        return results

    def save_overfitting_report(self, path: str):
        """
        Saves the overfitting report to a specified file path.

        Parameters
        ----------
        path : str
            The file path where the report will be saved.

        Raises
        ------
        ValueError
            If there is no overfitting report available or result storage
            is disabled.

        Notes
        -----
        The report contains information about training and test scores,
        and whether overfitting was detected.
        """
        if self.results_ is None:
            raise ValueError(
                "No overfitting report available. Ensure the model has been "
                "fitted and evaluated."
            )
        if not self.store_results:
            raise ValueError("Result storage is disabled.")

        with open(path, 'w') as file:
            file.write(str(self.results_))

        self.log_event("report_saved", {"path": path})


@smartFitRun
class DataIntegrity(BaseTest):
    """
    Performs various checks on a dataset to ensure it meets specified quality
    standards. This includes checks for missing values, duplicates, data types,
    value ranges, and custom validation functions. The class supports logging
    and parallel execution for large datasets.

    The integrity of the data is evaluated based on statistical checks and
    user-defined validation functions.

    .. math::
        \\text{Missing Ratio} = \\frac{\\text{Total Missing Values}}{\\text{Total Elements in Data}}

    Parameters
    ----------
    validation_checks : dict of str to callable, default=None
        A dictionary of custom validation check names and their corresponding
        functions that accept a DataFrame and return a boolean indicating
        success or failure.

    missing_value_threshold : float, default=0.0
        The acceptable threshold for the percentage of missing values in the
        dataset. Must be between 0.0 and 1.0.

    unique_check_columns : list of str, default=None
        A list of column names to check for uniqueness. If duplicates are found
        in these columns and `allow_duplicates` is ``False``, an issue will be
        raised.

    allow_duplicates : bool, default=False
        If ``True``, duplicate rows in the `unique_check_columns` will be
        allowed. If ``False``, duplicates will be flagged.

    data_types : dict of str to type, default=None
        A dictionary where keys are column names and values are the expected
        data types for those columns.

    range_checks : dict of str to tuple of (float, float), default=None
        A dictionary where keys are column names and values are tuples
        specifying the acceptable range (min, max) for values in those columns.

    enable_logging : bool, default=True
        If ``True``, logs events related to data integrity checks.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution for large datasets.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    **config_params : dict
        Additional configuration parameters for the test.

    Attributes
    ----------
    issues_ : list
        A list containing issues found during the data integrity check.

    Methods
    -------
    run(data, **run_kwargs)
        Executes the data integrity check on the provided dataset.

    save_issues(path)
        Saves the identified issues to a specified file.

    Notes
    -----
    - The integrity checks can be customized through `validation_checks`,
      allowing users to define their own data validation logic.
    - Range checks are performed on specified columns to ensure values fall
      within the provided bounds.

    Examples
    --------
    >>> from gofast.mlops.testing import DataIntegrity
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({
    ...     'age': [25, 30, 22, 29, None],
    ...     'income': [50000, 60000, 40000, None, 45000],
    ...     'country': ['US', 'UK', 'US', 'UK', 'UK']
    ... })
    >>>
    >>> integrity_test = DataIntegrity(
    ...     missing_value_threshold=0.05,
    ...     unique_check_columns=['age'],
    ...     data_types={'age': float, 'income': float},
    ...     range_checks={'age': (18, 65), 'income': (0, 100000)}
    ... )
    >>>
    >>> issues = integrity_test.run(data)
    >>> print(issues)

    See Also
    --------
    pandas.DataFrame.isnull : Check for missing values in a DataFrame.
    pandas.DataFrame.duplicated : Identify duplicate rows in a DataFrame.

    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
       in Python," Proceedings of the 9th Python in Science Conference.
    """

    @validate_params({
        'validation_checks': [dict, None],
        'missing_value_threshold': [Interval(Real, 0.0, 1.0, closed='both')],
        'unique_check_columns': [list, None],
        'allow_duplicates': [bool],
        'data_types': [dict, None],
        'range_checks': [dict, None],
        'enable_logging': [bool],
        'parallel_execution': [bool],
        'random_seed': [int, None],
    })
    def __init__(
        self,
        validation_checks: Optional[Dict[str, Callable]] = None,
        missing_value_threshold: float = 0.0,
        unique_check_columns: Optional[List[str]] = None,
        allow_duplicates: bool = False,
        data_types: Optional[Dict[str, type]] = None,
        range_checks: Optional[Dict[str, Tuple[float, float]]] = None,
        enable_logging: bool = True,
        parallel_execution: bool = False,
        random_seed: Optional[int] = None,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            store_results=True,
            enable_logging=enable_logging,
            parallel_execution=parallel_execution,
            random_seed=random_seed,
            **config_params
        )
        self.validation_checks = validation_checks or {}
        self.missing_value_threshold = missing_value_threshold
        self.unique_check_columns = unique_check_columns or []
        self.allow_duplicates = allow_duplicates
        self.data_types = data_types or {}
        self.range_checks = range_checks or {}
        self.issues_ = []  


    @RunReturn(attribute_name='issues_')
    def run(self, data, **run_kwargs):
        """
        Executes the data integrity checks on the provided dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data on which the integrity checks are performed.

        **run_kwargs : dict
            Additional arguments that may influence the execution of the test.

        Returns
        -------
        issues_ : list
            A list of issues identified during the data integrity check.
            If no issues are found, a success message is returned.

        Notes
        -----
        The following checks are performed:

        - Missing values are checked against the `missing_value_threshold`.
        - Duplicates are identified based on `unique_check_columns`.
        - Data types of columns are verified against `data_types`.
        - Values in columns are checked for falling within specified
          `range_checks`.
        - Custom validation checks are executed, if provided.

        Raises
        ------
        ValueError
            If any of the data validation checks fail.

        See Also
        --------
        pandas.DataFrame.isnull : Check for missing values in a DataFrame.
        pandas.DataFrame.duplicated : Identify duplicate rows in a DataFrame.
        """
        
        issues = []

        # Missing values check
        if self.missing_value_threshold > 0:
            total_missing = data.isnull().sum().sum()
            missing_ratio = total_missing / data.size
            if missing_ratio > self.missing_value_threshold:
                issues.append(
                    f"Missing values exceed threshold: {missing_ratio:.2%}"
                )

        # Duplicates check
        if not self.allow_duplicates and self.unique_check_columns:
            duplicates = data.duplicated(subset=self.unique_check_columns).sum()
            if duplicates > 0:
                issues.append(
                    f"Duplicate rows found in columns {self.unique_check_columns}: "
                    f"{duplicates} duplicates."
                )

        # Data type checks
        for column, expected_type in self.data_types.items():
            if column in data.columns:
                # Check data types, allowing for NaN values
                is_correct_type = data[column].apply(
                    lambda x: isinstance(x, expected_type) or pd.isnull(x)
                ).all()
                if not is_correct_type:
                    issues.append(
                        f"Data type mismatch in column '{column}'. Expected {expected_type}."
                    )

        # Range checks
        for column, (min_val, max_val) in self.range_checks.items():
            if column in data.columns:
                # Exclude NaN values from range check
                non_na_values = data[column].dropna()
                out_of_range = ((non_na_values < min_val) | (non_na_values > max_val)).sum()
                if out_of_range > 0:
                    issues.append(
                        f"Values out of range in column '{column}': "
                        f"{out_of_range} values outside [{min_val}, {max_val}]."
                    )

        # Custom validation checks
        for check_name, check_func in self.validation_checks.items():
            if not check_func(data):
                issues.append(f"{check_name} failed.")

        # If no issues were found
        if not issues:
            issues.append("Data integrity check passed successfully.")

        # Log the test results if logging is enabled
        self.log_event("data_integrity_test", {"issues": issues})

        # Set the test as run
        self._set_runned()

        # Set the issues_ attribute
        self.issues_ = issues

    def save_issues(self, path: str):
        """
        Saves the identified issues to a specified file.

        Parameters
        ----------
        path : str
            The file path where the issues will be saved.

        Raises
        ------
        ValueError
            If there are no issues to save or result storage is disabled.

        Notes
        -----
        The issues are saved as a text file containing the list of issues.
        """
        # Check if the run method has been executed
        check_is_runned(
            self, msg="The run method must be executed before saving issues.")

        if not self.store_results:
            raise ValueError("Result storage is disabled.")

        if not self.issues_:
            raise ValueError("No issues to save.")

        with open(path, 'w') as file:
            for issue in self.issues_:
                file.write(f"{issue}\n")

        self.log_event("issues_saved", {"path": path})


@smartFitRun
class BiasDetection(BaseTest):
    """
    Assesses the fairness of a machine learning model with respect to a specified
    sensitive feature. It measures bias using a fairness metric and compares the
    resulting bias score against a threshold to determine whether the model
    exhibits bias.

    The bias is calculated by comparing the model's predictions across different
    groups defined by the sensitive feature. This process can be carried out
    using a single fairness metric or multiple metrics if specified.

    .. math::
        \\text{Bias} = \\frac{1}{G} \\sum_{g=1}^{G} M(P_g, y_g)

    where :math:`M` is the fairness metric applied to the predicted outcomes
    :math:`P_g` and true labels :math:`y_g` of group :math:`g`, and :math:`G`
    is the total number of groups based on the sensitive feature.

    Parameters
    ----------
    model : estimator object
        The machine learning model to be evaluated for bias. The model should
        follow the scikit-learn API, having ``fit`` and ``predict`` methods.

    sensitive_feature : str
        The column name of the feature in the data that represents a sensitive
        attribute (e.g., gender, race) used to detect bias.

    fairness_metric : callable or dict of str to callable
        A fairness metric or a dictionary of fairness metrics that measure bias.
        The function(s) should accept predictions, true labels, and the sensitive
        feature values as input and return a bias score.

    bias_threshold : float, default=0.1
        The threshold above which bias is considered to be detected.

    multi_metric : bool, default=False
        If ``True``, multiple fairness metrics are used for bias detection.

    log_results : bool, default=True
        If ``True``, logs the results of the bias detection.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution where applicable.

    store_results : bool, default=True
        If ``True``, stores the results of the bias detection.

    **config_params : dict
        Additional configuration parameters.

    Attributes
    ----------
    results_ : dict
        Stores the results of the bias detection, including bias scores and
        whether bias was detected.

    Methods
    -------
    fit(X, y, **fit_params)
        Fits the model to the provided training data.

    detect_bias(X, y)
        Detects bias in the model based on the specified fairness metric(s)
        and the sensitive feature.

    fit_and_detect_bias(X, y, **fit_params)
        Combines ``fit`` and ``detect_bias`` into one method to fit the model
        and detect bias in one call.

    Notes
    -----
    - The fairness metric(s) should be designed to handle both binary and
      multi-class classification problems.
    - If ``multi_metric`` is enabled, bias detection will aggregate the results
      from multiple fairness metrics and calculate an average bias score.
    - The input data ``X`` must be a pandas DataFrame when using a sensitive
      feature, as the feature is accessed by column name.

    Examples
    --------
    >>> from gofast.mlops.testing import BiasDetection
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>> import pandas as pd
    >>>
    >>> def fairness_metric(predictions, y, sensitive_values):
    ...     # Example fairness metric based on accuracy difference
    ...     group_1 = sensitive_values == 1
    ...     group_2 = sensitive_values == 0
    ...     acc_group_1 = accuracy_score(y[group_1], predictions[group_1])
    ...     acc_group_2 = accuracy_score(y[group_2], predictions[group_2])
    ...     return abs(acc_group_1 - acc_group_2)
    >>>
    >>> model = RandomForestClassifier()
    >>> bias_detector = BiasDetection(
    ...     model=model,
    ...     sensitive_feature='gender',
    ...     fairness_metric=fairness_metric,
    ...     bias_threshold=0.05
    ... )
    >>> X_train = pd.DataFrame({'feature1': [0, 1], 'gender': [0, 1]})
    >>> y_train = [0, 1]
    >>> bias_detector.fit(X_train, y_train)
    >>> bias_detector.detect_bias(X_train, y_train)
    >>> print(bias_detector.results_)

    See Also
    --------
    sklearn.base.BaseEstimator : Base class for all estimators in scikit-learn.
    sklearn.metrics : Common evaluation metrics, which may serve as fairness metrics.

    References
    ----------
    .. [1] Barocas, S., Hardt, M., Narayanan, A. (2019). "Fairness and Machine Learning,"
       fairmlbook.org, available at https://fairmlbook.org.
    """

    @validate_params({
        'model': [HasMethods(['fit', 'predict'])],
        'sensitive_feature': [str],
        'fairness_metric': [callable, dict],
        'bias_threshold': [Interval(Real, 0, None, closed='left')],
        'multi_metric': [bool],
        'log_results': [bool],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
    })
    def __init__(
        self,
        model,
        sensitive_feature: str,
        fairness_metric: Union[Callable, Dict[str, Callable]],
        bias_threshold: float = 0.1,
        multi_metric: bool = False,
        log_results: bool = True,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=log_results,
            **config_params
        )
        self.model = model
        self.sensitive_feature = sensitive_feature
        self.fairness_metric = fairness_metric
        self.bias_threshold = bias_threshold
        self.multi_metric = multi_metric
        self.log_results = log_results
        self.results_ = None  # Initialize results_ attribute

    def fit(self, X, y, **fit_params):
        """
        Fits the machine learning model to the provided training data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The training data. Must be a DataFrame containing the sensitive feature.

        y : array-like of shape (n_samples,)
            The target labels.

        **fit_params : dict
            Additional parameters passed to the model's ``fit`` method.

        Returns
        -------
        self : BiasDetection
            Returns the instance itself.

        Notes
        -----
        This method ensures that the model is trained on the provided data
        before bias detection can occur.

        Raises
        ------
        ValueError
            If the sensitive feature is not present in ``X``.
        """
        # Check if sensitive_feature is in X
        if self.sensitive_feature not in X.columns:
            raise ValueError(
                f"The sensitive feature '{self.sensitive_feature}' is not present in X."
            )

        X_values, y = check_X_y(X, y)

        # Fit the model with the data
        self.model.fit(X_values, y, **fit_params)
        self._set_fitted()
        return self

    def detect_bias(self, X, y):
        """
        Detects bias in the trained model using the specified fairness metric(s).

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data used to detect bias.

        y : array-like of shape (n_samples,)
            The true labels for the input data.

        Returns
        -------
        self : BiasDetection
            Returns the instance itself.

        Notes
        -----
        - This method requires that the model is already fitted on the data.
        - The bias is calculated based on predictions for different groups
          defined by the sensitive feature.

        Raises
        ------
        ValueError
            If the model has not been fitted before calling this method.
        ValueError
            If the sensitive feature is not present in ``X``.
        """
        self.check_is_fitted()

        # Check if sensitive_feature is in X
        if self.sensitive_feature not in X.columns:
            raise ValueError(
                f"The sensitive feature '{self.sensitive_feature}' is not present in X."
            )

        # Run predictions
        predictions = self.model.predict(X)

        # Extract sensitive feature values
        sensitive_values = X[self.sensitive_feature]

        # Calculate the bias score using the fairness metric(s)
        if self.multi_metric:
            bias_scores = {}
            for metric_name, metric_func in self.fairness_metric.items():
                bias_scores[metric_name] = metric_func(
                    predictions, y, sensitive_values
                )
            overall_bias = sum(bias_scores.values()) / len(bias_scores)
        else:
            bias_score = self.fairness_metric(predictions, y, sensitive_values)
            bias_scores = {'fairness_metric': bias_score}
            overall_bias = bias_score

        # Log event if enabled
        if self.log_results:
            self.log_event('bias_detection', {
                'bias_scores': bias_scores,
                'sensitive_feature': self.sensitive_feature
            })

        # Determine if the model exceeds the bias threshold
        is_biased = overall_bias > self.bias_threshold

        # Store results in results_ attribute
        self.results_ = {
            'bias_scores': bias_scores,
            'sensitive_feature': self.sensitive_feature,
            'overall_bias': overall_bias,
            'is_biased': is_biased,
            'bias_threshold': self.bias_threshold
        }

        return self

    def fit_and_detect_bias(self, X, y, **fit_params):
        """
        Fits the model and then detects bias in a single step.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data. Must contain the sensitive feature.

        y : array-like of shape (n_samples,)
            The true labels for the input data.

        **fit_params : dict
            Additional parameters passed to the model's ``fit`` method.

        Returns
        -------
        self : BiasDetection
            Returns the instance itself.

        Notes
        -----
        This method combines the model training (``fit``) and bias detection
        (``detect_bias``) processes in one step for convenience.
        """
        self.fit(X, y, **fit_params)
        self.detect_bias(X, y)
        return self

@smartFitRun
class ModelVersionCompliance(BaseTest):
    """
    Provides a robust mechanism for validating the versioning of a machine
    learning model. It checks whether the current model version matches the
    expected version, supports minor version mismatches, detects deprecated
    versions, and applies custom version policies.

    The model versioning system is designed to ensure compatibility and warn
    against deprecated versions or unsupported versions based on defined policies.

    .. math::
        \\text{version\\_match} =
        \\begin{cases}
        \\text{True} & \\text{if major.minor match} \\\\
        \\text{False} & \\text{otherwise}
        \\end{cases}

    Parameters
    ----------
    expected_version : str
        The expected version of the model (e.g., '1.0.0').

    allow_minor_version_mismatch : bool, default=False
        If ``True``, allows minor version mismatches between the model and
        the expected version (e.g., '1.0.x').

    check_deprecation : bool, default=False
        If ``True``, checks whether the model's version is deprecated.

    deprecated_versions : list of str or None, default=None
        A list of deprecated model versions.

    custom_policies : dict or None, default=None
        A dictionary of custom versioning policies, including version ranges,
        pre-release version deprecation, and minimum supported versions.

    api_url : str or None, default="https://api.example.com/deprecated-versions"
        The URL of an external service to fetch deprecated versions.

    config_file_path : str or None, default="config/deprecated_versions.json"
        Path to a local configuration file containing deprecated versions.

    fallback_versions : list of str or None, default=['1.0.0', '1.1.0', '2.0.0-beta']
        A hardcoded list of fallback deprecated versions to use if external
        services or configuration files are unavailable.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution where applicable.

    store_results : bool, default=True
        If ``True``, stores the results of the versioning checks.

    enable_logging : bool, default=True
        If ``True``, logs events related to versioning checks.

    **config_params : dict
        Additional configuration parameters for custom policies or logging.

    Attributes
    ----------
    results_ : dict
        Stores the results of the versioning checks, including whether the
        version matched the expected version, whether it is deprecated, and
        the deprecation source.

    Methods
    -------
    run(model, **run_kwargs)
        Performs the versioning check by comparing the current model version
        with the expected version, checking for deprecation, and logging results.

    Notes
    -----
    - This class assumes the model object has a ``get_version`` method that
      returns the model's version as a string.
    - Custom policies can include minimum supported versions, deprecated
      versions, version ranges, and pre-release version deprecation.

    Examples
    --------
    >>> from gofast.mlops.testing import ModelVersionCompliance
    >>> model = SomeModelClass()
    >>> version_check = ModelVersionCompliance(expected_version='2.0.0')
    >>> version_check.run(model)
    >>> print(version_check.results_)

    See Also
    --------
    packaging.version.Version : Provides version parsing and comparison utilities.

    References
    ----------
    .. [1] Semantic Versioning (https://semver.org/).
    """

    @validate_params({
        'expected_version': [str],
        'allow_minor_version_mismatch': [bool],
        'check_deprecation': [bool],
        'deprecated_versions': [list, None],
        'custom_policies': [dict, None],
        'api_url': [str, None],
        'config_file_path': [str, None],
        'fallback_versions': [list, None],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'enable_logging': [bool],
    })
    def __init__(
        self,
        expected_version: str,
        allow_minor_version_mismatch: bool = False,
        check_deprecation: bool = False,
        deprecated_versions: Optional[List[str]] = None,
        custom_policies: Optional[Dict[str, Any]] = None,
        api_url: Optional[str] = None,
        config_file_path: Optional[str] = None,
        fallback_versions: Optional[List[str]] = None,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        enable_logging: bool = True,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=enable_logging,
            **config_params
        )
        self.expected_version = expected_version
        self.allow_minor_version_mismatch = allow_minor_version_mismatch
        self.check_deprecation = check_deprecation
        self.deprecated_versions = deprecated_versions or []
        self.custom_policies = custom_policies or {}
        self.api_url = api_url or "https://api.example.com/deprecated-versions"
        self.config_file_path = config_file_path or "config/deprecated_versions.json"
        self.fallback_versions = fallback_versions or ['1.0.0', '1.1.0', '2.0.0-beta']
        
        self.results_ = None  
        self.deprecation_source_ = None  

    @RunReturn(attribute_name='results_')
    def run(self, model, **run_kwargs):
        """
        Runs the model versioning check by comparing the current model version
        with the expected version, checking for deprecation, and logging results.

        Parameters
        ----------
        model : object
            The model object that contains a ``get_version`` method to return
            the model's version.

        **run_kwargs : dict
            Additional arguments for running the versioning check.

        Returns
        -------
        results_ : dict
            A dictionary containing the results of the version check, including
            whether the version matched the expected version, whether it is
            deprecated, and the deprecation source.

        Notes
        -----
        - This method sets the ``results_`` attribute with the version check results.
        - It requires that the model object has a ``get_version`` method.

        Raises
        ------
        ValueError
            If the model does not have a ``get_version`` method.
        """
        if not hasattr(model, 'get_version'):
            raise ValueError("The model object must have a 'get_version' method.")

        current_version = model.get_version()

        if self.allow_minor_version_mismatch:
            version_match = self._check_minor_version(current_version)
        else:
            version_match = current_version == self.expected_version

        is_deprecated = False

        if self.check_deprecation:
            is_deprecated = self._check_deprecation(current_version)
            deprecation_source = self.deprecation_source_
        else:
            deprecation_source = None

        if self.enable_logging:
            self.log_event('model_versioning_check', {
                'current_version': current_version,
                'expected_version': self.expected_version,
                'version_match': version_match,
                'is_deprecated': is_deprecated,
                'deprecation_source': deprecation_source,
            })

        self._set_runned()

        self.results_ = {
            'current_version': current_version,
            'expected_version': self.expected_version,
            'version_match': version_match,
            'is_deprecated': is_deprecated,
            'deprecation_source': deprecation_source,
            'allow_minor_version_mismatch': self.allow_minor_version_mismatch
        }

    def _check_minor_version(self, current_version: str) -> bool:
        """
        Checks whether the major and minor versions match between the current
        version and the expected version, allowing for minor version mismatches
        if configured.

        Parameters
        ----------
        current_version : str
            The current version of the model.

        Returns
        -------
        bool
            ``True`` if the major and minor versions match, ``False`` otherwise.

        Examples
        --------
        >>> self.expected_version = "1.2.0"
        >>> current_version = "1.2.5"
        >>> self._check_minor_version(current_version)
        True
        """

        expected_ver = Version(self.expected_version)
        current_ver = Version(current_version)

        # Compare major and minor versions
        return (expected_ver.major == current_ver.major and
                expected_ver.minor == current_ver.minor)

    def _check_deprecation(self, current_version: str) -> bool:
        """
        Checks whether the current model version is deprecated based on a list
        of deprecated versions, external sources, or custom policies.

        Parameters
        ----------
        current_version : str
            The current version of the model.

        Returns
        -------
        bool
            ``True`` if the current version is deprecated, ``False`` otherwise.

        Notes
        -----
        - The method sets the ``deprecation_source_`` attribute indicating the
          source of deprecation information.
        """
        deprecated_versions = self.deprecated_versions or self._fetch_deprecated_versions()

        # Check for exact match in deprecated versions
        if current_version in deprecated_versions:
            self.deprecation_source_ = 'direct_deprecation_list'
            return True

        # Apply custom deprecation policies
        if self._check_custom_deprecation_policies(current_version):
            self.deprecation_source_ = 'custom_policy'
            return True

        return False

    def _check_custom_deprecation_policies(self, current_version: str) -> bool:
        """
        Applies custom deprecation policies, such as minimum supported versions,
        version ranges, or pre-release version deprecation.

        Parameters
        ----------
        current_version : str
            The current version of the model.

        Returns
        -------
        bool
            ``True`` if the current version is deprecated according to custom
            policies, ``False`` otherwise.

        Examples
        --------
        >>> self.custom_policies = {
        ...     "min_supported_version": "1.5.0",
        ...     "deprecated_versions": ["2.0.0"],
        ...     "deprecated_version_ranges": [
        ...         {"start": "1.0.0", "end": "1.5.0"}
        ...     ]
        ... }
        >>> self._check_custom_deprecation_policies("1.4.0")
        True
        """
        
        current_ver = Version(current_version)

        # Policy 1: Deprecate versions below a minimum supported version
        min_supported_version = self.custom_policies.get("min_supported_version")
        if min_supported_version:
            min_supported_ver = Version(min_supported_version)
            if current_ver < min_supported_ver:
                return True

        # Policy 2: Deprecate specific versions
        deprecated_versions = self.custom_policies.get("deprecated_versions", [])
        if current_version in deprecated_versions:
            return True

        # Policy 3: Deprecate version ranges
        version_ranges = self.custom_policies.get("deprecated_version_ranges", [])
        for vrange in version_ranges:
            start_ver = Version(vrange["start"])
            end_ver = Version(vrange["end"])
            if start_ver <= current_ver <= end_ver:
                return True

        # Policy 4: Deprecate pre-release versions
        deprecate_pre_releases = self.custom_policies.get("deprecate_pre_releases", False)
        if deprecate_pre_releases and current_ver.is_prerelease:
            return True

        return False

    def _fetch_deprecated_versions(self) -> List[str]:
        """
        Fetches the list of deprecated versions from external services,
        configuration files, or fallback lists.

        Returns
        -------
        list of str
            A list of deprecated versions.

        Notes
        -----
        - This method tries to fetch from external services first, then from
          configuration files, and finally falls back to a hardcoded list.
        """
        try:
            external_versions = self._fetch_from_external_service()
            if external_versions:
                return external_versions
        except Exception as e:
            if self.enable_logging:
                self.log_event('external_service_error', {'error': str(e)})

        try:
            config_versions = self._fetch_from_config_file()
            if config_versions:
                return config_versions
        except FileNotFoundError:
            if self.enable_logging:
                self.log_event('config_file_error', {'error': 'Config file not found'})

        if self.enable_logging:
            self.log_event('fallback_to_static_list', {'versions': self.fallback_versions})

        return self.fallback_versions

    @ensure_pkg(
        "requests",
        extra="ModelVersionCompliance requires the 'requests' package.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _fetch_from_external_service(self) -> List[str]:
        """
        Fetches deprecated versions from an external API service.

        Returns
        -------
        list of str
            A list of deprecated versions obtained from the external service.

        Notes
        -----
        - The external service is expected to return a JSON response with a
          'deprecated_versions' field.

        Raises
        ------
        Exception
            If the external service call fails.
        """
        import requests
        response = requests.get(self.api_url)
        if response.status_code == 200:
            data = response.json()
            return data.get('deprecated_versions', [])
        else:
            raise Exception(
                f"Failed to fetch from external service: {response.status_code}")


    def _fetch_from_config_file(self) -> List[str]:
        """
        Loads deprecated versions from a local configuration file.

        Returns
        -------
        list of str
            A list of deprecated versions read from the configuration file.

        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        """
        
        with open(self.config_file_path, 'r') as f:
            config_data = json.load(f)
        return config_data.get('deprecated_versions', [])


@smartFitRun 
class PerformanceRegression(BaseTest):
    """
    Evaluates and detects performance regressions in a machine learning model
    by comparing it against a baseline model. It uses specified metrics to
    evaluate performance and detects if the current model's performance has
    fallen below an acceptable threshold compared to the baseline.

    The primary comparison is between the model's performance and the baseline
    model's performance. If a performance drop greater than the threshold is
    detected, the system flags the issue as a regression.

    .. math::
        \\text{regression} = (S_{\\text{model}} < S_{\\text{baseline}} - \\text{threshold})

    where :math:`S_{\\text{model}}` is the performance score of the current
    model, :math:`S_{\\text{baseline}}` is the score of the baseline model,
    and :math:`\\text{threshold}` is the acceptable difference before
    considering it a regression.

    Parameters
    ----------
    model : estimator object
        The machine learning model to evaluate for performance regression.
        The model should follow the scikit-learn API, having ``fit`` and
        ``predict`` methods.

    baseline_model : estimator object or None, default=None
        The baseline model to compare against the current model's performance.
        If ``None``, no comparison is made against a baseline.

    metrics : dict of str to callable or None, default=None
        A dictionary of metric names and corresponding callable functions to
        evaluate the model's performance. The functions should accept
        ``y_true`` and ``y_pred`` as inputs and return a float score. If
        ``None``, the default metric is the model's ``score`` method.

    threshold : float, default=0.05
        The acceptable difference between the current model's performance and
        the baseline model's performance before considering it a regression.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution where applicable.

    store_results : bool, default=True
        If ``True``, stores the results of the performance regression test.

    enable_logging : bool, default=True
        If ``True``, logs events related to the performance regression test.

    **config_params : dict
        Additional configuration parameters for test customization.

    Attributes
    ----------
    results_ : dict
        The results of the performance regression test, including whether
        regression was detected.

    Methods
    -------
    fit(X, y, **fit_params)
        Fits the current model (and baseline model if provided) to the data.

    evaluate(X, y)
        Evaluates the model's performance using custom metrics and checks for
        regression against the baseline model.

    compare_with_baseline(X, y)
        Compares the current model's performance with the baseline and prints
        whether regression is detected.

    Notes
    -----
    - This class is designed to work with models that follow the scikit-learn
      API.
    - If no baseline model is provided, the system will only evaluate the
      current model.

    Examples
    --------
    >>> from gofast.mlops.testing import PerformanceRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> model = RandomForestClassifier(random_state=42)
    >>> baseline_model = RandomForestClassifier(random_state=42)
    >>> regression_test = PerformanceRegression(
    ...     model=model,
    ...     baseline_model=baseline_model,
    ...     metrics={'accuracy': accuracy_score},
    ...     threshold=0.01
    ... )
    >>> X_train, y_train = ...
    >>> regression_test.fit(X_train, y_train)
    >>> regression_test.evaluate(X_train, y_train)
    >>> print(regression_test.results_)

    See Also
    --------
    sklearn.metrics : Common evaluation metrics such as ``accuracy_score``,
    ``precision_score``.

    References
    ----------
    .. [1] Kuhn, M., Johnson, K. (2013). "Applied Predictive Modeling,"
       Springer.
    """

    @validate_params({
        'model': [HasMethods(['fit', 'predict'])],
        'baseline_model': [HasMethods(['fit', 'predict']), None],
        'metrics': [dict, None],
        'threshold': [Interval(Real, 0, None, closed='left')],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'enable_logging': [bool],
    })
    def __init__(
        self,
        model,
        baseline_model=None,
        metrics: Optional[Dict[str, Callable]] = None,
        threshold: float = 0.05,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        enable_logging: bool = True,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=enable_logging,
            **config_params
        )
        self.model = model
        self.baseline_model = baseline_model
        self.metrics = metrics
        self.threshold = threshold
        self.results_ = None 

    def fit(self, X, y, **fit_params):
        """
        Fits the current model (and baseline model if provided) to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values for training.

        **fit_params : dict
            Additional parameters passed to the model's ``fit`` method.

        Returns
        -------
        self : PerformanceRegression
            Returns the instance itself.

        Notes
        -----
        - Both the current model and baseline model (if provided) are fitted
          to the same training data.
        """
        # Check the data for validity
        X, y = check_X_y(X, y)

        # Fit the current model
        self.model.fit(X, y, **fit_params)

        # Fit the baseline model if provided
        if self.baseline_model:
            self.baseline_model.fit(X, y, **fit_params)

        self._set_fitted()

        return self

    def evaluate(self, X, y):
        """
        Evaluates the model's performance using custom metrics and checks for
        regression against the baseline model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for evaluation.

        y : array-like of shape (n_samples,)
            The true target values for evaluation.

        Returns
        -------
        self : PerformanceRegression
            Returns the instance itself.

        Notes
        -----
        - This method checks if the performance of the current model is
          significantly worse than the baseline model based on the threshold.
        - The results are stored in the ``results_`` attribute.

        Raises
        ------
        ValueError
            If the model has not been fitted before running the evaluation.
        """
        # Ensure the model has been fitted
        check_is_fitted(self.model)
        if self.baseline_model:
            check_is_fitted(self.baseline_model)

        # Validate the input data
        X, y = check_X_y(X, y)

        # Get predictions from the model
        y_pred = self.model.predict(X)

        # Evaluate the model using the specified metrics
        if self.metrics is None:
            # Use the model's score method
            model_score = self.model.score(X, y)
            model_results = {'score': model_score}
        else:
            model_results = {}
            for name, metric_func in self.metrics.items():
                model_results[name] = metric_func(y, y_pred)

        baseline_results = {}
        regression_detected = False

        # If a baseline model is provided, compare its results
        if self.baseline_model:
            y_pred_baseline = self.baseline_model.predict(X)
            if self.metrics is None:
                baseline_score = self.baseline_model.score(X, y)
                baseline_results = {'score': baseline_score}
            else:
                baseline_results = {}
                for name, metric_func in self.metrics.items():
                    baseline_results[name] = metric_func(y, y_pred_baseline)

            # Check for regression
            regression_detected = any(
                (model_results[name] < baseline_results[name] - self.threshold)
                for name in model_results
            )

        # Log the results of the test
        if self.enable_logging:
            self.log_event('performance_regression_test', {
                'model_results': model_results,
                'baseline_results': baseline_results,
                'regression_detected': regression_detected
            })

        # Store the results
        self.results_ = {
            'model_results': model_results,
            'baseline_results': baseline_results,
            'regression_detected': regression_detected
        }

        return self

    def compare_with_baseline(self, X, y):
        """
        Compares the current model's performance with the baseline model
        (if provided).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for evaluation.

        y : array-like of shape (n_samples,)
            The true target values for evaluation.

        Returns
        -------
        self : PerformanceRegression
            Returns the instance itself.

        Notes
        -----
        - If no performance regression is detected, the method will print
          "No performance regression detected."
        - If a regression is detected, the method will print "Performance
          regression detected based on the threshold."

        Raises
        ------
        ValueError
            If the model has not been fitted before comparing with the baseline.
        """
        # Ensure the model has been fitted
        check_is_fitted(self.model)
        if self.baseline_model:
            check_is_fitted(self.baseline_model)

        # Evaluate the model if not already done
        if self.results_ is None:
            self.evaluate(X, y)

        if not self.results_['regression_detected']:
            print("No performance regression detected.")
        else:
            print("Performance regression detected based on the threshold.")

        return self
 
@smartFitRun 
class CIIntegration(BaseTest):
    """
    Provides an interface to integrate with a Continuous Integration (CI) tool.
    It allows the triggering of CI actions for a specific project with options
    for retries, response validation, timeouts, and logging.

    This class can handle retries and failures and provides mechanisms to log
    events and build detailed results for each CI action run.

    Parameters
    ----------
    ci_tool : object
        The CI tool object that exposes a ``trigger_action`` method and has a
        ``name`` attribute. This object should interact with the CI system
        (e.g., Jenkins, Travis, GitLab CI).

    project_name : str
        The name of the project in the CI tool for which the action will be
        triggered.

    trigger_action : str
        The action to be triggered on the CI tool (e.g., 'build', 'deploy').

    retry_attempts : int, default=3
        The number of retry attempts if the CI action fails.

    retry_delay : int, default=5
        The delay in seconds between retry attempts.

    timeout : int or None, default=None
        The maximum time to wait for the CI action to complete. If ``None``,
        the CI tool's default timeout is used.

    validate_response : bool, default=True
        If ``True``, the response from the CI tool is validated using a built-in
        validation method.

    random_seed : int or None, default=None
        The random seed for reproducibility.

    parallel_execution : bool, default=False
        If ``True``, enables parallel execution where applicable.

    store_results : bool, default=True
        If ``True``, stores the results of the CI action.

    enable_logging : bool, default=True
        If ``True``, logs the results of the CI action and retries.

    **config_params : dict
        Additional configuration parameters.

    Attributes
    ----------
    results_ : dict
        The results of the CI action, including success/failure status and any
        response details.

    Methods
    -------
    run(**run_kwargs)
        Executes the CI action with retries and logging. Handles retries on
        failure, validates responses, and logs the process.

    Notes
    -----
    - This class expects the CI tool to provide a ``trigger_action`` method
      that takes ``project_name`` and ``action`` as required arguments, and
      have a ``name`` attribute.
    - If retries are enabled, the action will be retried based on the configured
      delay and attempt limits.
    - Response validation can be customized by overriding the
      ``_validate_ci_response`` method.

    Examples
    --------
    >>> from gofast.mlops.testing import CIIntegration
    >>> ci_tool = Jenkins()  # Example CI tool object with 'trigger_action' method
    >>> ci_test = CIIntegration(
    ...     ci_tool=ci_tool,
    ...     project_name="MyProject",
    ...     trigger_action="build"
    ... )
    >>> ci_test.run()
    >>> print(ci_test.results_)

    See Also
    --------
    requests.get : For making actual HTTP requests if the CI tool uses a REST API.

    References
    ----------
    .. [1] Humble, J., Farley, D. (2010). "Continuous Delivery: Reliable Software
       Releases through Build, Test, and Deployment Automation," Addison-Wesley.
    """

    @validate_params({
        'ci_tool': [object],
        'project_name': [str],
        'trigger_action': [str],
        'retry_attempts': [Interval(Integral, 1, None, closed='left')],
        'retry_delay': [Interval(Integral, 0, None, closed='left')],
        'timeout': [Interval(Integral, 0, None, closed='left'), None],
        'validate_response': [bool],
        'random_seed': [int, None],
        'parallel_execution': [bool],
        'store_results': [bool],
        'enable_logging': [bool],
    })
    def __init__(
        self,
        ci_tool,
        project_name: str,
        trigger_action: str,
        retry_attempts: int = 3,
        retry_delay: int = 5,
        timeout: Optional[int] = None,
        validate_response: bool = True,
        random_seed: Optional[int] = None,
        parallel_execution: bool = False,
        store_results: bool = True,
        enable_logging: bool = True,
        **config_params
    ):
        super().__init__(
            test_name=self.__class__.__name__,
            random_seed=random_seed,
            parallel_execution=parallel_execution,
            store_results=store_results,
            enable_logging=enable_logging,
            **config_params
        )

        # Verify that ci_tool has required methods and attributes
        if not hasattr(ci_tool, 'trigger_action') or not callable(ci_tool.trigger_action):
            raise ValueError("ci_tool must have a 'trigger_action' method.")

        if not hasattr(ci_tool, 'name'):
            raise ValueError("ci_tool must have a 'name' attribute.")

        self.ci_tool = ci_tool
        self.project_name = project_name
        self.trigger_action = trigger_action
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.validate_response = validate_response
        self.results_ = None 

    @RunReturn(attribute_name="results_")
    def run(self, **run_kwargs):
        """
        Executes the CI action with retries and logging.

        This method attempts to trigger the specified CI action, retries if it
        fails, and logs each attempt. It validates the response from the CI tool
        if ``validate_response`` is ``True`` and handles failures according to
        the retry policy.

        Parameters
        ----------
        **run_kwargs : dict
            Additional runtime parameters to pass to the CI tool's
            ``trigger_action`` method.

        Returns
        -------
        results_ : dict
            A dictionary containing the results of the CI action, including
            success or failure details, response status, and any errors
            encountered.

        Raises
        ------
        ValueError
            If the CI action fails after all retry attempts or the response
            validation fails.

        Notes
        -----
        - The method sets the ``results_`` attribute with the CI action results.
        - It uses exponential backoff for retries.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self._trigger_ci_action(**run_kwargs)
                if self.validate_response:
                    if not self._validate_ci_response(response):
                        raise ValueError("Invalid response from CI tool.")
                self._set_runned()

                # Log success
                self._log_ci_event(response)
                self.results_ = self._build_run_results(response, success=True)
                break  # Exit loop if successful

            except Exception as e:
                if self.enable_logging:
                    self.log_event('ci_action_error', {
                        'attempt': attempt,
                        'error': str(e),
                        'project_name': self.project_name,
                        'ci_tool': self.ci_tool.name
                    })

                # Retry logic
                if attempt < self.retry_attempts:
                    self._handle_retry(attempt)
                else:
                    self.results_ = self._build_run_results(
                        None, success=False, error=str(e))
                    raise ValueError(
                        f"CI action failed after {self.retry_attempts} attempts.")

    def _trigger_ci_action(self, **run_kwargs):
        """
        Triggers the action on the CI tool with optional timeout and runtime
        configurations.

        Parameters
        ----------
        **run_kwargs : dict
            Additional parameters to pass to the CI tool's ``trigger_action``
            method.

        Returns
        -------
        response : object
            The response from the CI tool after triggering the action.

        Notes
        -----
        - This method handles triggering the CI action and passing any necessary
          runtime configurations such as timeouts.
        """
        if self.timeout:
            return self.ci_tool.trigger_action(
                self.project_name,
                action=self.trigger_action,
                timeout=self.timeout,
                **run_kwargs
            )
        else:
            return self.ci_tool.trigger_action(
                self.project_name,
                action=self.trigger_action,
                **run_kwargs
            )

    def _validate_ci_response(self, response) -> bool:
        """
        Validates the response from the CI tool to ensure success.

        Parameters
        ----------
        response : object
            The response object returned from the CI tool.

        Returns
        -------
        bool
            ``True`` if the response indicates success, ``False`` otherwise.

        Notes
        -----
        - The default validation checks for an HTTP 200 status and a ``success``
          field in the JSON response. This can be customized based on the CI
          tool's response format.
        """
        return response.status_code == 200 and 'success' in response.json()

    def _handle_retry(self, attempt: int):
        """
        Handles retry logic, including delays between attempts.

        Parameters
        ----------
        attempt : int
            The current attempt number.

        Notes
        -----
        - The delay between retries increases linearly based on the attempt
          number.
        - Logs the retry attempt details.
        """
        retry_in = self.retry_delay * attempt  # Linear backoff
        if self.enable_logging:
            self.log_event('ci_action_retry', {
                'attempt': attempt,
                'retry_in': retry_in,
                'project_name': self.project_name,
                'ci_tool': self.ci_tool.name
            })
        time.sleep(retry_in)

    def _log_ci_event(self, response):
        """
        Logs the details of the CI event, including the response.

        Parameters
        ----------
        response : object
            The response object returned from the CI tool.
        """
        if self.enable_logging:
            self.log_event('ci_integration_test', {
                'ci_tool': self.ci_tool.name,
                'project_name': self.project_name,
                'action_triggered': self.trigger_action,
                'response_status': response.status_code,
                'response': response.json()
            })

    def _build_run_results(self, response, success: bool,
                           error: Optional[str] = None) -> dict:
        """
        Builds and returns the results of the run, including success or error
        details.

        Parameters
        ----------
        response : object
            The response from the CI tool after triggering the action.

        success : bool
            Indicates whether the action was successful.

        error : str or None, default=None
            The error message if the action failed.

        Returns
        -------
        result : dict
            A dictionary containing details of the CI action results, including
            success status, response data, and any error messages.
        """
        result = {
            'ci_tool': self.ci_tool.name,
            'project_name': self.project_name,
            'action_triggered': self.trigger_action,
            'success': success
        }

        if success and response:
            result.update({
                'response_status': response.status_code,
                'response': response.json()
            })
        elif not success:
            result['error'] = error

        return result
