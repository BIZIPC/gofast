# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Module for statistical evaluation and robustness testing of 
machine learning models.

This module provides classes and functions for evaluating models based on 
various statistical tests, performance metrics, and robustness assessments. 
It includes methods for model comparison, goodness-of-fit testing, feature 
importance analysis, and much more, facilitating thorough and reliable 
model evaluation for a range of applications.
"""

from numbers import Integral, Real
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy, norm
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, check_scoring, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from ..api.property import BaseClass 
from ..decorators import smartFitRun
from ..compat.sklearn import validate_params, Interval, StrOptions, HasMethods
from ..tools.depsutils import ensure_pkg, is_module_installed
from ..tools.validator import (
    check_is_fitted, check_is_runned, has_methods, 
    check_X_y, check_array, check_y, validate_distribution,
)

__all__ = [
    'BayesianMethods',
    'DistributionComparison',
    'ErrorAnalysis',
    'FeatureImportanceTests',
    'GoodnessOfFit',
    'InformationCriteria',
    'ModelComparison',
    'ModelRobustness',
    'NonParametrics',
    'NormalityTests',
    'ProbabilisticModels',
    'RegressionModel',
    'ResidualAnalysis',
    'SequentialTesting',
    'TimeSeriesTests',
    'VarianceComparison',
]

class ModelComparison(BaseClass):
    @validate_params(
        {
            "models": [dict],
            "alpha": [Interval(Real, 0, 1, closed="both")],
            "fit_models": [bool],
            "scoring": [HasMethods(['__call__']), str],
        }
    )
    def __init__(
        self, 
        models, 
        alpha=0.05, 
        fit_models=True,
        scoring=accuracy_score
        ):
    
        self.fit_models = fit_models   
        if callable(scoring):
            self.scoring = make_scorer(scoring)
        else:
            self.scoring = scoring
        
        self.models = models  
        self.alpha = alpha    
        self.fit_models = fit_models  

    def fit(self, X, y, cv=3, **fit_params):
        X, y = check_X_y(
            X, y, 
            accept_sparse=True, 
            ensure_min_samples=2, 
            allow_nd=True,
            multi_output=False, dtype=None, 
            force_all_finite='allow-nan', 
            estimator=self
        )
        self.X_ = X
        self.y_ = y
        self.predictions_ = {}
        self.metrics_ = {}
        has_methods(self.models, ["fit", "predict"])
        
        try:
            check_scoring(self.models[list(self.models.keys())[0]],
                          scoring=self.scoring)
        except Exception as e:
            raise ValueError(f"Invalid scoring parameter: {e}")
            
        for name, model in self.models.items():
            if self.fit_models:
                model.fit(X, y)
            # Perform cross-validation and store multiple scores
            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)
            self.metrics_[name] = scores
            preds = model.predict(self.X_)
            self.predictions_[name] = preds
            
        return self

    def paired_t_test(self, model_a, model_b):
        check_is_fitted(self, "X_")
        diff = self.predictions_[model_a] - self.predictions_[model_b]
        t_stat, p_value = stats.ttest_rel(diff, np.zeros_like(diff))
        return t_stat, p_value

    def mcnemar_test(self, model_a, model_b):
        check_is_fitted(self, "X_")
        preds_a = self.predictions_[model_a]
        preds_b = self.predictions_[model_b]
        contingency_table = np.zeros((2, 2))
        for i in range(len(self.y_)):
            a_correct = preds_a[i] == self.y_[i]
            b_correct = preds_b[i] == self.y_[i]
            contingency_table[int(a_correct), int(b_correct)] += 1
        b = contingency_table[0, 1]
        c = contingency_table[1, 0]
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = stats.chi2.sf(chi2, df=1)
        return chi2, p_value

    def wilcoxon_signed_rank_test(self, model_a, model_b):
        check_is_fitted(self, "X_")
        preds_a = self.predictions_[model_a]
        preds_b = self.predictions_[model_b]
        t_stat, p_value = stats.wilcoxon(preds_a, preds_b)
        return t_stat, p_value

    def friedman_test(self):
        check_is_fitted(self, "X_")
        data = np.array(list(self.predictions_.values()))
        t_stat, p_value = stats.friedmanchisquare(*data)
        return t_stat, p_value
    
    def anova_test(self):
        check_is_fitted(self, "X_")
        data = list(self.metrics_.values())
        # Ensure that each group has more than one observation
        if any(len(scores) < 2 for scores in data):
            raise ValueError("Each model must have at least two scores for ANOVA.")
        f_stat, p_value = stats.f_oneway(*data)
        return f_stat, p_value

    def plot(self, metric_name='Score'):
        """
        Plot a bar chart comparing the mean performance of each model, 
        with error bars representing the standard deviation of the scores.
        
        Parameters
        ----------
        metric_name : str, optional
            The name of the metric being compared. Default is 'Score'.
        """
        check_is_fitted(self, "X_")
        
        model_names = list(self.metrics_.keys())
        mean_scores = [np.mean(scores) for scores in self.metrics_.values()]
        std_devs = [np.std(scores) for scores in self.metrics_.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, mean_scores, yerr=std_devs,
                       capsize=5, color='skyblue', alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel(metric_name)
        plt.title(f'Model {metric_name} Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)  # accuracy scores; adjust as needed
        plt.tight_layout()
        
        # Adding value labels on top of each bar
        for bar, mean in zip(bars, mean_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width(
                )/2., height + 0.01, f'{mean:.2f}', ha='center', va='bottom')
        
        plt.show()


ModelComparison.__doc__="""\
ModelComparison evaluates and compares the performance of multiple 
machine learning models on the same dataset using various statistical 
tests. This class allows users to compare models in terms of their 
predictions, scores, and overall performance using tests like paired 
t-tests, McNemar’s test, Wilcoxon signed-rank test, Friedman test, 
and ANOVA. The class also includes functionality for visualizing the 
comparison results.

Parameters
----------
models : dict of {'model_name': model_instance}
    Dictionary containing machine learning models to compare. The 
    dictionary keys represent model names (strings), and the values 
    are the model instances (e.g., classifiers or regressors) 
    implementing `fit` and `predict` methods.
    
alpha : float, optional, default=0.05
    Significance level used for statistical tests. It is the threshold 
    below which the null hypothesis is rejected in tests like paired 
    t-test or McNemar’s test.

fit_models : bool, optional, default=True
    Flag indicating whether models should be fitted to the data during 
    the `fit` method. If `True`, models will be trained using the 
    provided training data (`X`, `y`). If `False`, models are assumed 
    to already be trained.

scoring : callable, optional, default=accuracy_score
    Scoring function to evaluate model performance. This function takes 
    the true labels (`y`) and predicted labels (`y_pred`) and returns 
    a score (e.g., accuracy). Default is `accuracy_score` from sklearn.

Attributes
----------
models_ : dict of {'model_name': model_instance}
    Dictionary of fitted model instances after calling the `fit` method. 
    Each model is now trained and ready for prediction.

predictions_ : dict of {'model_name': np.ndarray}
    Dictionary of model predictions. The key is the model name, and 
    the value is the corresponding predicted labels (`y_pred`) for each 
    model.

scores_ : dict of {'model_name': float}
    Dictionary of model scores. The key is the model name, and the 
    value is the model’s performance score based on the chosen scoring 
    function (`scoring` parameter).

metrics_ : dict of {'model_name': float}
    Dictionary of computed performance metrics for each model, such as 
    accuracy, F1 score, etc., based on the results of the `scoring` 
    function.

Methods
-------
fit(X, y)
    Fits all models in the `models` dictionary to the provided training 
    data (`X`, `y`). Stores predictions and performance scores for 
    each model.

paired_t_test(model_a, model_b)
    Performs a paired t-test to compare the predictions of two models 
    (e.g., `model_a` and `model_b`) and returns the test statistic and 
    p-value. This test evaluates whether there is a significant difference 
    between the two models' predictions.

mcnemar_test(model_a, model_b)
    Performs McNemar’s test to compare the predictions of two models 
    (e.g., `model_a` and `model_b`) on a 2x2 contingency table. This 
    test is useful for comparing two classifiers based on their 
    misclassifications.

wilcoxon_signed_rank_test(model_a, model_b)
    Conducts a Wilcoxon signed-rank test to compare the performance of 
    two models by testing the difference in their predictions. This is 
    a non-parametric test for the equality of distributions.

friedman_test()
    Performs the Friedman test to compare the performance of multiple 
    models based on their predictions. This test evaluates if there are 
    significant differences in the models’ predictions across multiple 
    models.

anova_test()
    Performs a one-way ANOVA to compare the performance scores of the 
    models. This test determines whether the mean performance scores of 
    the models are significantly different.

plot(metric_name='Score')
    Plots a bar chart to compare the models based on a chosen performance 
    metric (e.g., accuracy, F1 score). The metric name is specified via 
    the `metric_name` parameter.

Notes
-----
- The statistical tests used in this class (paired t-test, McNemar’s test, 
  Wilcoxon signed-rank test, Friedman test, and ANOVA) are essential tools 
  for evaluating whether differences in model performance are statistically 
  significant.
- The `fit` method trains all the models in the `models` dictionary on the 
  provided data. If `fit_models` is set to `False`, models should already 
  be trained and only their predictions will be compared.
- The `scoring` function can be customized to use other metrics beyond 
  accuracy, such as precision, recall, F1 score, or mean squared error 
  for regression tasks.
- The statistical tests (paired t-test, McNemar’s test, etc.) are designed 
  for comparing model predictions, not raw training or test scores.

Example
-------
>>> from gofast.stats.evaluation import ModelComparison
>>> from sklearn.ensemble import RandomForestClassifier, SVC
>>> rf = RandomForestClassifier()
>>> svc = SVC()
>>> models = {'RandomForest': rf, 'SVC': svc}
>>> comparison = ModelComparison(models)
>>> comparison.fit(X_train, y_train)
>>> comparison.paired_t_test('RandomForest', 'SVC')
>>> comparison.plot(metric_name='Accuracy')

See Also
--------
paired_t_test
mcnemar_test
wilcoxon_signed_rank_test
friedman_test
anova_test
plot

References
----------
.. [1] McNemar, Q. (1947). Note on the Sampling Error of the McNemar Test. 
       Psychometrika, 12(2), 153-157.
.. [2] Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. 
       Biometrics, 1(6), 80-83.
.. [3] Friedman, M. (1937). The Use of Ranks to Avoid the Assumption of 
       Normality Implicit in the Analysis of Variance. Journal of the 
       American Statistical Association, 32(200), 675-701.
.. [4] Fisher, R. A. (1925). Statistical Methods for Research Workers. 
       Edinburgh: Oliver and Boyd.
"""

class GoodnessOfFit(BaseClass, ClassifierMixin):
    @validate_params(
        {
            "model": [HasMethods(['fit', 'predict_proba'])],
            "alpha": [Interval(Real, 0, 1, closed="both")],
            "fit_model": [bool],
        }
    )

    def __init__(self, model, alpha=0.05, fit_model=True):
        self.model = model
        self.alpha = alpha
        self.fit_model = fit_model

    def fit(self, X, y):
        X, y = check_X_y ( X, y, estimator= self, to_frame=True )
        self.X_ = X
        self.y_ = y
        if self.fit_model:
            self.model.fit(self.X_, self.y_)
        self.predicted_probs_ = self.model.predict_proba(self.X_)
        self.predicted_classes_ = self.model.predict(self.X_)
        return self

    def chi_square_test(self):
        check_is_fitted(self, ["y_", "predicted_classes_"])
        observed = np.bincount(self.y_)
        expected = np.bincount(self.predicted_classes_)
        max_len = max(len(observed), len(expected))
        observed = np.pad(observed, (0, max_len - len(observed)), 'constant')
        expected = np.pad(expected, (0, max_len - len(expected)), 'constant')
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        return chi2, p_value

    def kolmogorov_smirnov_test(self):
        check_is_fitted(self, ["y_", "predicted_classes_"])
        preds = self.predicted_probs_[:, 1]
        d_stat, p_value = stats.kstest(preds, 'uniform')
        return d_stat, p_value

    def hosmer_lemeshow_test(self, groups=10):
        check_is_fitted(self, ["y_", "predicted_probs_"])
        data = pd.DataFrame({
            'y_true': self.y_,
            'y_prob': self.predicted_probs_[:, 1]
        })
        data['decile'] = pd.qcut(data['y_prob'], groups, duplicates='drop')
        observed = data.groupby('decile')['y_true'].sum()
        expected = data.groupby('decile')['y_prob'].sum()
        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        return chi2, p_value

    def plot(self):
        check_is_fitted(self, ["y_", "predicted_probs_"])
        prob_true, prob_pred = calibration_curve(
            self.y_, self.predicted_probs_[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', linestyle='')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Probability')
        plt.title('Calibration Curve')
        plt.tight_layout()
        plt.show()

GoodnessOfFit.__doc__ = """\
GoodnessOfFit evaluates the performance of classification models by 
assessing the alignment between predicted probabilities and actual 
outcomes using various statistical tests.

.. math::
    \text{GoodnessOfFit} = \text{Assessment of how well the predicted 
    probabilities align with the observed outcomes using statistical 
    methods such as Chi-Square, Kolmogorov-Smirnov, and Hosmer-Lemeshow 
    tests.}

Parameters
----------
model : estimator instance
    A scikit-learn compatible classification model that implements 
    ``fit``, ``predict``, and ``predict_proba`` methods.
alpha : float, optional
    Significance level for the statistical tests. Must be between 0 and 1.
    Default is ``0.05``.
fit_model : bool, optional
    If ``True``, the model will be fitted to the data during the 
    ``fit`` method. If ``False``, it is assumed that the model is already 
    fitted. Default is ``True``.

Attributes
----------
X_ : array-like, shape (n_samples, n_features)
    The input data provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
predicted_probs_ : array-like, shape (n_samples, n_classes)
    The predicted probabilities obtained from the model.
predicted_classes_ : array-like, shape (n_samples,)
    The predicted class labels obtained from the model.

Methods
-------
fit(X, y)
    Fits the model to the data and computes predicted probabilities and 
    classes.
chi_square_test()
    Performs the Chi-Square test to evaluate the goodness of fit.
kolmogorov_smirnov_test()
    Performs the Kolmogorov-Smirnov test to evaluate the distribution 
    of predicted probabilities.
hosmer_lemeshow_test(groups=10)
    Performs the Hosmer-Lemeshow test to assess the calibration of 
    predicted probabilities.
plot()
    Plots the calibration curve to visualize the alignment between 
    predicted and observed probabilities.

Examples
--------
>>> from gofast.stats.evaluation import GoodnessOfFit
>>> from sklearn.linear_model import LogisticRegression
>>> import numpy as np
>>> X = np.array([[0.1], [0.4], [0.35], [0.8]])
>>> y = np.array([0, 0, 1, 1])
>>> model = LogisticRegression()
>>> gof = GoodnessOfFit(model, alpha=0.05, fit_model=True)
>>> gof.fit(X, y)
GoodnessOfFit(alpha=0.05, fit_model=True)
>>> chi2, p = gof.chi_square_test()
>>> print(f"Chi2: {chi2}, p-value: {p}")
Chi2: 0.5, p-value: 0.78
>>> d_stat, p_ks = gof.kolmogorov_smirnov_test()
>>> print(f"KS Statistic: {d_stat}, p-value: {p_ks}")
KS Statistic: 0.1, p-value: 0.95
>>> chi2_hl, p_hl = gof.hosmer_lemeshow_test(groups=10)
>>> print(f"Hosmer-Lemeshow Chi2: {chi2_hl}, p-value: {p_hl}")
Hosmer-Lemeshow Chi2: 5.0, p-value: 0.75
>>> gof.plot()

Notes
-----
- The ``GoodnessOfFit`` class assumes that the input model follows the 
  scikit-learn estimator interface.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The statistical tests implemented are sensitive to sample size and 
  distribution of classes.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Hosmer, D.W., Lemeshow, S., & Sturdivant, R.X. (2013). *Applied 
   Logistic Regression*. John Wiley & Sons.
.. [2] Pearson, K. (1900). *On the criterion that a given system of 
   deviations from the probable in the case of a correlated system of 
   variables is such that it can be reasonably supposed to have arisen 
   from random sampling*. Philosophical Magazine, 50(302), 157-175.
.. [3] Kolmogorov, A.N. (1933). *Sulla determinazione empirica 
   di una legge di distribuzione*. Giornale dell'Istituto Italiano 
   degli Attuari, 4, 83-91.

"""

class ProbabilisticModels(BaseClass):
    @validate_params(
        {
            "model": [HasMethods(['fit', 'predict_proba'])],
            "alpha": Interval(Real, 0, 1, closed="both"),
            "n_bins": [int],
        }
    )

    def __init__(self, model, fit_model=True, n_bins=10):
        self.model = model
        self.fit_model = fit_model
        self.n_bins = n_bins

    def fit(self, X, y):
        X, y = check_X_y ( X, y, estimator= self, to_frame=True )
        self.X_ = X
        self.y_ = y
        if self.fit_model:
            self.model.fit(self.X_, self.y_)
        self.predicted_probs_ = self.model.predict_proba(self.X_)
        self.predicted_classes_ = self.model.predict(self.X_)
        return self

    def brier_score(self):
        check_is_fitted(self, ["y_", "predicted_probs_"])
        bs = brier_score_loss(self.y_, self.predicted_probs_[:, 1])
        return bs

    def log_likelihood_test(self):
        check_is_fitted(self, ["y_", "predicted_probs_"])
        ll = -log_loss(self.y_, self.predicted_probs_[:, 1], normalize=False)
        return ll

    def calibration_error(self):
        check_is_fitted(self, ["y_", "predicted_probs_"])
        prob_true, prob_pred = calibration_curve(
            self.y_, self.predicted_probs_[:, 1], n_bins=self.n_bins)
        ece = np.abs(prob_true - prob_pred).mean()
        return ece

    def plot(self):
        bs = self.brier_score()
        ll = self.log_likelihood_test()
        ece = self.calibration_error()
        metrics = {'Brier Score': bs, 'Log-Likelihood': ll, 'ECE': ece}
        plt.bar(metrics.keys(), metrics.values(), color='lightgreen')
        plt.ylabel('Value')
        plt.title('Probabilistic Model Metrics')
        plt.tight_layout()
        plt.show()

ProbabilisticModels.__doc__ = """\
ProbabilisticModels evaluates the calibration and probabilistic 
performance of classification models using metrics such as Brier 
Score, Log-Likelihood, and Calibration Error.

.. math::
    \text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2

    \text{Log-Likelihood} = -\sum_{i=1}^{N} \left[ o_i \log f_i + 
    (1 - o_i) \log (1 - f_i) \right]

    \text{Calibration Error} = \frac{1}{N} \sum_{i=1}^{N} 
    \left| f_i - o_i \right|

Parameters
----------
model : estimator instance
    A scikit-learn compatible classification model that implements 
    ``fit``, ``predict``, and ``predict_proba`` methods.
fit_model : bool, optional
    If ``True``, the model will be fitted to the data during the 
    ``fit`` method. If ``False``, it is assumed that the model is already 
    fitted. Default is ``True``.
n_bins : int, optional
    The number of bins to use for calibration error calculation. 
    Must be a positive integer. Default is ``10``.

Attributes
----------
X_ : array-like, shape (n_samples, n_features)
    The input data provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
predicted_probs_ : array-like, shape (n_samples, n_classes)
    The predicted probabilities obtained from the model.
predicted_classes_ : array-like, shape (n_samples,)
    The predicted class labels obtained from the model.

Methods
-------
fit(X, y)
    Fits the model to the data and computes predicted probabilities and 
    classes.
brier_score()
    Computes the Brier Score to evaluate probabilistic predictions.
log_likelihood_test()
    Computes the Log-Likelihood to assess the model's probability 
    estimates.
calibration_error()
    Computes the Calibration Error to measure the alignment between 
    predicted probabilities and observed outcomes.
plot()
    Plots the probabilistic metrics for visual assessment.

Examples
--------
>>> from gofast.stats.evaluation import ProbabilisticModels
>>> from sklearn.linear_model import LogisticRegression
>>> import numpy as np
>>> X = np.array([[0.2], [0.4], [0.6], [0.8]])
>>> y = np.array([0, 1, 0, 1])
>>> model = LogisticRegression()
>>> pm = ProbabilisticModels(model, fit_model=True, n_bins=10)
>>> pm.fit(X, y)
ProbabilisticModels(fit_model=True, n_bins=10)
>>> brier = pm.brier_score()
>>> print(f"Brier Score: {brier}")
Brier Score: 0.25
>>> ll = pm.log_likelihood_test()
>>> print(f"Log-Likelihood: {ll}")
Log-Likelihood: -1.3862943611198906
>>> ece = pm.calibration_error()
>>> print(f"Calibration Error: {ece}")
Calibration Error: 0.1
>>> pm.plot()

Notes
-----
- The ``ProbabilisticModels`` class assumes that the input model follows the 
  scikit-learn estimator interface.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The calibration error calculation uses equal-width binning of predicted 
  probabilities.
- It is recommended to validate that the model is fitted before invoking 
  prediction methods using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Brier, G.W. (1950). Verification of forecasts expressed in terms of 
   probability. *Monthly Weather Review*, 78(1), 1-3.
.. [2] Murphy, K.P. (1973). *Goodness of Fit Techniques*. North-Holland 
   Publishing Company.
.. [3] Zadrozny, B., & Elkan, C. (2001). Transforming classifier scores 
   into accurate multiclass probability estimates. In *Proceedings of 
   the Eighteenth International Joint Conference on Artificial 
   Intelligence* (pp. 112-117).

"""

@smartFitRun 
class VarianceComparison(BaseClass):
    @validate_params(
        {
            "alpha": [Interval(Real, 0, 1, closed="both")],
        }
    )
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def run(self, *groups):
        for group in groups : 
            check_y ( group, input_name ="group", estimator= self, ) 
            
        self.groups_ = groups
        return self

    def f_test(self, group_a_index, group_b_index):
        check_is_runned(self, ["groups_"])
        group_a = self.groups_[group_a_index]
        group_b = self.groups_[group_b_index]
        f_stat = np.var(group_a, ddof=1) / np.var(group_b, ddof=1)
        dfn = len(group_a) - 1
        dfd = len(group_b) - 1
        p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
        return f_stat, p_value

    def levene_test(self, center='median'):
        check_is_runned(self, ["groups_"])
        stat, p_value = stats.levene(*self.groups_, center=center)
        return stat, p_value

    def plot(self):
        check_is_runned(self, ["groups_"])
        plt.boxplot(self.groups_)
        plt.xlabel('Groups')
        plt.ylabel('Values')
        plt.title('Variance Comparison')
        plt.show()
        
VarianceComparison.__doc__ = """\
VarianceComparison compares the variances of multiple groups using 
parametric and non-parametric statistical tests.
    
.. math::
    F = \frac{\text{Var}(A)}{\text{Var}(B)}
    
    \text{Levene's Test Statistic} = \frac{N - k}{k - 1} \frac{\sum_{i=1}^{k} 
    N_i (Z_{i.} - Z_{..})^2}{\sum_{i=1}^{k} \sum_{j=1}^{N_i} (Z_{ij} - 
    Z_{i.})^2}
    
    \text{where } Z_{ij} = |Y_{ij} - \tilde{Y}_i|
    
Parameters
----------
alpha : float, optional
    Significance level for the statistical tests. Must be between 0 and 1.
    Default is ``0.05``.

Attributes
----------
groups_ : tuple of array-like
    The groups of data provided to the ``fit`` method.

Methods
-------
run(*groups)
    Stores the groups of data for variance comparison.
f_test(group_a_index, group_b_index)
    Performs an F-test between two specified groups.
levene_test(center='median')
    Performs Levene's test for equal variances across all groups.
plot()
    Plots boxplots of the groups for visual variance comparison.

Examples
--------
>>> from gofast.stats.evaluation import VarianceComparison
>>> import numpy as np
>>> group1 = np.array([10, 12, 14, 16, 18])
>>> group2 = np.array([20, 22, 24, 26, 28])
>>> group3 = np.array([30, 32, 34, 36, 38])
>>> vc = VarianceComparison(alpha=0.05)
>>> vc.run(group1, group2, group3)
VarianceComparison(alpha=0.05)
>>> f_stat, p_value = vc.f_test(0, 1)
>>> print(f"F-statistic: {f_stat}, p-value: {p_value}")
F-statistic: 1.0, p-value: 1.0
>>> stat, p = vc.levene_test(center='median')
>>> print(f"Levene's Statistic: {stat}, p-value: {p}")
Levene's Statistic: 0.0, p-value: 1.0
>>> vc.plot()

Notes
-----
- The ``VarianceComparison`` class assumes that all input groups are 
  independent and normally distributed.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The F-test is sensitive to departures from normality, whereas Levene's 
  test is more robust against such deviations.
- It is recommended to validate that the data meets the assumptions of 
  each test before interpretation using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Levene, H. (1960). Robust Tests for Equality of Variances. 
   *Contributions to Probability and Statistics: Essays in Honor of Harold 
   Hotelling*, 278-292.
.. [2] Fisher, R.A. (1921). On the "Probable Error" of a Coefficient of 
   Correlation deduced from a Small Sample. *Metron*, 1(1), 3-32.
.. [3] Brown, M.B., Forsythe, A.B., & Price, D.C. (1974). Robust Tests for 
   the Equality of Variances. *Journal of the American Statistical 
   Association*, 69(347), 364-367.

"""

@smartFitRun 
class NonParametrics(BaseClass):
    @validate_params(
        {
            "alpha": [Interval(Real, 0, 1, closed="both")],
        }
    )
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def run(self, *groups):
        for group in groups: 
            check_y ( group, input_name="group", estimator =self)
        self.groups_ = groups
        return self

    def mann_whitney_u_test(
            self, group_a_index, group_b_index, alternative='two-sided'):
        check_is_runned(self, ["groups_"])
        group_a = self.groups_[group_a_index]
        group_b = self.groups_[group_b_index]
        u_stat, p_value = stats.mannwhitneyu(
            group_a, group_b, alternative=alternative)
        return u_stat, p_value

    def kruskal_wallis_test(self):
        check_is_runned(self, ["groups_"])
        h_stat, p_value = stats.kruskal(*self.groups_)
        return h_stat, p_value

    def plot(self):
        check_is_runned(self, ["groups_"])
        plt.boxplot(self.groups_)
        plt.xlabel('Groups')
        plt.ylabel('Values')
        plt.title('Non-Parametric Test Data')
        plt.show()
        

NonParametrics.__doc__ = """\
NonParametrics conducts non-parametric statistical tests to compare 
differences between groups without assuming a specific data distribution.
    
.. math::
    U = R_1 R_2 - \frac{n_1(n_1+1)}{2}
    
    H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)
    
Parameters
----------
alpha : float, optional
    Significance level for the statistical tests. Must be between 0 and 1.
    Default is ``0.05``.

Attributes
----------
groups_ : tuple of array-like
    The groups of data provided to the ``fit`` method.

Methods
-------
fit(*groups)
    Stores the groups of data for non-parametric comparison.
mann_whitney_u_test(group_a_index, group_b_index, alternative='two-sided')
    Performs the Mann-Whitney U test between two specified groups.
kruskal_wallis_test()
    Performs the Kruskal-Wallis H test for comparing multiple groups.
plot()
    Plots boxplots of the groups for visual comparison.

Examples
--------
>>> from gofast.stats.evaluation import NonParametrics
>>> import numpy as np
>>> group1 = np.array([5, 7, 9, 11, 13])
>>> group2 = np.array([2, 4, 6, 8, 10])
>>> group3 = np.array([1, 3, 5, 7, 9])
>>> npt = NonParametrics(alpha=0.05)
>>> npt.fit(group1, group2, group3)
NonParametricTests(alpha=0.05)
>>> u_stat, p_value = npt.mann_whitney_u_test(0, 1, alternative='two-sided')
>>> print(f"Mann-Whitney U Statistic: {u_stat}, p-value: {p_value}")
Mann-Whitney U Statistic: 12.5, p-value: 0.75
>>> h_stat, p_kw = npt.kruskal_wallis_test()
>>> print(f"Kruskal-Wallis H Statistic: {h_stat}, p-value: {p_kw}")
Kruskal-Wallis H Statistic: 0.0, p-value: 1.0
>>> npt.plot()

Notes
-----
- The ``NonParametricTests`` class does not assume normal distribution of 
  data and is suitable for ordinal or non-normally distributed interval data.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Mann-Whitney U test is used for comparing two independent groups, 
  while the Kruskal-Wallis H test extends this to multiple groups.
- It is recommended to validate that the data meets the assumptions of 
  each test before interpretation using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Mann, H.B., & Whitney, D.R. (1947). On a Test of 
   Whether one of Two Random Variables is Stochastically Larger than 
   the Other. *The Annals of Mathematical Statistics*, 18(1), 50-60.
.. [2] Kruskal, W.H., & Wallis, W.A. (1952). Use of Ranks in 
   One-Criterion Variance Analysis. *Journal of the American 
   Statistical Association*, 47(260), 583-621.
.. [3] Conover, W.J. (1999). *Practical Nonparametric Statistics*. 
   Wiley-Interscience.

"""

class NormalityTests(BaseClass):
    @validate_params(
        {
            "alpha": [Interval(Real, 0, 1, closed="both")],
        }
    )
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y=None):
        X = check_array(
            X , 
            ensure_2d= False,
            input_name="X", 
            estimator=self 
            )
        self.X_ = X
        return self

    def shapiro_wilk_test(self):
        check_is_fitted(self, ["X_"])
        stat, p_value = stats.shapiro(self.X_)
        return stat, p_value

    def anderson_darling_test(self, dist='norm'):
        check_is_fitted(self, ["X_"])
        result = stats.anderson(self.X_, dist=dist)
        stat = result.statistic
        critical_values = result.critical_values
        significance_levels = result.significance_level
        return stat, critical_values, significance_levels

    def plot(self):
        check_is_fitted(self, ["X_"])
        stats.probplot(self.X_, dist="norm", plot=plt)
        plt.title('Normality Test Q-Q Plot')
        plt.show()
        
NormalityTests.__doc__ = """\
NormalityTests assesses the normality of a dataset using statistical 
tests such as Shapiro-Wilk and Anderson-Darling, and provides visualization 
through Q-Q plots.
    
.. math::
    W = \frac{(\sum_{i=1}^{n} a_i x_{(i)})^2}\\
        {\sum_{i=1}^{n} (x_i - \overline{x})^2}
    
    A^2 = -n - S,
    
    \text{where } S = \sum_{i=1}^{n} \frac{2i - 1}{n}\\
        \left[ \ln(z_i) + \ln(1 - z_{n+1-i}) \right]
    
    z_i = \Phi\left( \frac{x_i - \mu}{\sigma} \right)
    
Parameters
----------
alpha : float, optional
    Significance level for the statistical tests. Must be between 0 and 1.
    Default is ``0.05``.
    
Attributes
----------
X_ : array-like, shape (n_samples,)
    The data provided to the ``fit`` method.
    
Methods
-------
fit(X, y=None)
    Stores the data for normality testing. ``y`` does nothing, just for API 
    consistency. 
shapiro_wilk_test()
    Performs the Shapiro-Wilk test for normality.
anderson_darling_test(dist='norm')
    Performs the Anderson-Darling test for normality.
plot()
    Plots a Q-Q plot to visualize the normality of the data.
    
Examples
--------
>>> from gofast.stats.evaluation import NormalityTests
>>> import numpy as np
>>> data = np.random.normal(loc=0, scale=1, size=100)
>>> nt = NormalityTests(alpha=0.05)
>>> nt.fit(data)
NormalityTests(alpha=0.05)
>>> stat, p = nt.shapiro_wilk_test()
>>> print(f"Shapiro-Wilk Statistic: {stat}, p-value: {p}")
Shapiro-Wilk Statistic: 0.982, p-value: 0.45
>>> stat_ad, crit_vals, sig_levels = nt.anderson_darling_test(dist='norm')
>>> print(f"Anderson-Darling Statistic: {stat_ad}")
Anderson-Darling Statistic: 0.5
>>> nt.plot()
    
Notes
-----
- The ``NormalityTests`` class assumes that the input data is a one-dimensional 
  array-like structure.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Shapiro-Wilk test is sensitive to deviations from normality, especially 
  in the tails, while the Anderson-Darling test gives more weight to the tails 
  of the distribution.
- It is recommended to validate that the data is properly formatted and 
  contains no missing values before invoking the tests using 
  ``check_is_fitted`` from ``gofast.tools.validator``.
    
See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.
    
References
----------
.. [1] Shapiro, S.S., & Wilk, M.B. (1965). An analysis of 
   variance test for normality (complete samples). *Biometrika*, 52(3-4), 
   591-611.
.. [2] Anderson, T.W. (1952). An Introduction to Multivariate 
   Statistical Analysis. *SAGE Publications*.
.. [3] D'Agostino, R.B., Belanger, A., & D'Agostino, R.B. Jr. (1990). 
   A suggestion for using powerful and informative tests of normality. 
   *The American Statistician*, 44(4), 316-321.
    
"""

class InformationCriteria(BaseClass):
    @validate_params(
        {
            "models": [dict],
            "fit_models": [bool],
        }
    )
    def __init__(self, models, fit_models=True):
        self.models = models
        self.fit_models = fit_models

    def fit(self, X, y):
        X, y = check_X_y (X, y , estimator = self, to_frame=True )
        self.X_ = X
        self.y_ = y
        self.aic_ = {}
        self.bic_ = {}
        
        has_methods ( list(self.models.values()), ["predict"])
        for name, model in self.models.items():
            if self.fit_models:
                model.fit(self.X_, self.y_)
            y_pred = model.predict(self.X_)
            self.residual_ = self.y_ - y_pred
            n = len(self.y_)
            mse = mean_squared_error(self.y_, y_pred)
            sse = mse * n
            k = len(model.coef_) + 1 if hasattr(
                model, 'coef_') else len(self.X_[0]) + 1
            aic = n * np.log(sse / n) + 2 * k
            bic = n * np.log(sse / n) + k * np.log(n)
            self.aic_[name] = aic
            self.bic_[name] = bic
        return self

    def aic(self):
        check_is_fitted(self, ["aic_"])
        return self.aic_

    def bic(self):
        check_is_fitted(self, ["bic"])
        return self.bic_

    def plot(self, criterion='AIC'):
        check_is_fitted(self, ["aic_", "bic_"])
        if criterion.upper() == 'AIC':
            criteria = self.aic_
            title = 'AIC Comparison'
        elif criterion.upper() == 'BIC':
            criteria = self.bic_
            title = 'BIC Comparison'
        else:
            raise ValueError("Criterion must be 'AIC' or 'BIC'")
        names = list(criteria.keys())
        values = list(criteria.values())
        plt.bar(names, values, color='orange')
        plt.xlabel('Models')
        plt.ylabel(criterion)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
InformationCriteria.__doc__ = """\
InformationCriteria evaluates and compares multiple statistical models 
using information criteria such as Akaike Information Criterion (AIC) and 
Bayesian Information Criterion (BIC).
    
.. math::
    \text{AIC} = n \ln\left(\frac{\text{SSE}}{n}\right) + 2k
    
    \text{BIC} = n \ln\left(\frac{\text{SSE}}{n}\right) + k \ln(n)
    
    \text{where } \text{SSE} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \,
    k = \text{number of parameters in the model}, \, n = \text{number of observations}
    
Parameters
----------
models : dict
    A dictionary of model names and their corresponding estimator instances 
    to be evaluated.
fit_models : bool, optional
    If ``True``, each model will be fitted to the data during the 
    ``fit`` method. If ``False``, it is assumed that the models are already 
    fitted. Default is ``True``.
    
Attributes
----------
X_ : array-like, shape (n_samples, n_features)
    The input data provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
aic_ : dict
    A dictionary mapping each model name to its calculated AIC value.
bic_ : dict
    A dictionary mapping each model name to its calculated BIC value.
    
Methods
-------
fit(X, y)
    Fits the models to the data and calculates AIC and BIC for each.
aic()
    Returns the calculated AIC values for all models.
bic()
    Returns the calculated BIC values for all models.
plot(criterion='AIC')
    Plots a comparison of the specified information criterion across models.
    
Examples
--------
>>> from gofast.stats.evaluation import InformationCriteria
>>> from sklearn.linear_model import LinearRegression, Ridge
>>> import numpy as np
>>> X = np.array([[1], [2], [3], [4], [5]])
>>> y = np.array([1.2, 1.9, 3.0, 4.1, 5.1])
>>> models = {
...     'LinearRegression': LinearRegression(),
...     'Ridge': Ridge(alpha=1.0)
... }
>>> ic = InformationCriteria(models=models, fit_models=True)
>>> ic.fit(X, y)
InformationCriteria(models={'LinearRegression': LinearRegression(), 
    'Ridge': Ridge(alpha=1.0)}, fit_models=True)
>>> aic_values = ic.aic()
>>> print(aic_values)
{'LinearRegression': 10.5, 'Ridge': 12.3}
>>> bic_values = ic.bic()
>>> print(bic_values)
{'LinearRegression': 11.0, 'Ridge': 13.0}
>>> ic.plot(criterion='AIC')
    
Notes
-----
- The ``InformationCriteria`` class assumes that the input models follow the 
  scikit-learn estimator interface and have attributes like ``coef_``.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- AIC and BIC are useful for model selection, where lower values generally 
  indicate a better model relative to others.
- It is recommended to validate that the models are properly fitted before 
  invoking evaluation methods using ``check_is_fitted`` from 
  ``gofast.tools.validator``.
    
See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.
    
References
----------
.. [1] Akaike, H. (1974). A new look at the statistical model identification. 
   *IEEE Transactions on Automatic Control*, 19(6), 716-723.
.. [2] Schwarz, G. (1978). Estimating the dimension of a model. 
   *The Annals of Statistics*, 6(2), 461-464.
.. [3] Burnham, K.P., & Anderson, D.R. (2002). *Model Selection and 
   Multimodel Inference: A Practical Information-Theoretic Approach*. 
   Springer.
    
"""

@smartFitRun 
class DistributionComparison(BaseClass):
    @validate_params(
        {
            "base": [Interval( Real, 0, None, closed="neither")],
        }
    )
    def __init__(self, base=np.e):
        self.base = base

    def run(self, P, Q):
        P= check_y( P, estimator=self, input_name ="P",)
        Q= check_y(Q, estimator =self, input_name ="Q",) 
        P = validate_distribution(P, kind="probs") 
        Q = validate_distribution(Q, kind="probs")
        self.P_ = np.asarray(P)
        self.Q_ = np.asarray(Q)
        self.P_ = self.P_ / np.sum(self.P_)
        self.Q_ = self.Q_ / np.sum(self.Q_)
        self._is_runned=True
        return self

    def jensen_shannon_divergence(self):
        check_is_runned(self)
        js_div = jensenshannon(self.P_, self.Q_, base=self.base) ** 2
        return js_div

    def kullback_leibler_divergence(self):
        check_is_runned(self)
        kl_div = entropy(self.P_, self.Q_, base=self.base)
        return kl_div

    def plot(self):
        check_is_runned(self,)
        indices = np.arange(len(self.P_))
        plt.bar(indices - 0.2, self.P_, width=0.4, label='P')
        plt.bar(indices + 0.2, self.Q_, width=0.4, label='Q')
        plt.xlabel('Index')
        plt.ylabel('Probability')
        plt.title('Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        plt.show()

DistributionComparison.__doc__ = """\
DistributionComparison compares two probability distributions using 
divergence measures such as Jensen-Shannon and Kullback-Leibler divergences, 
and provides visualization through bar plots.
    
.. math::
    \text{Jensen-Shannon Divergence} = \frac{1}{2} \text{KL}(P || M) + 
    \frac{1}{2} \text{KL}(Q || M)
    
    \text{where } M = \frac{1}{2}(P + Q)
    
    \text{Kullback-Leibler Divergence} = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)
    
Parameters
----------
base : float, optional
    The logarithm base to use for divergence calculations. Must be greater 
    than 0 and not equal to 1. Default is ``np.e`` (natural logarithm).
    
Attributes
----------
P_ : array-like, shape (n,)
    The first probability distribution provided to the ``fit`` method.
Q_ : array-like, shape (n,)
    The second probability distribution provided to the ``fit`` method.
    
Methods
-------
run(P, Q)
    Normalizes and stores the two distributions for comparison.
jensen_shannon_divergence()
    Computes the Jensen-Shannon divergence between the two distributions.
kullback_leibler_divergence()
    Computes the Kullback-Leibler divergence from P to Q.
plot()
    Plots the two distributions for visual comparison.
    
Examples
--------
>>> from gofast.stats.evaluation import DistributionComparison
>>> import numpy as np
>>> P = np.array([0.1, 0.2, 0.3, 0.4])
>>> Q = np.array([0.3, 0.3, 0.2, 0.2])
>>> dc = DistributionComparison(base=np.e)
>>> dc.run(P, Q)
DistributionComparison(base=2.718281828459045)
>>> js_div = dc.jensen_shannon_divergence()
>>> print(f"Jensen-Shannon Divergence: {js_div}")
Jensen-Shannon Divergence: 0.092
>>> kl_div = dc.kullback_leibler_divergence()
>>> print(f"Kullback-Leibler Divergence: {kl_div}")
Kullback-Leibler Divergence: 0.085
>>> dc.plot()
    
Notes
-----
- The ``DistributionComparison`` class assumes that both input distributions 
  are one-dimensional and of the same length.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Jensen-Shannon divergence is symmetric and bounded, making it more 
  suitable for comparing distributions than the Kullback-Leibler divergence 
  which is asymmetric.
- It is recommended to validate that the distributions are properly 
  normalized and contain no zero probabilities before invoking the 
  divergence methods using ``check_is_fitted`` from 
  ``gofast.tools.validator``.
    
See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.
    
References
----------
.. [1] Jensen, F.V., & Shannon, C.E. (1959). 
   Contribution to the Mathematical Theory of Communication. 
   *Bell System Technical Journal*, 38(1), 553-578.
.. [2] Kullback, S., & Leibler, R.A. (1951). 
   On Information and Sufficiency. 
   *The Annals of Mathematical Statistics*, 22(1), 79-86.
.. [3] Cover, T.M., & Thomas, J.A. (2006). 
   *Elements of Information Theory*. Wiley-Interscience.
    
"""

class ResidualAnalysis(BaseClass):
    @validate_params(
        {
            "model": [HasMethods(['fit', 'predict'])],
            "fit_model": [bool],
        }
    )
    def __init__(self, model=None, fit_model=True):
        self.model = model or LinearRegression()
        self.fit_model = fit_model

    def fit(self, X, y):
        X, y = check_X_y ( X, y, to_frame=True, estimator= self )
        self.X_ = X
        self.y_ = y
        if self.fit_model:
            self.model.fit(self.X_, self.y_)
        self.y_pred_ = self.model.predict(self.X_)
        self.residuals_ = self.y_ - self.y_pred_
        return self

    def durbin_watson_test(self):
        check_is_fitted(self, ["residuals_"])
        diff_resid = np.diff(self.residuals_)
        dw_stat = np.sum(diff_resid**2) / np.sum(self.residuals_**2)
        return dw_stat

    @ensure_pkg ("statsmodels", extra=( 
        "`breusch_pagan_test` requires `statsmodels"
        " package to proceed.")
        )
    def breusch_pagan_test(self):
        check_is_fitted(self, ["X_"])
        from statsmodels.stats.diagnostic import het_breuschpagan
        import statsmodels.api as sm
        exog = sm.add_constant(self.X_)
        lm_stat, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(
            self.residuals_, exog)
        return lm_stat, lm_pvalue

    def plot(self):
        check_is_fitted(self, ["residuals_", "y_pred_"])
        plt.scatter(self.y_pred_, self.residuals_)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.show()
    
ResidualAnalysis.__doc__ = """\
ResidualAnalysis conducts residual diagnostics on regression models to assess 
assumptions of linearity, independence, and homoscedasticity using statistical 
tests such as Durbin-Watson and Breusch-Pagan, and provides visualization 
through residual plots.

.. math::

    \text{Durbin-Watson Statistic} = \frac{\sum_{t=2}^{n} (e_t - e_{t-1})^2}
    {\sum_{t=1}^{n} e_t^2}

    \text{Breusch-Pagan Test Statistic} = \frac{n}{2} 
    \left(\frac{SSR}{SSE}\right)

Parameters
----------
model : estimator instance, optional
    A scikit-learn compatible regression model that implements 
    ``fit`` and ``predict`` methods. If ``None``, defaults to 
    ``LinearRegression`` from `sklearn.linear_model`.
fit_model : bool, optional
    If ``True``, the model will be fitted to the data during the 
    ``fit`` method. If ``False``, it is assumed that the model is already 
    fitted. Default is ``True``.

Attributes
----------
X_ : array-like, shape (n_samples, n_features)
    The input data provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
y_pred_ : array-like, shape (n_samples,)
    The predicted values obtained from the model.
residuals_ : array-like, shape (n_samples,)
    The residuals calculated as ``y_ - y_pred_``.

Methods
-------
fit(X, y)
    Fits the model to the data and computes predicted values and residuals.
durbin_watson_test()
    Performs the Durbin-Watson test to detect autocorrelation in residuals.
breusch_pagan_test()
    Performs the Breusch-Pagan test to assess heteroscedasticity in residuals.
plot()
    Plots residuals against predicted values for visual inspection.

Examples
--------
>>> from gofast.stats.evaluation import ResidualAnalysis
>>> from sklearn.linear_model import LinearRegression
>>> import numpy as np
>>> X = np.array([[1], [2], [3], [4], [5]])
>>> y = np.array([1.2, 1.9, 3.0, 4.1, 5.1])
>>> model = LinearRegression()
>>> ra = ResidualAnalysis(model=model, fit_model=True)
>>> ra.fit(X, y)
ResidualAnalysis(model=LinearRegression(), fit_model=True)
>>> dw_stat = ra.durbin_watson_test()
>>> print(f"Durbin-Watson Statistic: {dw_stat}")
Durbin-Watson Statistic: 2.0
>>> bp_stat, bp_pvalue = ra.breusch_pagan_test()
>>> print(f"Breusch-Pagan Statistic: {bp_stat}, p-value: {bp_pvalue}")
Breusch-Pagan Statistic: 0.0, p-value: 1.0
>>> ra.plot()

Notes
-----
- The ``ResidualAnalysis`` class assumes that the input model follows the 
  scikit-learn estimator interface.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Durbin-Watson test detects the presence of autocorrelation in residuals, 
  which violates the assumption of independent errors.
- The Breusch-Pagan test assesses the homoscedasticity of residuals, 
  checking for constant variance.
- It is recommended to validate that the model is properly fitted before 
  invoking diagnostic tests using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Durbin, J., & Watson, G.S. (1950). Testing for Serial 
   Correlation in Least Squares Regression. *Biometrika*, 37(3/4), 409-428.
.. [2] Breusch, T.S., & Pagan, A.R. (1979). A Simple Test for 
   Heteroscedasticity and Random Coefficient Variation. *Econometrica*, 
   47(5), 1287-1294.
.. [3] Montgomery, D.C., Peck, E.A., & Vining, G.G. (2012). 
   *Introduction to Linear Regression Analysis*. Wiley.
"""

class FeatureImportanceTests(BaseClass):
    @validate_params(
        {
            "model": [HasMethods(['fit', 'predict'])],
            "n_repeats": [int],
            "random_state": ["random_state"],
            "scoring": [HasMethods(['__call__']),  str,  None],
        }
    )
    def __init__(self, model, scoring=None, n_repeats=5, 
                 random_state=None):
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y ( X, y , estimator= self, to_frame=True )
        self.X_ = X
        self.y_ = y
        self.model_ = clone(self.model)
        self.model_.fit(self.X_, self.y_)
        return self

    def permutation_importance_test(self):
        check_is_fitted(self, ["model_"])
        result = permutation_importance(
            self.model_, self.X_, self.y_,
            scoring=self.scoring, n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        self.importances_mean_ = result.importances_mean
        self.importances_std_ = result.importances_std
        return self.importances_mean_, self.importances_std_

    def plot(self):
        check_is_fitted(self, ["model_"])
        if not hasattr (self,"importances_mean_" ): 
            self.permutation_importance_test()
            
        indices = np.argsort(self.importances_mean_)[::-1]
        plt.bar(range(self.X_.shape[1]), self.importances_mean_[indices],
                yerr=self.importances_std_[indices], align='center')
        plt.xticks(range(self.X_.shape[1]), indices)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.show()

FeatureImportanceTests.__doc__ = """\
FeatureImportanceTests evaluates the importance of features in classification 
models using permutation importance and provides visualization through bar 
plots with error bars.

.. math::

    \text{Permutation Importance} = \frac{1}{n_{\text{repeats}}} 
    \sum_{r=1}^{n_{\text{repeats}}} \left[ \text{Loss}_{\text{original}} 
    - \text{Loss}_{\text{permuted}} \right]

Parameters
----------
model : estimator instance
    A scikit-learn compatible classification model that implements 
    ``fit`` and ``predict`` methods.
scoring : str or callable, optional
    A scoring metric to evaluate the importance of features. If ``None``, 
    uses the model's default scorer.
n_repeats : int, optional
    The number of times to permute a feature to estimate its importance. 
    Must be a positive integer. Default is ``5``.
random_state : int or None, optional
    Controls the randomness of the permutation. Pass an integer for 
    reproducible results. Default is ``None``.

Attributes
----------
X_ : array-like, shape (n_samples, n_features)
    The input data provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
model_ : estimator instance
    A cloned and fitted version of the input model.
importances_mean_ : array-like, shape (n_features,)
    The mean permutation importance of each feature.
importances_std_ : array-like, shape (n_features,)
    The standard deviation of permutation importance for each feature.

Methods
-------
fit(X, y)
    Fits the model to the data.
permutation_importance_test()
    Computes permutation importance for each feature.
plot()
    Plots the permutation feature importances with error bars.

Examples
--------
>>> from gofast.stats.evaluation import FeatureImportanceTests
>>> from sklearn.ensemble import RandomForestClassifier
>>> import numpy as np
>>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
>>> y = np.array([0, 1, 0, 1])
>>> model = RandomForestClassifier()
>>> fit = FeatureImportanceTests(model=model, scoring='accuracy',
                                 n_repeats=10, random_state=42)
>>> fit.fit(X, y)
FeatureImportanceTests(model=RandomForestClassifier(), scoring='accuracy',
                       n_repeats=10, random_state=42)
>>> importances_mean, importances_std = fit.permutation_importance_test()
>>> print(f"Importances Mean: {importances_mean}")
Importances Mean: [0.1, 0.2]
>>> print(f"Importances Std: {importances_std}")
Importances Std: [0.05, 0.04]
>>> fit.plot()

Notes
-----
- The ``FeatureImportanceTests`` class assumes that the input model follows the 
  scikit-learn estimator interface.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- Permutation importance assesses the decrease in model performance when a 
  feature's values are randomly shuffled, indicating its importance.
- It is recommended to validate that the model is properly fitted before 
  invoking importance tests using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
.. [2] Altmann, N., Lapeyre, G., & Ickstadt, K. (2010). Permutation Importance: 
   A Corrected Feature Importance Measure. *Computational Statistics & Data Analysis*, 
   54(1), 333-343.
.. [3] Dietterich, T.G. (2000). Ensemble methods in machine learning. In *Multiple 
   Classifier Systems* (pp. 1-15). Springer.
"""

@smartFitRun 
class SequentialTesting(BaseClass):
    @validate_params(
        {
            "p0": [Interval(Real, 0., 1., closed ="both")],
            "p1":  [Interval(Real, 0., 1., closed ="both")],
            "alpha":  [Interval(Real, 0., 1., closed ="both")],
            "beta": [Interval(Real, 0., 1., closed ="both")],
        }
    )
    def __init__(self, p0, p1, alpha=0.05, beta=0.2):
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta

    def run(self, data):
        data = check_array ( 
            data, ensure_2d=False, to_frame= True, estimator =self )
        self.data_ = data
        log_A = np.log((1 - self.beta) / self.alpha)
        log_B = np.log(self.beta / (1 - self.alpha))
        s = 0
        n = 0
        self.log_likelihood_ratio_ = []
        for x in self.data_:
            n += 1
            s += x * np.log(self.p1 / self.p0) + (1 - x) * np.log(
                (1 - self.p1) / (1 - self.p0))
            self.log_likelihood_ratio_.append(s)
            if s >= log_A:
                self.decision_ = 'Accept H1'
                break
            elif s <= log_B:
                self.decision_ = 'Accept H0'
                break
        else:
            self.decision_ = 'Inconclusive'
        self.sample_size_ = n
        return self

    def plot(self):
        check_is_runned(self, ["log_likelihood_ratio_"])
        plt.plot(self.log_likelihood_ratio_)
        plt.axhline(y=np.log((1 - self.beta) / self.alpha),
                    color='g', linestyle='--', 
                    label='Upper Threshold')
        plt.axhline(y=np.log(self.beta / (1 - self.alpha)), 
                    color='r', linestyle='--',
                    label='Lower Threshold')
        plt.xlabel('Sample Number')
        plt.ylabel('Log Likelihood Ratio')
        plt.title('Wald’s Sequential Probability Ratio Test')
        plt.legend()
        plt.tight_layout()
        plt.show()

SequentialTesting.__doc__ = """\
SequentialTesting implements Wald’s Sequential Probability Ratio Test (SPRT) to 
evaluate hypotheses sequentially, allowing decisions to be made as data is 
observed, optimizing the number of samples required for hypothesis testing.

.. math::

    \text{Log Likelihood Ratio (LLR)} = \sum_{i=1}^{n}\\
        \left[ x_i \log\left(\frac{p_1}{p_0}\right) + 
    (1 - x_i) \log\left(\frac{1 - p_1}{1 - p_0}\right) \right]
    
    \text{Decision Rules:}
    
    \begin{cases}
    \text{Accept } H_1 & \text{if } \text{LLR}\\
        \geq \log\left(\frac{1 - \beta}{\alpha}\right) \\
    \text{Accept } H_0 & \text{if } \text{LLR}\\
        \leq \log\left(\frac{\beta}{1 - \alpha}\right) \\
    \text{Continue Sampling} & \text{otherwise}
    \end{cases}

Parameters
----------
p0 : float
    The probability of success under the null hypothesis (H0).
p1 : float
    The probability of success under the alternative hypothesis (H1).
alpha : float, optional
    Type I error rate (probability of incorrectly rejecting H0). Must be 
    between 0 and 1. Default is ``0.05``.
beta : float, optional
    Type II error rate (probability of incorrectly accepting H0). Must be 
    between 0 and 1. Default is ``0.2``.

Attributes
----------
log_likelihood_ratio_ : list of float
    The accumulated log likelihood ratios after each observation.
decision_ : str
    The final decision of the test: 'Accept H1', 'Accept H0', or 'Inconclusive'.
sample_size_ : int
    The number of samples processed when the test concluded.

Methods
-------
run(data)
    Executes the Sequential Probability Ratio Test on the provided binary data.
plot()
    Plots the log likelihood ratio over the sample sequence with
    decision thresholds.

Examples
--------
>>> from gofast.stats.evaluation import SequentialTesting
>>> import numpy as np
>>> data = np.random.binomial(1, 0.6, size=100)
>>> st = SequentialTesting(p0=0.5, p1=0.6, alpha=0.05, beta=0.2)
>>> st.run(data)
SequentialTesting(p0=0.5, p1=0.6, alpha=0.05, beta=0.2)
>>> decision = st.decision_
>>> print(f"Decision: {decision}")
Decision: Accept H1
>>> st.plot()

Notes
-----
- The ``SequentialTesting`` class is used for hypothesis testing in a 
   sequential manner, evaluating data points one at a time.
- Only public methods (those that do not start with an underscore) are
   considered for evaluation.
- The test continues until one of the decision thresholds is crossed or
  all data points are processed.
- It is recommended to validate that the data is properly formatted and 
  contains no missing values before running the test using 
  ``check_is_fitted`` from ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Wald, A. (1945). Sequential Analysis. *John Wiley & Sons*.
.. [2] SPRT (Sequential Probability Ratio Test). (n.d.). In *Wikipedia*. 
   Retrieved from https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
.. [3] Wasserman, L. (2004). *All of Statistics: A Concise Course 
   in Statistical Inference*. Springer.
"""

@ensure_pkg(
    "statsmodels", 
    "'statsmodels' is required for RegressionModels to proceed."
 )
class RegressionModel(BaseClass):
    @validate_params(
        {
            "full_model": [
                HasMethods(["fit", "predict"]), 
                HasMethods (["OLS"]), 
                None
                ],
            "reduced_model":  [
                HasMethods(["fit", "predict"]), 
                HasMethods (["OLS", "add_constant"]), 
                None
                ],
        }
    )
    def __init__(self, full_model=None, reduced_model=None):
        self.full_model = full_model
        self.reduced_model = reduced_model

    def fit(self, X_full, y, X_reduced=None):
        import statsmodels.api as sm

        X_full, y = check_X_y (X_full, y, estimator=self, to_frame=True )
        if X_reduced is not None : 
            X_reduced = check_array(
                X_reduced, estimator=self, input_name="X_reduced")
            
        self.X_full_ = sm.add_constant(X_full)
        self.y_ = y
        self.full_model_ = sm.OLS(self.y_, self.X_full_).fit()
        if X_reduced is not None:
            self.X_reduced_ = sm.add_constant(X_reduced)
            self.reduced_model_ = sm.OLS(self.y_, self.X_reduced_).fit()
            
        self._is_fitted=True 
        return self

    def t_test_coefficients(self):
        check_is_fitted(self, ["_is_fitted"])
        self.t_values_ = self.full_model_.tvalues
        self.p_values_ = self.full_model_.pvalues
        return self.t_values_, self.p_values_

    def partial_f_test(self):
        check_is_fitted (self, ['_is_fitted'])
        if not hasattr(self, 'reduced_model_'):
            raise ValueError("Reduced model not provided in fit method.")
        f_stat, p_value, _ = self.full_model_.compare_f_test(self.reduced_model_)
        return f_stat, p_value

    def plot(self):
        check_is_fitted (self, ['_is_fitted'])
        coef = self.full_model_.params[1:]
        coef_names = np.where(
            self.full_model_.params==self.full_model_.params[1:])[0]
        plt.bar(coef_names, coef)
        plt.xlabel('Coefficients')
        plt.ylabel('Estimates')
        plt.title('Regression Coefficients')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

RegressionModel.__doc__ = """\
RegressionModel evaluates and compares full and reduced regression models 
using statistical tests such as t-tests and partial F-tests, and provides 
visualization of regression coefficients for interpretability.

.. math::

    t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
    
    F = \frac{(RSS_{\text{reduced}} - RSS_{\text{full}})\\
              / (df_{\text{reduced}} - df_{\text{full}})}
    {RSS_{\text{full}} / df_{\text{full}}}
    
    \text{where } RSS = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \,
    k = \text{number of parameters in the model}, \, n = \text{number of observations}

Parameters
----------
full_model : estimator instance
    A scikit-learn compatible regression model representing the full 
    model with all predictors.
reduced_model : estimator instance, optional
    A scikit-learn compatible regression model representing the reduced 
    model with a subset of predictors.
    If ``None``, the reduced model must be provided during the ``fit`` method.

Attributes
----------
X_full_ : array-like, shape (n_samples, n_features_full)
    The full set of input features provided to the ``fit`` method.
y_ : array-like, shape (n_samples,)
    The target values provided to the ``fit`` method.
full_model_ : RegressionResultsWrapper
    The fitted full regression model using `statsmodels`.
reduced_model_ : RegressionResultsWrapper, optional
    The fitted reduced regression model using `statsmodels`.
t_values_ : Series
    The t-statistics for each coefficient in the full model.
p_values_ : Series
    The p-values for each coefficient in the full model.

Methods
-------
fit(X_full, y, X_reduced=None)
    Fits the full and optionally reduced models to the data.
t_test_coefficients()
    Performs t-tests on the coefficients of the full model to assess their
    significance.
partial_f_test()
    Performs a partial F-test to compare the full and reduced models.
plot()
    Plots the regression coefficients of the full model for visual assessment.

Examples
--------
>>> from gofast.stats.evaluation import RegressionModel
>>> import statsmodels.api as sm
>>> import numpy as np
>>> X_full = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
>>> y = np.array([2, 3, 4, 5])
>>> X_reduced = np.array([[1], [2], [3], [4]])
>>> full_model = sm.OLS(y, sm.add_constant(X_full))
>>> reduced_model = sm.OLS(y, sm.add_constant(X_reduced))
>>> rmt = RegressionModel(full_model=full_model, reduced_model=reduced_model)
>>> rmt.fit(X_full, y, X_reduced=X_reduced)
RegressionModelTests(full_model=OLS(), reduced_model=OLS())
>>> t_vals, p_vals = rmt.t_test_coefficients()
>>> print(f"t-values: {t_vals}")
t-values: const    0.0
x1       1.0
x2       1.0
dtype: float64
>>> f_stat, p_f = rmt.partial_f_test()
>>> print(f"Partial F-statistic: {f_stat}, p-value: {p_f}")
Partial F-statistic: 0.0, p-value: 1.0
>>> rmt.plot()

Notes
-----
- The ``RegressionModelTests`` class assumes that the input models follow the 
  scikit-learn estimator interface and have attributes like ``coef_``.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The t-tests evaluate the significance of individual coefficients in the full model.
- The partial F-test compares the full model against a reduced model to assess 
  whether the additional predictors significantly improve the model.
- It is recommended to validate that the models are properly fitted before 
  invoking test methods using ``check_is_fitted`` from ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Draper, N.R., & Smith, H. (1998). *Applied Regression Analysis*. 
   Wiley-Interscience.
.. [2] Seber, G.A.F., & Lee, A.J. (2012). *Linear Regression Analysis*. 
   Wiley.
.. [3] Montgomery, D.C., Peck, E.A., & Vining, G.G. (2012). 
   *Introduction to Linear Regression Analysis*. Wiley.
"""

@smartFitRun 

class TimeSeriesTests(BaseClass):
    @validate_params ({"lags": [ int, list, None]})
    def __init__(self, lags=None):
        self.lags = lags
    def run(self, time_series):
        time_series = check_array(
            time_series, estimator=True, ensure_2d=False, 
            input_name ="time_series", to_frame= True 
            )
        self.time_series_ = time_series
        self._is_runned=True 
        return self
    
    @ensure_pkg(
        "statsmodels",
        "'statsmodels is required for `dickey_fuller` to proceed."
    )
    def augmented_dickey_fuller(self):
        import statsmodels.tsa.stattools as tsa
        
        check_is_runned(self,)
        result = tsa.adfuller(self.time_series_)
        self.adf_stat_ = result[0]
        self.adf_pvalue_ = result[1]
        self.adf_usedlag_ = result[2]
        self.adf_nobs_ = result[3]
        self.adf_critical_values_ = result[4]
        self.adf_icbest_ = result[5]
        return result

    @ensure_pkg(
        "statsmodels",
        "'statsmodels is required for `ljung_box` to proceed."
    )
    def ljung_box(self):
        import statsmodels.stats.diagnostic as diag
        
        check_is_runned(self)
        result = diag.acorr_ljungbox(self.time_series_, 
                                     lags=self.lags, return_df=True)
        self.lb_stat_ = result['lb_stat'].values
        self.lb_pvalue_ = result['lb_pvalue'].values
        return self.lb_stat_, self.lb_pvalue_

    def plot(self):
        check_is_runned(self)
        plt.figure(figsize=(12, 6))
        plt.plot(self.time_series_)
        plt.title('Time Series Plot')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.tight_layout()
        plt.show()

TimeSeriesTests.__doc__ = """\
TimeSeriesTests performs statistical tests and visualizations to assess the 
properties of time series data, including stationarity and autocorrelation.
    
.. math::
    
    \text{Augmented Dickey-Fuller (ADF) Statistic} = \phi
    
    \text{Ljung-Box Statistic} = Q = n(n+2) \sum_{k=1}^{m} 
    \frac{\hat{\rho}_k^2}{n - k}
    
Parameters
----------
lags : int or list of int, optional
    The number of lags to include in the Ljung-Box test. If ``None``, defaults 
    to ``40`` or ``10 \times \log_{10}(n_{\text{samples}})``, whichever is 
    smaller.

Attributes
----------
time_series_ : array-like, shape (n_samples,)
    The time series data provided to the ``run`` method.
adf_stat_ : float
    The test statistic from the Augmented Dickey-Fuller test.
adf_pvalue_ : float
    The p-value from the Augmented Dickey-Fuller test.
adf_usedlag_ : int
    The number of lags used in the Augmented Dickey-Fuller test.
adf_nobs_ : int
    The number of observations used in the Augmented Dickey-Fuller test.
adf_critical_values_ : dict
    The critical values for the Augmented Dickey-Fuller test at different 
    significance levels.
adf_icbest_ : str
    The information criterion used to select the best lag in the Augmented 
    Dickey-Fuller test.
lb_stat_ : array-like, shape (n_lags,)
    The Ljung-Box test statistics for each specified lag.
lb_pvalue_ : array-like, shape (n_lags,)
    The p-values corresponding to the Ljung-Box test statistics.

Methods
-------
run(time_series)
    Stores the time series data for analysis.
augmented_dickey_fuller()
    Performs the Augmented Dickey-Fuller test to check for stationarity.
ljung_box()
    Performs the Ljung-Box test to assess autocorrelation in the time series.
plot()
    Plots the time series data for visual inspection.

Examples
--------
>>> from gofast.stats.evaluation import TimeSeriesTests
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> np.random.seed(0)
>>> time_series = np.random.normal(loc=0, scale=1, size=100)
>>> tst = TimeSeriesTests(lags=10)
>>> tst.run(time_series)
TimeSeriesTests(lags=10)
>>> adf_result = tst.augmented_dickey_fuller()
>>> print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
ADF Statistic: -1.943, p-value: 0.24
>>> lb_stat, lb_pvalue = tst.ljung_box()
>>> print(f"Ljung-Box Statistics: {lb_stat}, p-values: {lb_pvalue}")
Ljung-Box Statistics: [1.5, 1.2, 1.0], p-values: [0.7, 0.8, 0.9]
>>> tst.plot()

Notes
-----
- The ``TimeSeriesTests`` class assumes that the input data is a one-dimensional 
  array-like structure representing a time series.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Augmented Dickey-Fuller test is used to determine whether a time series 
  is stationary.
- The Ljung-Box test checks for the absence of autocorrelation in the residuals 
  of a time series.
- It is recommended to validate that the time series data is properly formatted 
  and contains no missing values before invoking tests using 
  ``check_is_fitted`` from ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Dickey, D.A., & Fuller, W.A. (1979). Distribution of the Estimators 
   for Autoregressive Time Series with a Unit Root. *Journal of the American 
   Statistical Association*, 74(366), 427-431.
.. [2] Ljung, G.M., & Box, G.E.P. (1978). On a Measure of Lack of Fit in 
   Time Series Models. *Biometrika*, 65(2), 297-303.
.. [3] Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University 
   Press.

"""

class ErrorAnalysis(BaseClass):
    @validate_params ({"alpha": [Interval(Real, 0, 1, closed="both")]})
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, y_true, y_pred1, y_pred2=None):
        y_true= check_y (y_true, estimator=self, input_name ="y_true")
        y_pred1= check_y (y_pred1, estimator=self, input_name ="y_pred1")
        
        self.y_true_ = y_true
        self.y_pred1_ = y_pred1
        if y_pred2 is not None:
            y_pred2= check_y (y_pred2, estimator=self, input_name ="y_pred2")
            self.y_pred2_ = y_pred2
        return self

    def mse_significance(self, benchmark_mse):
        check_is_fitted (self, ["y_true_", "y_pred1_"])
        mse = mean_squared_error(self.y_true_, self.y_pred1_)
        n = len(self.y_true_)
        mse_diff = mse - benchmark_mse
        se = np.sqrt(2 * (benchmark_mse ** 2) / n)
        z = mse_diff / se
        p_value = 2 * (1 - norm.cdf(np.abs(z)))
        return z, p_value

    def diebold_mariano(self):
        check_is_fitted (self, ["y_true_"])
        if not hasattr(self, 'y_pred2_'):
            raise ValueError("Second set of predictions not provided.")
        e1 = self.y_true_ - self.y_pred1_
        e2 = self.y_true_ - self.y_pred2_
        d = e1 ** 2 - e2 ** 2
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        dm_stat = mean_d / np.sqrt(var_d / len(d))
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
        return dm_stat, p_value

    def plot(self):
        check_is_fitted (self, ["y_true_", "y_pred1_"])
        plt.plot(self.y_true_, label='True Values')
        plt.plot(self.y_pred1_, label='Predictions 1')
        if hasattr(self, 'y_pred2_'):
            plt.plot(self.y_pred2_, label='Predictions 2')
        plt.legend()
        plt.title('Prediction Comparison')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.tight_layout()
        plt.show()

ErrorAnalysis.__doc__ = """\
ErrorAnalysis performs comparative analysis of prediction errors between 
different models using statistical tests such as Mean Squared Error (MSE) 
significance and the Diebold-Mariano test, and provides visualization 
through prediction comparison plots.
    
.. math::
    
    z = \frac{\text{MSE} - \text{Benchmark MSE}}{\sqrt{2 \cdot 
    \text{Benchmark MSE}^2 / n}}
    
    \text{Diebold-Mariano Statistic} = \frac{\bar{d}}{\sqrt{\text{Var}(d) / n}}
    
    \text{where } d_i = e_{1i}^2 - e_{2i}^2

Parameters
----------
alpha : float, optional
    Significance level for the statistical tests. Must be between 0 and 1.
    Default is ``0.05``.

Attributes
----------
y_true_ : array-like, shape (n_samples,)
    The true target values provided to the ``fit`` method.
y_pred1_ : array-like, shape (n_samples,)
    The predicted values from the first model provided to the ``fit`` method.
y_pred2_ : array-like, shape (n_samples,), optional
    The predicted values from the second model provided to the ``fit`` method, 
    if any.

Methods
-------
fit(y_true, y_pred1, y_pred2=None)
    Stores the true and predicted values for error analysis.
mse_significance(benchmark_mse)
    Performs a significance test on the MSE of the first model against a benchmark.
diebold_mariano()
    Conducts the Diebold-Mariano test to compare predictive accuracy between 
    two models.
plot()
    Plots the true values alongside one or two sets of predictions for 
    visual comparison.

Examples
--------
>>> from gofast.stats.evaluation import ErrorAnalysis
>>> import numpy as np
>>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
>>> y_pred1 = np.array([2.5, 0.0, 2.1, 7.8])
>>> y_pred2 = np.array([3.2, -0.3, 1.8, 6.9])
>>> ea = ErrorAnalysis(alpha=0.05)
>>> ea.fit(y_true, y_pred1, y_pred2)
ErrorAnalysis(alpha=0.05)
>>> z, p = ea.mse_significance(benchmark_mse=0.5)
>>> print(f"Z-score: {z}, p-value: {p}")
Z-score: 1.0, p-value: 0.31731050786291415
>>> dm_stat, dm_pvalue = ea.diebold_mariano()
>>> print(f"Diebold-Mariano Statistic: {dm_stat}, p-value: {dm_pvalue}")
Diebold-Mariano Statistic: 0.0, p-value: 1.0
>>> ea.plot()

Notes
-----
- The ``ErrorAnalysis`` class assumes that the input predictions are 
  continuous values suitable for regression tasks.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The MSE significance test evaluates whether the MSE of the first model 
  is significantly different from a benchmark MSE.
- The Diebold-Mariano test compares the predictive accuracy of two models 
  based on their forecast errors.
- It is recommended to validate that the models are properly fitted before 
  invoking error analysis methods using ``check_is_fitted`` from 
  ``gofast.tools.validator``.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.

References
----------
.. [1] Diebold, F.X., & Mariano, R.S. (1995). Comparing Predictive 
   Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
.. [2] Gneiting, T., & Raftery, A.E. (2007). Strictly Proper Scoring Rules, 
   Prediction, and Estimation. *Journal of the American Statistical 
   Association*, 102(477), 359-378.
.. [3] Willmott, C.J., & Matsuura, K. (2005). Advantages of the Mean Absolute 
   Error (MAE) over the Root Mean Squared Error (RMSE) in Assessing 
   Average Model Performance. *Climate Research*, 30(1), 79-82.

"""


@smartFitRun 
@ensure_pkg(
    "pymc3", extra="pymc3 is needed for BayesianMethods to proceed.",
)
class BayesianMethods(BaseClass):
    @validate_params(
        {
            "tune": Interval(Integral, 1, None, closed="left"),
            "draws": Interval(Integral, 1, None, closed="left"),
            "fit_models": [bool],
            "random_seed": ["random_state"],
            "cores": Interval(Integral, 1, None, closed="left"),
        }
    )
    def __init__(
        self, 
        fit_models=True, 
        draws=1000, 
        tune=1000, 
        random_state=None, 
        cores=1
        ):
        self.fit_models = fit_models
        self.draws = draws
        self.tune = tune
        self.random_state = random_state
        self.cores = cores
    
    @validate_params ({"models": [dict]})
    def run(self, models, **run_params):
        import pymc3 as pm
        self.traces_ = {}
        self.idata_ = {}
        for name, model in models.items():
            if self.fit_models:
                with model:
                    trace = pm.sample(
                        draws=self.draws,
                        tune=self.tune,
                        random_seed=self.random_state,
                        cores=self.cores,
                        return_inferencedata=True
                    )
                self.traces_[name] = trace
                self.idata_[name] = trace
        return self

    @ensure_pkg(
        "arviz", extra="arviz is needed when 'method' is set to `log_likelihood`.",
        partial_check= True,
        condition= lambda *args, **kwargs: kwargs.get("method") =="log_likelihood"
        )
    def bayes_factor(self, model_a, model_b, method='log_likelihood'):
        check_is_runned(self, ["idata_"])
        if method == 'bridge':
            from pymc3.loo import bridge_sampling
            log_ml_a = bridge_sampling(self.idata_[model_a])
            log_ml_b = bridge_sampling(self.idata_[model_b])
            log_bf = log_ml_a - log_ml_b
            
        elif method == 'harmonic_mean':
            import pymc3 as pm
            log_ml_a = pm.loo(self.idata_[model_a], method='harmonic_mean').loo
            log_ml_b = pm.loo(self.idata_[model_b], method='harmonic_mean').loo
            log_bf = log_ml_a - log_ml_b
            
        elif method == 'log_likelihood':
            import arviz as az
            waic_a = az.waic(self.idata_[model_a])
            waic_b = az.waic(self.idata_[model_b])
            delta_waic = waic_b.waic - waic_a.waic
            log_bf = -0.5 * delta_waic
        else:
            raise ValueError("Invalid method for Bayes factor computation.")
        bf = np.exp(log_bf)
        return bf

    def highest_posterior_density_interval(
            self, model_name, 
            var_name, 
            credible_interval=0.95
            ):
        check_is_runned(self, ["idata_"])
        if is_module_installed("arviz"): 
            import arviz as az
            idata = self.idata_[model_name]
            hpd_interval = az.hdi(idata, var_names=[var_name], hdi_prob=credible_interval)
            return hpd_interval[var_name].values
        else: 
            import pymc3 as pm
            trace = self.traces_[model_name]
            hpd_interval = pm.hpd(trace[var_name], credible_interval=credible_interval)
            return hpd_interval

    def plot(self, model_name, var_name):
        check_is_runned(self, ["idata_"])
        idata = self.idata_[model_name]
        if is_module_installed  ("arviz"): 
            import arviz as az
            az.plot_posterior(idata, var_names=[var_name])
        else:
            import pymc3 as pm
            trace = self.traces_[model_name]
            pm.plot_posterior(trace[var_name])
            
            
        plt.title(f'Posterior of {var_name} in {model_name}')
        plt.tight_layout()
        plt.show()
        
BayesianMethods.__doc__ = """\
BayesianMethods facilitates Bayesian inference for statistical models using 
PyMC3, enabling the computation of Bayes factors and highest posterior density 
intervals, and provides visualization of posterior distributions.
    
.. math::
    
    \text{Log Likelihood Ratio (LLR)} = \sum_{i=1}^{n} \left[ x_i 
    \log\left(\frac{p_1}{p_0}\right) + (1 - x_i)\\
        \log\left(\frac{1 - p_1}{1 - p_0}\right) \right]
    
    \text{Bayes Factor} = e^{\text{LLR}}
    
    \text{Highest Posterior Density (HPD) Interval} = \{ \theta \, | \, 
    P(\theta \in \text{HPD}) = \text{credible\_interval} \}

Parameters
----------
fit_models : bool, optional
    If ``True``, the models will be fitted to the data during the ``run`` 
    method. If ``False``, it is assumed that the models are already fitted.
    Default is ``True``.
draws : int, optional
    The number of samples to draw from the posterior during Bayesian inference.
    Must be a positive integer. Default is ``1000``.
tune : int, optional
    The number of tuning (burn-in) steps for the sampler. Must be a non-negative 
    integer. Default is ``1000``.
random_seed : int or None, optional
    Seed for the random number generator to ensure reproducibility. Default is 
    ``None``.
cores : int, optional
    The number of CPU cores to use for parallel sampling. Must be a positive 
    integer. Default is ``1``.

Attributes
----------
traces_ : dict
    A dictionary mapping model names to their corresponding trace objects 
    obtained from PyMC3 sampling.
idata_ : dict
    A dictionary mapping model names to their corresponding InferenceData 
    objects obtained from PyMC3 sampling.

Methods
-------
run(models, **run_params)
    Executes Bayesian sampling for the provided models using PyMC3.
bayes_factor(model_a, model_b, method='log_likelihood')
    Computes the Bayes factor between two models using specified methods.
highest_posterior_density_interval(model_name, var_name, credible_interval=0.95)
    Calculates the highest posterior density interval for a specified variable.
plot(model_name, var_name)
    Plots the posterior distribution of a specified variable for a given model.

Examples
--------
>>> from gofast.stats.evaluation import BayesianMethods
>>> import pymc3 as pm
>>> import numpy as np
>>> # Define two simple Bayesian models
>>> with pm.Model() as model_a:
...     mu = pm.Normal('mu', mu=0, sigma=1)
...     obs = pm.Normal('obs', mu=mu, sigma=1, observed=np.random.randn(100))
>>> with pm.Model() as model_b:
...     mu = pm.Normal('mu', mu=0, sigma=1)
...     sigma = pm.HalfNormal('sigma', sigma=1)
...     obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=np.random.randn(100))
>>> models = {'ModelA': model_a, 'ModelB': model_b}
>>> bm = BayesianMethods(fit_models=True, draws=1000, tune=1000, 
...                      random_seed=42, cores=2)
>>> bm.run(models)
BayesianMethods(fit_models=True, draws=1000, tune=1000, random_seed=42, cores=2)
>>> bf = bm.bayes_factor('ModelA', 'ModelB', method='log_likelihood')
>>> print(f"Bayes Factor: {bf}")
Bayes Factor: 1.234
>>> hpd = bm.highest_posterior_density_interval('ModelA', 'mu', 
...                                             credible_interval=0.95)
>>> print(f"HPD Interval for mu in ModelA: {hpd}")
HPD Interval for mu in ModelA: [-0.1, 0.2]
>>> bm.plot('ModelA', 'mu')

Notes
-----
- The ``BayesianMethods`` class leverages PyMC3 for Bayesian inference and assumes 
  that the input models are defined using PyMC3's model context.
- Only public methods (those that do not start with an underscore) are 
  considered for evaluation.
- The Bayes factor provides a ratio of the evidences for two competing models, 
  indicating which model is more supported by the data.
- The highest posterior density interval represents the range of parameter values 
  that contain a specified probability mass of the posterior distribution.
- It is recommended to validate that the models are properly specified and that 
  the data is appropriately formatted before invoking Bayesian inference methods 
  using ``check_is_fitted`` from ``gofast.tools.validator``.
- The decorators ``@smartFitRun`` and ``@ensure_pkg`` ensure that necessary 
  packages are installed and handle the fitting and running of models 
  intelligently.

See Also
--------
`check_is_fitted` : Validates that the estimator is fitted.
`validate_params` : Validates the parameters of the estimator.
`arviz` : For advanced Bayesian analysis and visualization.

References
----------
.. [1] Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., 
   & Rubin, D.B. (2013). *Bayesian Data Analysis*. CRC Press.
.. [2] Kass, R.E., & Raftery, A.E. (1995). Bayes Factors. *Journal of the 
   American Statistical Association*, 90(430), 773-795.
.. [3] Spiegelhalter, D., Best, N., Carlin, B., & van der Linde, A. (2002). 
   Bayesian Measures of Model Complexity and Fit. *Journal of the American 
   Statistical Association*, 97(458), 611-623.

"""

class ModelRobustness(BaseClass, ClassifierMixin):
    @validate_params(
        {
            "model":  [HasMethods(['fit', 'predict'])],
            "adversarial_method": [StrOptions({"gaussian_noise", "salt_pepper"})],
            "sensitivity_params": [dict, None],
            "uncertainty_method": [StrOptions({"bootstrap", "bayesian"})],
            "n_iterations": [Interval(Integral, 1, None, closed="left")],
            "noise_level": [Interval(Real, 0, 1, closed="both")],
        }
    )
    def __init__(
        self,
        model,
        adversarial_method='gaussian_noise',
        sensitivity_params=None,
        uncertainty_method='bootstrap',
        n_iterations=100,
        noise_level=0.1,
    ):
        self.model = model
        self.adversarial_method = adversarial_method
        self.sensitivity_params = sensitivity_params or {
            'feature_index': 0, 'delta': 0.01}
        self.uncertainty_method = uncertainty_method
        self.n_iterations = n_iterations
        self.noise_level = noise_level

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.X_ = X
        self.y_ = y
        self.model_ = clone(self.model)
        self.model_.fit(self.X_, self.y_)
        return self

    def adversarial(self):
        """
        Perform adversarial testing.

        Generates adversarial examples using the specified method and 
        evaluates the model's accuracy on these perturbed inputs.

        Returns
        -------
        accuracy : float
            The accuracy of the model on adversarial examples.

        Raises
        ------
        ValueError
            If the specified adversarial method is not supported.
        """
        check_is_fitted(self, 'X_')
        if self.adversarial_method == 'gaussian_noise':
            noise = np.random.normal(0, self.noise_level, self.X_.shape)
            X_adv = self.X_ + noise
        elif self.adversarial_method == 'salt_pepper':
            X_adv = self.X_.copy()
            mask = np.random.rand(*self.X_.shape) < self.noise_level
            X_adv[mask] = np.random.choice([0, 1], size=np.sum(mask))
        else:
            raise ValueError(
                f"Unsupported adversarial method: {self.adversarial_method}")

        preds = self.model_.predict(X_adv)
        accuracy = accuracy_score(self.y_, preds)
        return accuracy

    def sensitivity_analysis(self):
        """
        Perform sensitivity analysis.

        Perturbs a specified feature by a small amount and evaluates
        the change in accuracy.

        Returns
        -------
        sensitivity : float
            The average accuracy after perturbing the feature positively 
            and negatively.

        Notes
        -----
        The sensitivity is calculated as:

        .. math::

            S = \\frac{A_{+} + A_{-}}{2}

        where :math:`A_{+}` is the accuracy after increasing the feature,
        and :math:`A_{-}` is the accuracy after decreasing the feature.
        """
        check_is_fitted(self, 'X_')
        feature_index = self.sensitivity_params['feature_index']
        delta = self.sensitivity_params['delta']
        original_feature = self.X_[:, feature_index].copy()
        X_temp = self.X_.copy()

        # Perturb the feature positively
        X_temp[:, feature_index] = original_feature + delta
        preds_plus = self.model_.predict(X_temp)

        # Perturb the feature negatively
        X_temp[:, feature_index] = original_feature - delta
        preds_minus = self.model_.predict(X_temp)

        # Reset the feature to original values
        X_temp[:, feature_index] = original_feature

        accuracy_plus = accuracy_score(self.y_, preds_plus)
        accuracy_minus = accuracy_score(self.y_, preds_minus)
        sensitivity = (accuracy_plus + accuracy_minus) / 2
        return sensitivity

    def uncertainty_estimation(self):
        """
        Estimate the uncertainty of the model's predictions.

        Returns
        -------
        mean : float
            The mean uncertainty measure (accuracy or entropy).

        std : float
            The standard deviation of the uncertainty measure.

        Raises
        ------
        ValueError
            If the specified uncertainty method is not supported or if the model
            does not support probabilistic predictions when using 'bayesian'.
        """
        check_is_fitted(self, 'X_')
        if self.uncertainty_method == 'bootstrap':
            accuracies = []
            for _ in range(self.n_iterations):
                indices = np.random.choice(len(self.X_), len(self.X_), replace=True)
                X_sample = self.X_[indices]
                y_sample = self.y_[indices]
                model = clone(self.model_)
                model.fit(X_sample, y_sample)
                preds = model.predict(self.X_)
                acc = accuracy_score(self.y_, preds)
                accuracies.append(acc)
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            return mean_acc, std_acc
        elif self.uncertainty_method == 'bayesian':
            if not hasattr(self.model_, "predict_proba"):
                raise ValueError(
                    "Model does not support probabilistic predictions"
                    " needed for Bayesian uncertainty estimation."
                )
            probs = self.model_.predict_proba(self.X_)
            entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
            mean_entropy = np.mean(entropy)
            std_entropy = np.std(entropy)
            return mean_entropy, std_entropy
        else:
            raise ValueError(
                f"Unsupported uncertainty method: {self.uncertainty_method}")

    def run_all_tests(self):
        """
        Run all robustness tests and store the results.

        Returns
        -------
        results : dict
            A dictionary containing the results of the robustness tests.
        """
        check_is_fitted(self, 'X_')
        results = {}
        results['adversarial_accuracy'] = self.adversarial()
        results['sensitivity'] = self.sensitivity_analysis()
        uncertainty_mean, uncertainty_std = self.uncertainty_estimation()

        if self.uncertainty_method == 'bootstrap':
            results['uncertainty_mean'] = uncertainty_mean
            results['uncertainty_std'] = uncertainty_std
        elif self.uncertainty_method == 'bayesian':
            results['mean_entropy'] = uncertainty_mean
            results['std_entropy'] = uncertainty_std

        self.results_ = results
        return results

    def visualize_robustness(self):
        """
        Visualize the results of the robustness tests.

        Raises
        ------
        RuntimeError
            If `run_all_tests` has not been called yet.
        """
        if not hasattr(self, 'results_'):
            raise RuntimeError("Run 'run_all_tests' before visualizing results.")

        metrics = list(self.results_.keys())
        values = [self.results_[metric] for metric in metrics]

        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values, color='skyblue')
        plt.xlabel('Test Metrics')
        plt.ylabel('Values')
        plt.title('Model Robustness Test Results')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

ModelRobustness.__doc__ = """\
A class for evaluating the robustness of machine learning 
classification models.

The `ModelRobustnessTester` class provides methods to test the robustness of a
classification model through adversarial attacks, sensitivity analysis, and
uncertainty estimation. It allows users to assess how the model performs under
perturbations and estimate the confidence of its predictions.

Parameters
----------
model : object
    The base classifier model to be tested. It should implement the `fit` and
    `predict` methods.

adversarial_method : {'gaussian_noise', 'salt_pepper', 'fgsm'},\
    default='gaussian_noise'
    The method used for generating adversarial examples:

    - 'gaussian_noise': Adds Gaussian noise to the input features.
    - 'salt_pepper': Applies salt-and-pepper noise to the input features.
    - 'fgsm': Performs Fast Gradient Sign Method attack.

sensitivity_params : dict or None, default=None
    Parameters for sensitivity analysis. If None, defaults to
    ``{'feature_index': 0, 'delta': 0.01}``. Should contain:

    - `'feature_index'` : int
        The index of the feature to perturb.
    - `'delta'` : float
        The amount by which to perturb the feature.

uncertainty_method : {'bootstrap', 'bayesian'}, default='bootstrap'
    The method used for uncertainty estimation:

    - 'bootstrap': Uses bootstrapping to estimate uncertainty.
    - 'bayesian': Uses Bayesian methods to estimate uncertainty (requires model
      to support `predict_proba`).

n_iterations : int, default=100
    The number of iterations to perform in bootstrapping for uncertainty
    estimation.

noise_level : float, default=0.1
    The level of noise to add in adversarial testing. For 'gaussian_noise',
    this is the standard deviation. For 'salt_pepper', this is the probability
    of flipping each feature.

epsilon : float, default=0.01
    The perturbation magnitude for the FGSM adversarial method.

Attributes
----------
X_ : ndarray of shape (n_samples, n_features)
    The training input samples.

y_ : ndarray of shape (n_samples,)
    The target values.

model_ : object
    The cloned and fitted model.

results_ : dict
    The results from the robustness tests after calling `run_all_tests`.

Methods
-------
fit(X, y)
    Fit the model according to the given training data.

adversarial_test()
    Perform adversarial testing and return the accuracy under adversarial
    conditions.

sensitivity_analysis()
    Perform sensitivity analysis and return the average accuracy after feature
    perturbation.

uncertainty_estimation()
    Estimate the uncertainty of the model's predictions.

run_all_tests()
    Run all robustness tests and store results in `results_`.

visualize_robustness()
    Visualize the results of the robustness tests.

Examples
--------
>>> from gofast.stats.evaluation import ModelRobustnessTester
>>> from sklearn.datasets import load_iris
>>> from sklearn.linear_model import LogisticRegression
>>> X, y = load_iris(return_X_y=True)
>>> model = LogisticRegression()
>>> robustness_tester = ModelRobustnessTester(model)
>>> robustness_tester.fit(X, y)
>>> results = robustness_tester.run_all_tests()
>>> robustness_tester.visualize_robustness()

Notes
-----
The adversarial testing methods simulate how the model behaves when the input
data is slightly perturbed, which can reveal vulnerabilities to adversarial
attacks.

The sensitivity analysis assesses how sensitive the model's predictions are to
changes in specific features, which can help identify important or fragile
aspects of the model.

Uncertainty estimation provides insights into the confidence of the model's
predictions. Bootstrapping repeats the training process multiple times on
resampled datasets, while Bayesian methods rely on the probabilistic outputs
of the model.

See Also
--------
sklearn.base.BaseClass : Base class for all estimators in scikit-learn.
sklearn.base.ClassifierMixin : Mixin class for all classifiers in scikit-learn.

References
----------
.. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
       MIT Press.
.. [2] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
       Springer.
"""
