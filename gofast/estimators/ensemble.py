# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
`ensemble` implements various ensemble methods for classification
and regression tasks for combining the predictions of multiple base estimators
to improve generalizability and  robustness over a single estimator.
"""

from __future__ import annotations 
import re  
import numpy as np
from tqdm import tqdm 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

from ..tools.validator import check_X_y, check_array 
from ..tools.validator import check_is_fitted, parameter_validator 
from ..tools.validator import validate_fit_weights
from ._ensemble import BaseEnsemble
from .util import fit_with_estimator, determine_weights, apply_scaling 
from .util import optimize_hyperparams, normalize_sum 

__all__=[
      "MajorityVoteClassifier", 
      "SimpleAverageRegressor",
      "SimpleAverageClassifier", 
      "WeightedAverageRegressor", 
      "WeightedAverageClassifier",
      "EnsembleClassifier", 
      "EnsembleRegressor"
  ]

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    A majority vote ensemble classifier.

    This classifier combines different classification algorithms, each
    associated with individual weights for confidence. The aim is to create
    a stronger meta-classifier that balances out individual classifiers'
    weaknesses on specific datasets. The majority vote considers the weighted
    contribution of each classifier in making the final decision.

    Mathematically, the weighted majority vote is expressed as:

    .. math::

        \hat{y} = \arg\max_{i} \sum_{j=1}^{m} weights_j \chi_{A}(C_j(x)=i)

    Here, :math:`weights_j` is the weight associated with the base classifier
    :math:`C_j`; :math:`\hat{y}` is the predicted class label by the ensemble;
    :math:`A` is the set of unique class labels; :math:`\chi_A` is the
    characteristic function or indicator function, returning 1 if the predicted
    class of the j-th classifier matches :math:`i`. For equal weights, the
    equation simplifies to:

    .. math::

        \hat{y} = \text{mode} \{ C_1(x), C_2(x), \ldots, C_m(x) \}

    Parameters
    ----------
    classifiers : array-like, shape (n_classifiers,)
        Different classifiers for the ensemble. Each classifier should be a
        scikit-learn compatible estimator.

    weights : array-like, shape (n_classifiers,), optional, default=None
        Weights for each classifier. Uniform weights are used if `weights` 
        is None.

    vote : str, {'classlabel', 'probability'}, default='classlabel'
        If 'classlabel', prediction is based on the argmax of class labels.
        If 'probability', prediction is based on the argmax of the sum of
        probabilities. Recommended for calibrated classifiers.

    verbose : bool, default=False
        If True, will print progress messages and use tqdm for progress display.

    random_state : int, RandomState instance or None, optional, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.

    class_weight : dict, list of dicts, 'balanced', or None, optional, default=None
        Weights associated with classes. If not given, all classes are supposed
        to have weight one. If 'balanced', class weights will be adjusted inversely
        proportional to class frequencies in the input data.

    tie_breaking_strategy : str, {'random', 'first'}, default='random'
        Strategy to handle ties in classlabel voting. 'random' will randomly
        choose among the tied classes, 'first' will choose the first occurrence.

    classifier_params : dict, default=None
        Optional parameter settings for the classifiers. If given, each key
        should be a classifier name followed by double underscores and the
        parameter name (e.g., 'clf__C').

    Attributes
    ----------
    classes_ : array-like, shape (n_classes,)
        Unique class labels.

    classifiers_ : list
        List of fitted classifiers.
    
    Examples 
    ---------
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.pipeline import Pipeline 
    >>> from sklearn.model_selection import cross_val_score, train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.estimators.ensemble import MajorityVoteClassifier 
    >>> from gofast.tools.coreutils import select_features 
    >>> data = fetch_data('bagoue original').frame
    >>> X0 = data.iloc [:, :-1]; y0 = data ['flow'].values  
    >>> # exclude the categorical value for demonstration 
    >>> # binarize the target y 
    >>> y = np.asarray (list(map (lambda x: 0 if x<=1 else 1, y0))) 
    >>> X = selectfeatures (X0, include ='number')
    >>> X = SimpleImputer().fit_transform (X) 
    >>> X, Xt , y, yt = train_test_split(X, y)
    >>> clf1 = LogisticRegression(penalty ='l2', solver ='lbfgs') 
    >>> clf2= DecisionTreeClassifier(max_depth =1 ) 
    >>> clf3 = KNeighborsClassifier( p =2 , n_neighbors=1) 
    >>> pipe1 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf1)])
    >>> pipe3 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf3)])
    
    (1) -> Test the each classifier results taking individually 
    
    >>> clf_labels =['Logit', 'DTC', 'KNN']
    >>> # test the results without using the MajorityVoteClassifier
    >>> for clf , label in zip ([pipe1, clf2, pipe3], clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), 
                                                     label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
    
    (2) -> Implement the MajorityVoteClassifier
    
    >>> # test the resuls with Majority vote  
    >>> mv_clf = MajorityVoteClassifier(classifiers = [pipe1, clf2, pipe3])
    >>> clf_labels += ['Majority voting']
    >>> all_classifiers = [pipe1, clf2, pipe3, mv_clf]
    >>> for clf , label in zip (all_classifiers, clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
        ROC AUC: 0.92 (+/- 0.06) [Majority voting] # give good score & less errors 
        
    
    Notes
    -----
    This classifier assumes that all base classifiers are capable of producing
    probability estimates if `vote` is set to 'probability'. In case of a tie in
    'classlabel' voting, the class label is determined based on the specified
    tie-breaking strategy.

    See Also
    --------
    VotingClassifier : Ensemble voting classifier in scikit-learn.
    BaggingClassifier : A Bagging ensemble classifier.
    RandomForestClassifier : A random forest classifier.

    References
    ----------
    .. [1] Breiman, L. (1996). Stacked Regressions. Machine Learning. 24(1):49-64.
    """
    def __init__(
        self, 
        classifiers, 
        weights=None, 
        vote='classlabel', 
        random_state=None, 
        class_weight=None, 
        tie_breaking_strategy='random', 
        classifier_params=None, 
        verbose=False, 
        ):
        self.classifiers = classifiers
        self.weights = weights
        self.vote = vote
        self.random_state = random_state
        self.class_weight = class_weight
        self.tie_breaking_strategy = tie_breaking_strategy
        self.classifier_params = ( classifier_params if classifier_params 
                                  is not None else {}
                                  )
        self.verbose = verbose
        self.classifier_names_ = {}
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the majority vote classifier.
    
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples. It is the observed data at training and
            prediction time, used as independent variables in learning.
        y : array-like of shape (n_samples,)
            The target values. It is the dependent variable in learning, usually
            the target of prediction.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
    
        Returns
        -------
        self : MajorityVoteClassifier instance
            Returns self for easy method chaining.
    
        Notes
        -----
        This method fits the base classifiers on the training data and the
        corresponding labels. If `sample_weight` is provided, it will be passed
        to the `fit` method of each base classifier.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import MajorityVoteClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        >>> clf1 = LogisticRegression()
        >>> clf2 = DecisionTreeClassifier()
        >>> clf3 = KNeighborsClassifier()
        >>> pipe1 = Pipeline([('sc', StandardScaler()), ('clf', clf1)])
        >>> pipe3 = Pipeline([('sc', StandardScaler()), ('clf', clf3)])
        >>> mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
        >>> mv_clf.fit(X_train, y_train)
        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__)
        self._check_classifiers_vote_and_weights()

        # Use label encoder to ensure that class labels start at 0
        self._labenc = LabelEncoder()
        self._labenc.fit(y)
        self.classes_ = self._labenc.classes_

        # Initialize random state for reproducibility
        self.random_state_ = check_random_state(self.random_state)

        self.classifiers_ = []
        
        if self.verbose:
            progress_bar = tqdm(total=len(self.classifiers), ascii=True, ncols=100, 
                                desc=f'Fitting {self.__class__.__name__}', )
            for name, clf in zip(
                    self.classifier_names_.keys(), self.classifiers):
                params = self.classifier_params.get(name, {})
                fitted_clf = clone(clf).set_params(**params)
                fitted_clf = fit_with_estimator(
                    fitted_clf, X, self._labenc.transform(y),
                    sample_weight=sample_weight 
                    )
                self.classifiers_.append(fitted_clf)
                progress_bar.update(1)
                
            progress_bar.close() 
        else:
            for name, clf in zip(self.classifier_names_.keys(), self.classifiers):
                params = self.classifier_params.get(name, {})
                fitted_clf = clone(clf).set_params(**params)
                fitted_clf = fit_with_estimator(
                    fitted_clf,  X, self._labenc.transform(y), 
                    sample_weight=sample_weight
                    )
                self.classifiers_.append(fitted_clf)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in `X`.
    
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        maj_vote : array-like of shape (n_samples,)
            Predicted class labels.
    
        Notes
        -----
        This method uses the fitted base classifiers to predict the class labels
        for the input samples. The final prediction is based on the majority vote
        of the base classifiers' predictions. If `vote` is set to 'probability',
        the prediction is based on the argmax of the sum of predicted probabilities.
    
        Examples
        --------
        >>> from gofast.estimators.benchmark import MajorityVoteClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        >>> clf1 = LogisticRegression()
        >>> clf2 = DecisionTreeClassifier()
        >>> clf3 = KNeighborsClassifier()
        >>> pipe1 = Pipeline([('sc', StandardScaler()), ('clf', clf1)])
        >>> pipe3 = Pipeline([('sc', StandardScaler()), ('clf', clf3)])
        >>> mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
        >>> mv_clf.fit(X_train, y_train)
        >>> y_pred = mv_clf.predict(X_test)
        """
        check_is_fitted(self, 'classifiers_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True, 
                        input_name="X") 
        
        if self.vote == 'proba':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # Collect results from clf.predict
            preds = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                self._tie_breaking(np.bincount, self.random_state_),
                axis=1,
                arr=preds
            )
            maj_vote = self._labenc.inverse_transform(maj_vote)

        return maj_vote

    def _tie_breaking(self, bincount, random_state):
        """
        Handle tie-breaking in majority voting.
    
        Parameters
        ----------
        bincount : callable
            Function to count the occurrence of each class label.
        random_state : RandomState instance
            Random number generator for tie-breaking.
    
        Returns
        -------
        tie_breaking_func : callable
            Function to apply tie-breaking strategy.
    
        Notes
        -----
        This method defines a function that handles tie-breaking based on the
        specified strategy. If the strategy is 'random', ties are broken randomly.
        If the strategy is 'first', the first occurring class is selected.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import MajorityVoteClassifier
        >>> mv_clf = MajorityVoteClassifier(classifiers=[clf1, clf2, clf3],
                                            tie_breaking_strategy='random')
        >>> tie_breaking_func = mv_clf._tie_breaking(np.bincount, np.random.RandomState(1))
        >>> tie_breaking_func([1, 2, 2, 3])
        2
        """
        def tie_breaking_func(x):
            counts = bincount(x, weights=self.weights)
            max_count = np.max(counts)
            max_classes = np.where(counts == max_count)[0]
            
            self.tie_breaking_strategy = parameter_validator(
                "tie_breaking_strategy", target_strs= {"random", 'first'})( 
                    self.tie_breaking_strategy )
            if len(max_classes) == 1:
                return max_classes[0]
            elif self.tie_breaking_strategy == 'random':
                return random_state.choice(max_classes)
            elif self.tie_breaking_strategy=='first':
                return max_classes[0]
            
        return tie_breaking_func
    
    def predict_proba(self, X):
        """
        Predict class probabilities and return average probabilities.
    
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        avg_proba : array-like of shape (n_samples, n_classes)
            Weighted average probabilities for each class per example.
    
        Notes
        -----
        This method uses the fitted base classifiers to predict the class
        probabilities for the input samples. The final prediction is the weighted
        average of the predicted probabilities from each base classifier.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import MajorityVoteClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        >>> clf1 = LogisticRegression()
        >>> clf2 = DecisionTreeClassifier()
        >>> clf3 = KNeighborsClassifier()
        >>> pipe1 = Pipeline([('sc', StandardScaler()), ('clf', clf1)])
        >>> pipe3 = Pipeline([('sc', StandardScaler()), ('clf', clf3)])
        >>> mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
        >>> mv_clf.fit(X_train, y_train)
        >>> y_proba = mv_clf.predict_proba(X_test)
        """
        check_is_fitted(self, 'classifiers_')
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
    
        return avg_proba

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.classifier_names_.copy()
            for name, step in self.classifier_names_.items():
                for key, value in step.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value

        return out

    def _check_classifiers_vote_and_weights(self):
        """
        Check the classifiers, vote type, and classifier weights.
        """
        if self.classifiers is None:
            raise TypeError("Expect at least one classifier.")

        if hasattr(self.classifiers, '__class__') and hasattr(
                self.classifiers, '__dict__'):
            self.classifiers = [self.classifiers]

        s = set([(hasattr(o, '__class__') and hasattr(
            o, '__dict__')) for o in self.classifiers])

        if not list(s)[0] or len(s) != 1:
            raise TypeError(
                "Classifier should be a class object, not {0!r}. Please refer"
                " to Scikit-Convention to write your own estimator.".format(
                    'type(self.classifiers).__name__'
                )
            )

        self.classifier_names_ = {k: v for k, v in _name_estimators(self.classifiers)}

        regex = re.compile(r'(class|label|target)|(proba)')
        v = regex.search(self.vote)
        if v is None:
            raise ValueError(
                "Vote argument must be 'probability' or 'classlabel', got %r" % self.vote)

        if v is not None:
            if v.group(1) is not None:
                self.vote = 'label'
            elif v.group(2) is not None:
                self.vote = 'proba'

        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError("Number of classifiers must be consistent with the"
                             " weights. Got {0} and {1} respectively.".format(
                len(self.classifiers), len(self.weights)
            ))
            
class SimpleAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Simple Average Ensemble Regressor.

    This ensemble model performs regression tasks by averaging the predictions
    of multiple base regression models. Utilizing a simple average is an
    effective technique to reduce the variance of individual model predictions,
    thereby improving the overall robustness and accuracy of the ensemble
    predictions.

    Mathematical Formulation:
    The predictions of the Simple Average Ensemble are computed as the
    arithmetic mean of the predictions made by each base estimator:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{pred}_i}

    where:
    - :math:`N` is the number of base estimators.
    - :math:`y_{\text{pred}_i}` is the prediction of the \(i\)-th base estimator.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support `fit` and `predict` methods. These base estimators will be
        combined through averaging to form the ensemble model.

    normalize_predictions : bool, default=False
        If True, normalize predictions from each base estimator before averaging.
        This can be useful if the base estimators have different prediction scales.

    random_state : int, RandomState instance or None, optional, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.

    verbose : bool, default=False
        If True, will print progress messages and use `tqdm` for progress display
        during fitting.

    estimator_params : dict, default=None
        Optional parameter settings for the base estimators. If given, each key
        should be the index of the estimator in the `base_estimators` list, and
        the value should be a dictionary of parameter names and values.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators.ensemble import SimpleAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> ensemble = SimpleAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Simple Average Ensemble is a basic ensemble technique that can be
      used to combine the predictions of multiple regression models.
    - It is particularly effective when the base estimators have diverse
      strengths and weaknesses, as it can balance out their individual
      performances.
    - This ensemble technique is computationally efficient and easy to
      implement, making it a practical choice for many regression problems.
    - While it provides robustness and improved generalization, it may not
      capture complex relationships between features and target variables as
      advanced ensemble methods like Random Forest or Gradient Boosting.

    See Also
    --------
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    RandomForestRegressor : An ensemble of decision trees with improved performance.
    GradientBoostingRegressor : An ensemble technique for boosting model performance.

    References
    ----------
    .. [1] Breiman, L. (1996). Bagging Predictors. Machine Learning. 24(2):123-140.
    """

    def __init__(
        self, 
        base_estimators, 
        normalize_predictions=False, 
        estimator_params=None, 
        random_state=None, 
        verbose=False
        ):
        self.base_estimators = base_estimators
        self.normalize_predictions = normalize_predictions
        self.random_state = random_state
        self.verbose = verbose
        self.estimator_params = estimator_params if estimator_params is not None else {}

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Simple Average Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        This method fits the base estimators on the training data and the
        corresponding labels. If `sample_weight` is provided, it will be passed
        to the `fit` method of each base estimator. If `normalize_predictions`
        is True, the predictions from each base estimator will be normalized
        before averaging.

        Examples
        --------
        >>> from gofast.estimators.benchmark import SimpleAverageRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> ensemble = SimpleAverageRegressor([LinearRegression(), DecisionTreeRegressor()])
        >>> ensemble.fit(X, y)
        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__)
        
        if sample_weight is not None:
            sample_weight = validate_fit_weights(y, sample_weight)
     
        np.random.seed(self.random_state)
        
        if self.verbose:
            with tqdm(total=len(self.base_estimators), ncols =100,
                      desc='Fitting {self.__class__.__name__}', ascii=True) as pbar:
                for i, estimator in enumerate(self.base_estimators):
                    params = self.estimator_params.get(i, {})
                    estimator = clone(estimator).set_params(**params)
                    estimator = fit_with_estimator(
                        estimator, X, y, sample_weight=sample_weight)
                    self.base_estimators[i] = estimator
                    pbar.update(1)
        else:
            for i, estimator in enumerate(self.base_estimators):
                params = self.estimator_params.get(i, {})
                estimator = clone(estimator).set_params(**params)
                estimator = fit_with_estimator(
                    estimator, X, y, sample_weight=sample_weight)
                self.base_estimators[i] = estimator
        
        self.fitted_ = True
        
        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all base estimators.

        Notes
        -----
        This method uses the fitted base estimators to predict the target values
        for the input samples. The final prediction is the arithmetic mean of
        the predictions from each base estimator. If `normalize_predictions`
        is True, the predictions from each base estimator will be normalized
        before averaging.

        Examples
        --------
        >>> from gofast.estimators.benchmark import SimpleAverageRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> ensemble = SimpleAverageRegressor([LinearRegression(), DecisionTreeRegressor()])
        >>> ensemble.fit(X, y)
        >>> y_pred = ensemble.predict(X)
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True,
                        to_frame=False)
        
        predictions = np.array(
            [estimator.predict(X) for estimator in self.base_estimators])
        
        if self.normalize_predictions:
            scaler = MinMaxScaler()
            predictions = np.array(
                [scaler.fit_transform(pred[:, np.newaxis]).flatten() 
                 for pred in predictions])
        
        y_pred = np.mean(predictions, axis=0)
        
        return y_pred

class SimpleAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple Average Ensemble Classifier.

    This classifier employs a simple average of the predicted class probabilities 
    from multiple base classifiers to determine the final class labels. It is 
    effective in reducing variance and improving generalization over using a 
    single classifier.

    The `SimpleAverageClassifier` aggregates predictions by averaging the 
    probabilities predicted by each classifier in the ensemble. This approach 
    can mitigate individual classifier biases and is particularly beneficial 
    when the classifiers vary widely in their predictions.

    The final predicted class label for an input `x` is determined by computing
    the simple average of the predicted class probabilities from all classifiers:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\frac{1}{n} \sum_{i=1}^{n} \mathbf{P}_i(x)\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input `x`.
    - :math:`n` is the number of base classifiers in the ensemble.
    - :math:`\mathbf{P}_i(x)` represents the predicted class probabilities by the 
      \(i\)-th base classifier for input `x`.

    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`,
        `predict`, and `predict_proba` methods.

    random_state : int, RandomState instance or None, optional, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.

    verbose : bool, default=False
        If True, will print progress messages and use `tqdm` for progress display
        during fitting.

    classifier_params : dict, default=None
        Optional parameter settings for the base classifiers. If given, each key
        should be the index of the classifier in the `base_classifiers` list, and
        the value should be a dictionary of parameter names and values.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers that have been trained on the 
        data.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators.ensemble import SimpleAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ensemble = SimpleAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()]
    ... )
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The `SimpleAverageClassifier` assumes all base classifiers implement the
      `fit`, `predict`, and `predict_proba` methods necessary for ensemble predictions.
    - This approach of averaging class probabilities helps to stabilize the 
      predictions by leveraging the collective intelligence of multiple models, 
      often resulting in more reliable and robust classification outcomes.

    See Also
    --------
    - sklearn.ensemble.VotingClassifier : Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier : Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    """
    def __init__(
        self, 
        base_classifiers, 
        classifier_params=None, 
        random_state=None, 
        verbose=False
        ):
        self.base_classifiers = base_classifiers
        self.random_state = random_state
        self.classifier_params = classifier_params if classifier_params is not None else {}
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Simple Average Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__)
        
        if sample_weight is not None:
            sample_weight = validate_fit_weights(y, sample_weight)
        
        np.random.seed(self.random_state)
        self.classifiers_ = []
        if self.verbose:
            with tqdm(total=len(self.base_classifiers), ncols =100,
                      desc='Fitting {self.__class__.__name__}', ascii=True) as pbar:
                for i, clf in enumerate(self.base_classifiers):
                    params = self.classifier_params.get(i, {})
                    clf = clone(clf).set_params(**params)
                    clf= fit_with_estimator(clf, X, y, sample_weight)
                    self.classifiers_.append(clf)
                    pbar.update(1)
        else:
            for i, clf in enumerate(self.base_classifiers):
                params = self.classifier_params.get(i, {})
                clf = clone(clf).set_params(**params)
                clf= fit_with_estimator(clf, X, y, sample_weight)
                self.classifiers_.append(clf)

        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'classifiers_')
        X = check_array(
            X, accept_large_sparse=True, 
            accept_sparse=True, 
            to_frame=False
            )
        
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)

        # Determine the class with the highest average probability 
        # as the predicted class
        y_pred = np.argmax(avg_proba, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Simple Average Ensemble
        Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_is_fitted(self, 'classifiers_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True,
                        to_frame=False)
        
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)
        return avg_proba
    
    def _average_proba(self, X):
        """
        Calculate the average predicted class probabilities from base classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        avg_proba : array-like of shape (n_samples, n_classes)
            The average predicted class probabilities.
        """
        # Get predicted probabilities from each base classifier
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers_])
        
        # Calculate the average predicted class probabilities
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

class WeightedAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted Average Ensemble Regressor.

    This ensemble model calculates the weighted average of the predictions from
    multiple base regression models. By multiplying each base model's prediction 
    with a corresponding weight, it allows for differential contributions of each 
    model to the final prediction. This technique is particularly effective when 
    base models vary in accuracy or are tuned to different aspects of the data.

    The predictions of the Weighted Average Ensemble are computed as:

    .. math::
        y_{\text{pred}} = \sum_{i=1}^{N} (w_i \times y_{\text{pred}_i})

    where:
    - :math:`w_i` is the weight assigned to the \(i\)-th base estimator.
    - :math:`y_{\text{pred}_i}` is the prediction made by the \(i\)-th base estimator.
    - \(N\) is the number of base estimators in the ensemble.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support `fit` and `predict` methods. These base estimators will be
        combined through weighted averaging to form the ensemble model.

    weights : array-like of shape (n_estimators,) or 'auto', default='auto'
        An array-like object containing the weights for each estimator in the
        `base_estimators` list. The weights determine the importance of each
        base estimator's prediction in the final ensemble prediction. If set
        to 'auto', the weights are determined based on the performance of the
        base models using cross-validation.

    optimizer : str or callable, default=None
        The optimization method to use for hyperparameter tuning. If a string is
        provided, it should be 'RSCV' (Randomized Search CV) or 'GSCV' (Grid Search CV).
        If callable, it should be a function that takes an estimator and a parameter
        grid and returns a fitted estimator.
        
    scaler : str, callable, or None, default=None
        The scaling method to use for feature scaling. If a string is provided,
        it should be one of the following:
        - 'minmax' or '01': Uses `MinMaxScaler` to scale features to [0, 1].
        - 'standard' or 'zscore': Uses `StandardScaler` to scale features to have
          zero mean and unit variance.
        - 'robust': Uses `RobustScaler` to scale features using statistics that
          are robust to outliers.
        - 'sum': Normalizes features by dividing each element by the sum of its
          respective row.
        If callable, it should be a scaler object with a `fit_transform` method.
        If None, no scaling is applied.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy for hyperparameter
        optimization.
        
    random_state : int, RandomState instance or None, optional, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        output across multiple function calls.

    verbose : bool, default=False
        If True, will print progress messages and use `tqdm` for progress display
        during fitting.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators.ensemble import WeightedAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> # Weighted with 0.7 for Linear Regression and 0.3 for Decision Tree
    >>> ensemble = WeightedAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ], [0.7, 0.3])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Weighted Average Ensemble is a versatile ensemble technique that
      allows for fine-grained control over the contribution of each base
      estimator to the final prediction.
    - The weights can be determined based on the performance of the base
      models, domain knowledge, or other criteria.
    - It is particularly effective when the base models have significantly
      different levels of accuracy or are suited to different parts of the
      input space.
    - This ensemble technique provides flexibility and can improve overall
      performance compared to simple averaging or individual estimators.

    See Also
    --------
    SimpleAverageRegressor : An ensemble that equally weights base model 
        predictions.
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    Random Forest : An ensemble of decision trees with improved performance.
    Gradient Boosting : An ensemble technique for boosting model performance.

    """
    def __init__(
        self, base_estimators, 
        weights='auto', 
        random_state=None, 
        verbose=False, 
        optimizer=None, 
        scaler=None, 
        cv=None
        ):
        self.base_estimators = base_estimators
        self.weights = weights
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer
        self.scaler= scaler 
        self.cv = cv

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Weighted Average Ensemble model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. These represent the features of the
            dataset used to train the model. Each row corresponds to a sample,
            and each column corresponds to a feature.
    
        y : array-like of shape (n_samples,)
            The target values (real numbers) for each sample. These are the 
            true values that the model aims to predict. Each element corresponds
            to the target value for a respective sample in `X`.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If provided, these weights will
            be used during the fitting process to give more importance to certain
            samples. This can be useful if some samples are more reliable or 
            representative of the target variable than others.
    
        Returns
        -------
        self : object
            Returns self. This enables method chaining, e.g.,
            `model.fit(X, y).predict(X_test)`.
    
        Notes
        -----
        This method fits the base estimators on the training data `X` and the
        corresponding target values `y`. If `sample_weight` is provided, it will 
        be passed to the `fit` method of each base estimator.
    
        The fitting process involves the following steps:
        1. Validate and preprocess the input data.
        2. Apply scaling to the input data if a scaler is specified.
        3. Fit each base estimator to the scaled data (or original data if no
           scaling is applied). If hyperparameter optimization is specified, it
           will be performed for each base estimator using cross-validation.
    
        Example
        -------
        >>> from gofast.estimators.ensemble import WeightedAverageRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
        >>> base_estimators = [LinearRegression(), DecisionTreeRegressor()]
        >>> weights = [0.7, 0.3]
        >>> ensemble = WeightedAverageRegressor(base_estimators, weights)
        >>> X = np.random.rand(100, 1)
        >>> y = np.random.rand(100)
        >>> ensemble.fit(X, y)
    
        See Also
        --------
        - `predict` : Predict using the fitted Weighted Average Ensemble model.
        - `apply_scaling` : Apply the specified scaler to the data.
        - `optimize_hyperparams` : Optimize hyperparameters for the base estimators.
    
        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__)
        
        sample_weight = validate_fit_weights(y, sample_weight )
       
        np.random.seed(self.random_state)
        
        self.scaler_key_ =None 
        if self.scaler is not None: 
            X, self.scaler, self.scaler_key_= apply_scaling(
                self.scaler,  X, return_keys= True )
            
        # Determine weights if set to 'auto'
        if self.weights == 'auto':
            self.weights = determine_weights(
                self.base_estimators, X, y, cv=self.cv)
        
        if self.verbose:
            with tqdm(total=len(self.base_estimators),
                      desc='Fitting estimators', ascii=True,
                      ncols =100 ) as pbar:
                for i, estimator in enumerate(self.base_estimators):
                    if self.optimizer and self.cv is not None:
                        estimator = optimize_hyperparams(
                            estimator, X, y, optimizer=self.optimizer, 
                            cv= self.cv 
                            )
                    estimator= fit_with_estimator(estimator, X, y, sample_weight)
                    self.base_estimators[i] = estimator
                    pbar.update(1)
        else:
            for i, estimator in enumerate(self.base_estimators):
                if self.optimizer and self.cv:
                    estimator = estimator = optimize_hyperparams(
                        estimator, X, y, optimizer=self.optimizer, 
                        cv= self.cv 
                        )
                estimator= fit_with_estimator(estimator, X, y, sample_weight)
                self.base_estimators[i] = estimator
        
        self.fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row corresponds to a sample, and each column
            corresponds to a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The weighted predicted values, averaged across all base estimators.
    
        Notes
        -----
        This method uses the fitted base estimators to predict the target values
        for the input samples `X`. The final prediction is computed as the weighted
        average of the predictions from each base estimator.
    
        The prediction process involves the following steps:
        1. Validate and preprocess the input data.
        2. Apply scaling to the input data if a scaler is specified.
        3. Collect predictions from each base estimator.
        4. Compute the weighted average of the predictions to produce the final
           predicted values.
    
        Mathematical Formulation:
        The final predicted value for an input sample :math:`x` is computed as:
    
        .. math::
            y_{\text{pred}}(x) = \sum_{i=1}^{N} w_i \times y_{\text{pred}_i}(x)
    
        where:
        - :math:`w_i` is the weight assigned to the \(i\)-th base estimator.
        - :math:`y_{\text{pred}_i}(x)` is the prediction made by the \(i\)-th base
          estimator for input sample :math:`x`.
        - \(N\) is the number of base estimators in the ensemble.
    
        Example
        -------
        >>> from gofast.estimators.ensemble import WeightedAverageRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> import numpy as np
        >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
        >>> base_estimators = [LinearRegression(), DecisionTreeRegressor()]
        >>> weights = [0.7, 0.3]
        >>> ensemble = WeightedAverageRegressor(base_estimators, weights)
        >>> X = np.random.rand(100, 1)
        >>> y = np.random.rand(100)
        >>> ensemble.fit(X, y)
        >>> y_pred = ensemble.predict(X)
    
        See Also
        --------
        - `fit` : Fit the Weighted Average Ensemble model to the data.
        - `apply_scaling` : Apply the specified scaler to the data.
        - `optimize_hyperparams` : Optimize hyperparameters for the base estimators.
    
        References
        ----------
        .. [1] Breiman, L. (1996). Bagging Predictors. Machine Learning. 24(2):123-140.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True,
                        to_frame=False)
        
        if self.scaler_key_ == 'sum':
            X= normalize_sum(X)
        elif self.scaler:
            X = self.scaler.transform(X)
    
        predictions = np.array([estimator.predict(X) for estimator in self.base_estimators])
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        return weighted_predictions
    
class WeightedAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted Average Ensemble Classifier.

    This classifier averages the predictions of multiple base classifiers,
    each weighted by its assigned importance. The ensemble prediction for 
    each class is the weighted average of the predicted probabilities from 
    all classifiers.
    
    See More in :ref:`User Guide`

    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`, 
        `predict`, and `predict_proba` methods.
        
    weights : array-like of shape (n_classifiers,), default='auto'
        Weights for each classifier in the `base_classifiers` list. If set to 
        'auto', weights are determined based on the performance of the base 
        classifiers using cross-validation.

    optimizer : str, default=None
        The optimization method to use for hyperparameter tuning. Supported 
        options include:
        - 'RSCV' : RandomizedSearchCV for random search over hyperparameter 
          distributions.
        - 'GSCV' : GridSearchCV for exhaustive search over specified 
          hyperparameter values.

    scaler : str, default=None
        The type of scaler to use for feature normalization. Supported options 
        include:
        - 'minmax' or '01': MinMaxScaler
        - 'standard' or 'zscore': StandardScaler
        - 'robust': RobustScaler
        - 'sum': Sum normalization

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. If None, 5-fold 
        cross-validation is used.

    random_state : int, default=None
        Controls the randomness for reproducibility.

    verbose : bool, default=False
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    scaler_ : object
        The scaler fitted on the input data.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators.ensemble import WeightedAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble = WeightedAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
    ...     weights='auto'
    ... )
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    The effectiveness of this ensemble classifier depends on the diversity 
    and performance of the base classifiers, as well as the appropriateness 
    of their assigned weights. Fine-tuning these elements is crucial for 
    optimal performance.

    The final predicted class label :math:`C_{\text{final}}(x)` for input 
    :math:`x` is determined by a weighted average of the predicted 
    probabilities from all classifiers, formulated as follows:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} 
        \text{weights}_i \cdot \mathbf{P}_i(x)\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for 
      input :math:`x`.
    - :math:`n` is the number of base classifiers in the ensemble.
    - :math:`\text{weights}_i` is the weight assigned to the :math:`i`-th 
      base classifier.
    - :math:`\mathbf{P}_i(x)` represents the predicted class probabilities 
      by the :math:`i`-th base classifier for input :math:`x`.

    See Also
    --------
    - sklearn.ensemble.VotingClassifier : Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier : Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    References
    ----------
    .. [1] Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). 
           Algorithms for Hyper-Parameter Optimization. In Advances in Neural 
           Information Processing Systems (pp. 2546-2554).
    .. [2] Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
    """

    def __init__(
        self, 
        base_classifiers, 
        weights='auto',
        optimizer=None, 
        scaler=None, 
        cv=None,
        random_state=None, 
        verbose=False, 
        ):
        self.base_classifiers = base_classifiers
        self.weights = weights
        self.optimizer = optimizer
        self.scaler = scaler
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
            
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Weighted Average Ensemble Classifier model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each row represents a sample, and each 
            column represents a feature.
    
        y : array-like of shape (n_samples,)
            The target class labels. Each element represents the target label for 
            a respective sample in `X`.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If provided, these weights are 
            used during the fitting process.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        - This method fits the model to the data `X` and `y` by training each 
          base classifier in the ensemble. If `sample_weight` is provided, it 
          will be used to weight the samples during fitting.
        - If `weights` is set to 'auto', the weights for each base classifier 
          are determined based on their performance using cross-validation.
        - If `scaler` is specified, the data `X` will be scaled accordingly 
          before fitting.
        - If `optimizer` and `cv` are specified, hyperparameter optimization 
          will be performed for each base classifier.
    
        The fitting process can be summarized as follows:
        1. Apply scaling to `X` if a scaler is specified.
        2. Determine weights for each base classifier if `weights` is set to 'auto'.
        3. Train each base classifier using the (optionally scaled) data `X` 
           and target `y`, optionally using sample weights.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import WeightedAverageClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
        >>> ensemble = WeightedAverageClassifier(
        ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
        ...     weights='auto'
        ... )
        >>> ensemble.fit(X, y)
    
        See Also
        --------
        WeightedAverageClassifier : Class for weighted averaging of classifier predictions.
        determine_weights : Function to determine weights based on classifier performance.
        optimize_hyperparams : Function to optimize hyperparameters of classifiers.
        apply_scaling : Function to scale input data.
    
        References
        ----------
        .. [1] Bergstra, J., Bardenet, R., Bengio

        """
        X, y = check_X_y(X, y, estimator=self.__class__.__name__)
        if sample_weight is not None:
            sample_weight = validate_fit_weights(y, sample_weight)

        np.random.seed(self.random_state)
        self.scaler_key_ = None 
        
        # Apply scaling if specified
        if self.scaler is not None: 
            X, self.scaler, self.scaler_key_ = apply_scaling(
                self.scaler, X, return_keys=True)
        
        # Determine weights if set to 'auto'
        if self.weights == 'auto':
            self.weights = determine_weights(
                self.base_classifiers, X, y, cv=self.cv, problem='classification'
            )
        
        if self.verbose:
            with tqdm(total=len(self.base_classifiers), desc='Fitting estimators',
                      ascii=True, ncols=100) as pbar:
                for i, classifier in enumerate(self.base_classifiers):
                    if self.optimizer and self.cv:
                        classifier = optimize_hyperparams(
                            classifier, X, y, optimizer=self.optimizer, cv=self.cv
                        )
                    classifier = fit_with_estimator(classifier, X, y, sample_weight)
                    self.base_classifiers[i] = classifier
                    pbar.update(1)
        else:
            for i, classifier in enumerate(self.base_classifiers):
                if self.optimizer and self.cv:
                    classifier = optimize_hyperparams(
                        classifier, X, y, optimizer=self.optimizer, cv=self.cv
                    )
                classifier = fit_with_estimator(classifier, X, y, sample_weight)
                self.base_classifiers[i] = classifier
        
        self.fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column 
            represents a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
    
        Notes
        -----
        - This method predicts class labels for the input samples by averaging 
          the predicted probabilities from each base classifier, weighted by 
          their respective importance.
        - If a scaler was used during fitting, the input samples `X` will be 
          scaled before prediction.
        - The final predicted class label :math:`\hat{y}_i` for each sample 
          :math:`i` is determined by the class with the highest weighted average 
          probability, formulated as:
          
          .. math::
              \hat{y}_i = \arg\max_c \sum_{j=1}^{n} w_j P_j(c \mid x_i)
          
          where:
          - :math:`\hat{y}_i` is the predicted class label for sample :math:`i`.
          - :math:`w_j` is the weight for the :math:`j`-th base classifier.
          - :math:`P_j(c \mid x_i)` is the predicted probability of class :math:`c` 
            by the :math:`j`-th base classifier for sample :math:`i`.
          - :math:`n` is the number of base classifiers in the ensemble.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import WeightedAverageClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
        >>> ensemble = WeightedAverageClassifier(
        ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
        ...     weights='auto'
        ... )
        >>> ensemble.fit(X, y)
        >>> y_pred = ensemble.predict(X)
    
        See Also
        --------
        WeightedAverageClassifier : Class for weighted averaging of classifier predictions.
        determine_weights : Function to determine weights based on classifier performance.
        apply_scaling : Function to scale input data.
    
        References
        ----------
        .. [1] Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). 
               Algorithms for Hyper-Parameter Optimization. In Advances in Neural 
               Information Processing Systems (pp. 2546-2554).
        .. [2] Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
        """

        check_is_fitted(self, 'fitted_')
        
        X = check_array(
            X, accept_large_sparse=True, accept_sparse=True, to_frame=False)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        weighted_sum = np.zeros((X.shape[0], len(
            np.unique(self.base_classifiers[0].classes_))))

        for clf, weight in zip(self.base_classifiers, self.weights):
            probabilities = clf.predict_proba(X)
            weighted_sum += weight * probabilities

        return np.argmax(weighted_sum, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Weighted Average Ensemble 
        Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column 
            represents a feature.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities. Each row represents a sample, 
            and each column represents the probability of the sample belonging 
            to a specific class.
    
        Notes
        -----
        - This method predicts class probabilities for the input samples by 
          averaging the predicted probabilities from each base classifier, 
          weighted by their respective importance.
        - If a scaler was used during fitting, the input samples `X` will be 
          scaled before prediction.
        - The final predicted probability :math:`P(c \mid x_i)` for class :math:`c` 
          and sample :math:`i` is determined by the weighted average of the 
          predicted probabilities from all base classifiers, formulated as:
          
          .. math::
              P(c \mid x_i) = \sum_{j=1}^{n} w_j P_j(c \mid x_i)
          
          where:
          - :math:`P(c \mid x_i)` is the predicted probability of class :math:`c` 
            for sample :math:`i`.
          - :math:`w_j` is the weight for the :math:`j`-th base classifier.
          - :math:`P_j(c \mid x_i)` is the predicted probability of class :math:`c` 
            by the :math:`j`-th base classifier for sample :math:`i`.
          - :math:`n` is the number of base classifiers in the ensemble.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import WeightedAverageClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
        >>> ensemble = WeightedAverageClassifier(
        ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
        ...     weights='auto'
        ... )
        >>> ensemble.fit(X, y)
        >>> y_proba = ensemble.predict_proba(X)
    
        See Also
        --------
        WeightedAverageClassifier : Class for weighted averaging of classifier predictions.
        determine_weights : Function to determine weights based on classifier performance.
        apply_scaling : Function to scale input data.
    
        References
        ----------
        .. [1] Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). 
               Algorithms for Hyper-Parameter Optimization. In Advances in Neural 
               Information Processing Systems (pp. 2546-2554).
        .. [2] Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
        """
        # Check if the classifier is fitted
        check_is_fitted(self, 'fitted_')
        
        # Validate input data
        X = check_array(
            X, accept_large_sparse=True, accept_sparse=True, to_frame=False)
        
        # Apply scaling if specified
        if self.scaler_key_ =='sum': 
            X= normalize_sum(X )
        elif self.scaler:
            X = self.scaler.transform(X)
        
        # Collect predicted probabilities from each base classifier
        probas = [clf.predict_proba(X) for clf in self.base_classifiers]
        
        # Calculate the weighted average of predicted probabilities
        weighted_avg_proba = np.average(probas, axis=0, weights=self.weights)
        
        return weighted_avg_proba
    
class EnsembleClassifier(ClassifierMixin, BaseEnsemble):
    """
    Ensemble Classifier.

    The `EnsembleClassifier` employs an ensemble approach, combining 
    multiple classification models to form a more robust and accurate model. 
    It supports three strategies: bagging, boosting, and a hybrid approach 
    that combines both bagging and boosting.

    Parameters
    ----------
    
    n_estimators : int, default=50
        The number of base estimators in the ensemble. For the hybrid strategy, 
        this is the total number of base estimators combined across bagging 
        and boosting.
        
    eta0 : float, default=0.1
        Learning rate shrinks the contribution of each classifier by `eta0`.
        This parameter is only used for the boosting and hybrid strategies.
        
    max_depth : int, default=None
        The maximum depth of the individual classification estimators. This
        parameter controls the complexity of each base estimator in the ensemble.
        Setting `max_depth` limits the number of nodes in each tree, thereby
        preventing overfitting. A higher `max_depth` allows the model to learn
        more intricate patterns in the data, but it can also lead to overfitting
        if set too high.

        - If `None`, then nodes are expanded until all leaves contain less than
          `min_samples_split` samples.
        - If an integer value is provided, the tree will grow until the specified
          maximum depth is reached.

        The choice of `max_depth` balances the bias-variance tradeoff:
        - A shallow tree (low `max_depth`) might underfit the data, capturing only
          the most obvious patterns and leaving out finer details.
        - A deep tree (high `max_depth`) might overfit the data, capturing noise
          and outliers along with the underlying patterns.

        It is crucial to tune this parameter based on cross-validation or other
        model evaluation techniques to ensure optimal performance.
        
    strategy : {'hybrid', 'bagging', 'boosting'}, default='hybrid'
        The strategy to use for the ensemble. Options are:
        - 'bagging': Use Bagging strategy.
        - 'boosting': Use Boosting strategy.
        - 'hybrid': Combine Bagging and Boosting strategies.
        
    max_samples : float or int, default=1.0
        The number of samples to draw from X to train each base estimator. If 
        float, then draw `max_samples * n_samples` samples.
        
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator. If 
        float, then draw `max_features * n_features` features.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling without 
        replacement is performed.
        
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
        
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization error.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit and 
        add more estimators to the ensemble.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`. 
        None means 1 unless in a `joblib.parallel_backend` context.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value. Used to control tree growth.
        
    init : estimator object, default=None
        An estimator object that is used to compute the initial predictions. 
        Used only for boosting.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        - If int, consider `min_samples_split` as the minimum number.
        - If float, `min_samples_split` is a fraction and 
        `ceil(min_samples_split * n_samples)` 
          is the minimum number of samples for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split 
        point at any depth will only be considered if it leaves at least 
        `min_samples_leaf` training samples in each of the left and right branches.
        - If int, consider `min_samples_leaf` as the minimum number.
        - If float, `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` is the minimum number of samples 
          for each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to 
        be at a leaf node.
        
    max_leaf_nodes : int, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes 
        are defined as relative reduction in impurity. If None, unlimited 
        number of leaf nodes.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Used only for boosting.
        
    n_iter_no_change : int, default=None
        Used to decide if early stopping will be used to terminate training 
        when validation score is not improving. Used only for boosting.
        
    tol : float, default=1e-4
        Tolerance for the early stopping. Used only for boosting.
        
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    
    base_estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset. If None, 
        then the base estimator is a `DecisionTreeClassifier`.
        
    random_state : int or RandomState, default=None
        Controls the randomness of the estimator for reproducibility. Pass 
        an int for reproducible output across multiple function calls.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values 
        indicate more messages.
        
    Attributes
    ----------
    model_ : object
        The fitted ensemble model.
        
    Examples
    --------
    Here's an example of how to use the `EnsembleClassifier` on a dataset:

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score
    >>> from gofast.estimators.ensemble import EnsembleClassifier

    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    
    >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
    ...                          random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print("Classification accuracy:", accuracy_score(y_test, y_pred))

    Notes
    -----
    This model combines the predictive power of multiple trees through 
    bagging, boosting, or a hybrid approach, effectively reducing variance 
    and improving accuracy over a single decision tree. The ensemble 
    prediction is computed by aggregating the predictions from each individual 
    classification model within the ensemble.

    The hybrid strategy uses a combination of bagging and boosting, where 
    the bagging model contains boosting models as its base estimators. This 
    leverages the strengths of both approaches to achieve better performance.

    .. math::
        P_{\text{class}} = \frac{1}{N} \sum_{i=1}^{N} P_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`P_{\text{class}}` is the final predicted probability aggregated from 
      all trees.
    - :math:`P_{\text{tree}_i}` represents the predicted probability made by the 
      :math:`i`-th classification tree.

    See Also
    --------
    sklearn.ensemble.BaggingClassifier : A bagging classifier.
    sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
    sklearn.metrics.accuracy_score : A common metric for evaluating 
        classification models.
    """
    is_classifier = True
    default_estimator = DecisionTreeClassifier 

    def __init__(
        self, 
        n_estimators=50,
        eta0=0.1,
        max_depth=None,
        strategy='hybrid',
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
        random_state=None,
        verbose=False
        ):
        super().__init__(
            n_estimators=n_estimators, 
            eta0=eta0,
            max_depth=max_depth,
            strategy=strategy,
            random_state=random_state,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha,
            estimator =estimator, 
            verbose=verbose
            )
        
    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted ensemble model.
    
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
        This method uses the fitted ensemble model to predict the probabilities
        of the classes for the input samples. The input samples `X` are checked
        for validity, ensuring they conform to the expected format and type.
    
        The probability predictions are computed by aggregating the probability
        outputs of the individual base estimators in the ensemble. This method
        is only applicable for classification tasks and will raise an error if
        used with a regressor.
    
        The mathematical formulation of the probability prediction process
        can be described as follows:
    
        .. math::
            P_{\text{class}} = \frac{1}{N} \sum_{i=1}^{N} P_{\text{est}_i}
    
        where:
        - :math:`N` is the number of base estimators in the ensemble.
        - :math:`P_{\text{class}}` is the final predicted probability aggregated
          from all base estimators.
        - :math:`P_{\text{est}_i}` represents the predicted probability made by
          the :math:`i`-th base estimator.
    
        The `predict_proba` method is particularly useful for tasks requiring
        probabilistic predictions rather than discrete class labels. It can be
        used for applications such as uncertainty estimation, thresholding,
        and calibration.
    
        Raises
        ------
        NotFittedError
            If the estimator is not fitted, i.e., `fit` has not been called
            before `predict_proba`.
    
        AttributeError
            If the base estimator does not have a `predict_proba` method.
    
        Examples
        --------
        Here's an example of how to use the `predict_proba` method with
        `EnsembleClassifier`:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
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
    
        See Also
        --------
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.utils.validation.check_is_fitted : Utility function to check if the
            estimator is fitted.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
    
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
            X, accept_large_sparse= True, 
            accept_sparse= True, 
            input_name="X", 
        )
        return self.model_.predict_proba(X)


class EnsembleRegressor(RegressorMixin, BaseEnsemble):
    """
    Ensemble Regressor.

    The `EnsembleRegressor` employs an ensemble approach, combining 
    multiple regression models to form a more robust and accurate model. 
    It supports three strategies: bagging, boosting, and a hybrid approach 
    that combines both bagging and boosting.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of base estimators in the ensemble. For the hybrid strategy, 
        this is the total number of base estimators combined across bagging 
        and boosting.
        
    eta0 : float, default=0.1
        Learning rate shrinks the contribution of each tree by `eta0`.
        This parameter is only used for the boosting and hybrid strategies.
        
    max_depth : int, default=None
        The maximum depth of the individual regression estimators. This
        parameter controls the complexity of each base estimator in the ensemble.
        Setting `max_depth` limits the number of nodes in each tree, thereby
        preventing overfitting. A higher `max_depth` allows the model to learn
        more intricate patterns in the data, but it can also lead to overfitting
        if set too high.

        - If `None`, then nodes are expanded until all leaves contain less than
          `min_samples_split` samples.
        - If an integer value is provided, the tree will grow until the specified
          maximum depth is reached.

        The choice of `max_depth` balances the bias-variance tradeoff:
        - A shallow tree (low `max_depth`) might underfit the data, capturing only
          the most obvious patterns and leaving out finer details.
        - A deep tree (high `max_depth`) might overfit the data, capturing noise
          and outliers along with the underlying patterns.

        It is crucial to tune this parameter based on cross-validation or other
        model evaluation techniques to ensure optimal performance.
            
    strategy : {'hybrid', 'bagging', 'boosting'}, default='hybrid'
        The strategy to use for the ensemble. Options are:
        - 'bagging': Use Bagging strategy.
        - 'boosting': Use Boosting strategy.
        - 'hybrid': Combine Bagging and Boosting strategies.
 
    max_samples : float or int, default=1.0
        The number of samples to draw from X to train each base estimator. If 
        float, then draw `max_samples * n_samples` samples.
        
    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator. If 
        float, then draw `max_features * n_features` features.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling without 
        replacement is performed.
        
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
        
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization error.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit and 
        add more estimators to the ensemble.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`. 
        None means 1 unless in a `joblib.parallel_backend` context.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value. Used to control tree growth.
        
    init : estimator object, default=None
        An estimator object that is used to compute the initial predictions. 
        Used only for boosting.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        - If int, consider `min_samples_split` as the minimum number.
        - If float, `min_samples_split` is a fraction and 
        `ceil(min_samples_split * n_samples)` 
          is the minimum number of samples for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split 
        point at any depth will only be considered if it leaves at least 
        `min_samples_leaf` training samples in each of the left and right branches.
        - If int, consider `min_samples_leaf` as the minimum number.
        - If float, `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` is the minimum number of samples 
          for each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to 
        be at a leaf node.
        
    max_leaf_nodes : int, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion. Best nodes 
        are defined as relative reduction in impurity. If None, unlimited 
        number of leaf nodes.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Used only for boosting.
        
    n_iter_no_change : int, default=None
        Used to decide if early stopping will be used to terminate training 
        when validation score is not improving. Used only for boosting.
        
    tol : float, default=1e-4
        Tolerance for the early stopping. Used only for boosting.
        
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
        
    base_estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset. If None, 
        then the base estimator is a `DecisionTreeRegressor`.
        
    random_state : int or RandomState, default=None
        Controls the randomness of the estimator for reproducibility. Pass 
        an int for reproducible output across multiple function calls.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values 
        indicate more messages.
        
    Attributes
    ----------
    model_ : object
        The fitted ensemble model.
        
    Examples
    --------
    Here's an example of how to use the `EnsembleHybridRegressor` on a dataset:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error
    >>> from gofast.estimators.ensemble import EnsembleRegressor

    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    
    >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
    ...                               random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print("Regression MSE:", mean_squared_error(y_test, y_pred))

    Notes
    -----
    This model combines the predictive power of multiple trees through 
    bagging, boosting, or a hybrid approach, effectively reducing variance 
    and improving accuracy over a single decision tree. The ensemble 
    prediction is computed by averaging the predictions from each individual 
    regression model within the ensemble.

    The hybrid strategy uses a combination of bagging and boosting, where 
    the bagging model contains boosting models as its base estimators. This 
    leverages the strengths of both approaches to achieve better performance.

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from 
      all trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the 
      :math:`i`-th regression tree.

    See Also
    --------
    sklearn.ensemble.BaggingRegressor : A bagging regressor.
    sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    sklearn.metrics.mean_squared_error : A common metric for evaluating 
        regression models.
    """
    is_classifier = False
    default_estimator = DecisionTreeRegressor 
    
    def __init__(
        self,
        n_estimators=50,
        eta0=0.1,
        max_depth=None,
        strategy='hybrid', 
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
        random_state=None,
        verbose=0
       ):
        super().__init__(
            n_estimators=n_estimators, 
            eta0=eta0,
            max_depth=max_depth,
            strategy=strategy,
            random_state=random_state,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            estimator =estimator, 
            ccp_alpha=ccp_alpha,
            verbose=verbose
       )

 



