# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
`boosting` module provides advanced boosting models for machine learning
for hybrid boosting approaches in classification and regression tasks.
"""

from __future__ import annotations 
import numpy as np
from tqdm import tqdm 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics  import  mean_squared_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted

__all__=[
    "BoostingTreeRegressor","BoostingTreeClassifier",
    "HybridBoostingClassifier","HybridBoostingRegressor",
    ]

class BoostingTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Enhanced Boosted Regression Tree (BRT) for Regression Tasks.

    The Enhanced Boosted Regression Tree (BRT) is an advanced implementation
    of the Boosted Regression Tree algorithm, aimed at improving performance 
    and reducing overfitting. This model incorporates features like support for 
    multiple loss functions, stochastic boosting, and controlled tree depth for 
    pruning.

    BRT builds on ensemble learning principles, combining multiple decision trees 
    (base learners) to enhance prediction accuracy. It focuses on improving areas 
    where previous iterations of the model underperformed.

    Features:
    - ``Different Loss Functions``: Supports 'linear', 'square', and 'exponential' 
      loss functions, utilizing their derivatives to update residuals.
      
    - ``Stochastic Boosting``: The model includes an option for stochastic 
      boosting, controlled by the subsample parameter, which dictates the 
      fraction of samples used for fitting each base learner. This can 
      help in reducing overfitting.
      
    - ``Tree Pruning``: While explicit tree pruning isn't detailed here, it can 
      be managed via the max_depth parameter. Additional pruning techniques can 
      be implemented within the DecisionTreeRegressor fitting process.
      
    The iterative updating process of the ensemble is mathematically
    represented as:

    .. math::
        F_{k}(x) = \text{Prediction of the ensemble at iteration } k.

        r = y - F_{k}(x) \text{ (Residual calculation)}

        F_{k+1}(x) = F_{k}(x) + \text{eta0} \cdot h_{k+1}(x)

    where:
    - :math:`F_{k}(x)` is the prediction of the ensemble at iteration \(k\).
    - \(r\) represents the residuals, calculated as the difference between the 
      actual values \(y\) and the ensemble's predictions.
    - :math:`F_{k+1}(x)` is the updated prediction after adding the new tree.
    - :math:`h_{k+1}(x)` is the contribution of the newly added tree at 
      iteration \(k+1\).
    - `eta0` is a hyperparameter that controls the influence of each 
      new tree on the final outcome.

    The Boosted Regression Tree method effectively harnesses the strengths 
    of multiple trees to achieve lower bias and variance, making it highly 
    effective for complex regression tasks with varied data dynamics.


    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble.
        
    eta0 : float, default=0.1
        The rate at which the boosting algorithm adapts from previous 
        trees' errors.
        
    max_depth : int, default=3
        The maximum depth of each regression tree.
        
    loss : str, default='linear'
        The loss function to use. Supported values are 'linear', 
        'square', and 'exponential'.
        
    subsample : float, default=1.0
        The fraction of samples to be used for fitting each individual base 
        learner. If smaller than 1.0, it enables stochastic boosting.
        
    criterion : str, default="squared_error"
        The function to measure the quality of a split. Supported criteria 
        are "squared_error" for the mean squared error, which is equal to 
        variance reduction as feature selection criterion, and "friedman_mse", 
        which uses mean squared error with Friedman's improvement score for 
        potential splits.
        
    splitter : str, default="best"
        The strategy used to choose the split at each node. Supported strategies 
        are "best" to choose the best split and "random" to choose the best 
        random split.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and ceil
        (min_samples_split * n_samples) are the minimum number of samples for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split 
        point at any depth will only be considered if it leaves at least 
        min_samples_leaf training samples in each of the left and right branches.
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and ceil
        (min_samples_leaf * n_samples) are the minimum number of samples for each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input 
        samples) required to be at a leaf node. Samples have equal weight when sample_weight 
        is not provided.
        
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features * n_features) 
          features are considered at each split.
        - If "auto", then max_features=n_features.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always 
        randomly permuted at each split. When max_features < n_features, 
        the algorithm will select max_features at random at each split 
        before finding the best split among them. 
        - Pass an int for reproducible output across multiple function calls.
        
    max_leaf_nodes : int or None, default=None
        Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined 
        as relative reduction in impurity. If None, then unlimited number of leaf nodes.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater 
        than or equal to this value.
        
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree 
        with the largest cost complexity that is smaller than ccp_alpha will be chosen.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    
    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    initial_prediction_ : float
        The initial prediction (mean of y).

    Examples
    --------
    >>> from gofast.estimators.boosting import BoostingTreeRegressor
    >>> brt = BoostingTreeRegressor(n_estimators=100, eta0=0.1, 
                                    max_depth=3, loss='linear', subsample=0.8)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> brt.fit(X, y)
    >>> y_pred = brt.predict(X)
    
    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: The scikit-learn library's
      implementation of gradient boosting for regression tasks.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.
  
    Notes
    -----
    - The Boosted Regression Tree model is built iteratively, focusing on
      minimizing errors and improving predictions.
    - Different loss functions can be selected, allowing flexibility in
      modeling.
    - Stochastic boosting can help in reducing overfitting by using a fraction
      of samples for fitting each tree.
    - Tree depth can be controlled to avoid overly complex models.


    Notes
    -----
    - The Boosted Regression Tree model is built iteratively, focusing on
      minimizing errors and improving predictions.
    - Different loss functions can be selected, allowing flexibility in
      modeling.
    - Stochastic boosting can help in reducing overfitting by using a fraction
      of samples for fitting each tree.
    - Tree depth can be controlled to avoid overly complex models.
    
    """
    def __init__(
        self, 
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3,
        loss='linear', 
        subsample=1.0, 
        criterion="squared_error", 
        splitter = "best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        ccp_alpha=0., 
        verbose=False 
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.loss = loss
        self.subsample = subsample
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
        self.verbose=verbose 

    def _loss_derivative(self, y, y_pred):
        """
        Compute the derivative of the loss function.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            True values.
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        loss_derivative : array-like of shape (n_samples,)
        """
        if self.loss == 'linear':
            return y - y_pred
        elif self.loss == 'square':
            return 2 * (y - y_pred)
        elif self.loss == 'exponential':
            return np.exp(y_pred - y)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        """
        Fit the Boosted Decision Tree Regressor model to the data.
    
        This method trains the Boosted Decision Tree Regressor on the provided 
        training data, `X` and `y`. It sequentially adds decision trees to the 
        ensemble, each time fitting a new tree to the residuals of the previous 
        trees' predictions. The residuals are computed using the specified loss 
        function, and the predictions are updated accordingly.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each row represents a sample, and each 
            column represents a feature.
    
        y : array-like of shape (n_samples,)
            The target values. Each element in the array represents the target 
            value for the corresponding sample in `X`.
    
        Returns
        -------
        self : object
            Returns self.
    
        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from gofast.estimators.tree import BoostingTreeRegressor
        >>> X, y = make_regression(n_samples=100, n_features=4)
        >>> reg = BoostingTreeRegressor(n_estimators=100, max_depth=3, eta0=0.1)
        >>> reg.fit(X, y)
        >>> print(reg.estimators_)
    
        Notes
        -----
        The fitting process involves iteratively training decision trees on the 
        residuals of the previous trees' predictions. The residuals are updated 
        using the specified loss function, and the predictions are adjusted using 
        the learning rate (`eta0`).
        """

        X, y = check_X_y(X, y, estimator= get_estimator_name(self), 
                         accept_large_sparse= True, accept_sparse =True 
                         )
        n_samples = X.shape[0]
        self.estimators_ = []
        self.initial_prediction_ = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction_)
        
        if self.verbose:
            progress_bar = tqdm(range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion=self.criterion,
                splitter=self.splitter,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha
            )
            
            sample_size = int(self.subsample * n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_subset, y_subset, y_pred_subset = X[indices], y[indices], y_pred[indices]
            
            residual = self._loss_derivative(y_subset, y_pred_subset)
            
            tree.fit(X_subset, residual)
            prediction = tree.predict(X)
            
            y_pred += self.eta0 * prediction
            self.estimators_.append(tree)
            
            if self.verbose: 
                progress_bar.update (1)
                
        if self.verbose: 
            progress_bar.close() 
            
        return self

    def predict(self, X):
        """
        Predict target values for samples in `X`.
    
        This method predicts the target values for the input samples in `X` using 
        the trained Boosted Decision Tree Regressor. It aggregates the predictions 
        from all the individual trees in the ensemble to produce the final 
        predicted values.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column 
            represents a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values for the input samples. Each element in the 
            array represents the predicted target value for the corresponding 
            sample in `X`.
    
        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from gofast.estimators.tree import BoostingTreeRegressor
        >>> X, y = make_regression(n_samples=100, n_features=4)
        >>> reg = BoostingTreeRegressor(n_estimators=100, max_depth=3, eta0=0.1)
        >>> reg.fit(X, y)
        >>> y_pred = reg.predict(X)
        >>> print(y_pred)
    
        Notes
        -----
        The prediction process involves aggregating the predictions from all the 
        trees in the ensemble. Each tree's prediction is scaled by the learning 
        rate (`eta0`) before being added to the cumulative prediction.
        """

        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        y_pred = np.full(X.shape[0], self.initial_prediction_)
        
        for tree in self.estimators_:
            y_pred += self.eta0 * tree.predict(X)

        return y_pred

class BoostingTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Boosted Decision Tree Classifier.

    This classifier employs an ensemble boosting method using decision 
    trees. It builds the model by sequentially adding trees, where each 
    tree is trained to correct the errors made by the previous ones. 
    The final model's prediction is the weighted aggregate of all 
    individual trees' predictions, where weights are adjusted by the 
    learning rate.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble. More trees can lead to 
        better performance but increase computational complexity.

    max_depth : int, default=3
        The maximum depth of each decision tree. Controls the complexity 
        of the model. Deeper trees can capture more complex patterns but 
        may overfit.

    eta0 : float, default=0.1
        The rate at which the boosting process adapts to the errors of the 
        previous trees. A lower rate requires more trees to achieve similar 
        performance levels but can yield a more robust model.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria 
        are "gini" for the Gini impurity and "entropy" for the information 
        gain.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported 
        strategies are "best" to choose the best split and "random" to 
        choose the best random split.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and 
          `ceil(min_samples_split * n_samples)` are the minimum number of 
          samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A 
        split point at any depth will only be considered if it leaves at 
        least `min_samples_leaf` training samples in each of the left and 
        right branches.
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

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form `{class_label: weight}`. 
        If None, all classes are supposed to have weight one. If "balanced", 
        the class weights will be adjusted inversely proportional to class 
        frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than 
        `ccp_alpha` will be chosen.
        
    verbose : int, default=0
        Controls the verbosity when fitting.
    
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators. Available after fitting.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    initial_prediction_ : float
        The initial prediction (log-odds) used for the logistic regression.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.estimators.tree import BoostingTreeClassifier
    >>> X, y = make_classification(n_samples=100, n_features=4)
    >>> clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
    >>> clf.fit(X, y)
    >>> print(clf.predict(X))

    Notes
    -----
    The BoostedClassifierTree combines weak learners (decision trees) into 
    a stronger model. The boosting process iteratively adjusts the weights 
    of observations based on the previous trees' errors. The final 
    prediction is made based on a weighted majority vote (or sum) of the 
    weak learners' predictions.

    The boosting procedure is mathematically formulated as:

    .. math::
        F_t(x) = F_{t-1}(x) + \\eta_t h_t(x)

    where \( F_t(x) \) is the model at iteration \( t \), \( \\eta_t \) is 
    the learning rate, and \( h_t(x) \) is the weak learner.

    References
    ----------
    1. Y. Freund, R. E. Schapire, "A Decision-Theoretic Generalization of 
       On-Line Learning and an Application to Boosting", 1995.
    2. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting 
       Machine", Annals of Statistics, 2001.
    3. T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical 
       Learning", Springer, 2009.

    See Also
    --------
    sklearn.tree.DecisionTreeClassifier : A decision tree classifier.
    sklearn.ensemble.AdaBoostClassifier : An adaptive boosting classifier.
    sklearn.ensemble.GradientBoostingClassifier : A gradient boosting 
        machine for classification.
    """
    def __init__(
        self,
        n_estimators=100, 
        max_depth=3, 
        eta0=0.1,
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        class_weight=None, 
        ccp_alpha=0., 
        verbose=0 
        ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta0 = eta0
        self.criterion = criterion 
        self.splitter = splitter 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf 
        self.max_features = max_features 
        self.random_state = random_state 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight 
        self.ccp_alpha = ccp_alpha 
        self.verbose=verbose 
        

    def fit(self, X, y):
        """
        Fit the Boosted Decision Tree Classifier model to the data.
    
        This method trains the Boosted Decision Tree Classifier on the provided 
        training data, `X` and `y`. It sequentially adds decision trees to the 
        ensemble, each time fitting a new tree to the residuals of the previous 
        trees' predictions. The residuals are computed using the logistic loss 
        function, and the predictions are updated accordingly.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Each row represents a sample, and each 
            column represents a feature.
    
        y : array-like of shape (n_samples,)
            The target class labels. Each element in the array represents the 
            class label for the corresponding sample in `X`.
    
        Returns
        -------
        self : object
            Returns self.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from gofast.estimators.tree import BoostingTreeClassifier
        >>> X, y = make_classification(n_samples=100, n_features=4)
        >>> clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
        >>> clf.fit(X, y)
        >>> print(clf.estimators_)
    
        Notes
        -----
        The fitting process involves iteratively training decision trees on the 
        residuals of the previous trees' predictions. The residuals are updated 
        using the logistic function, and the predictions are adjusted using the 
        learning rate (`eta0`).
        """
        X, y = check_X_y(X, y, estimator= get_estimator_name( self))
        self.classes_ = np.unique(y)
        self.estimators_ = []
        self.initial_prediction_ = np.log(np.mean(y) / (1 - np.mean(y)))
        y_pred = np.full(y.shape, self.initial_prediction_, dtype=float)
        
        if self.verbose:
            progress_bar = tqdm(range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                criterion=self.criterion, 
                splitter=self.splitter, 
                min_samples_split=self.min_samples_split, 
                min_samples_leaf=self.min_samples_leaf, 
                min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                max_features=self.max_features, 
                random_state=self.random_state, 
                max_leaf_nodes=self.max_leaf_nodes, 
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight, 
                ccp_alpha=self.ccp_alpha
            )
            # Compute residual
            # Update residuals using logistic function
            residual = y - (1 / (1 + np.exp(-y_pred)))  
            
            # Fit tree on residual
            tree.fit(X, np.sign(residual))
            prediction = tree.predict(X)
            
            # Update predictions
            y_pred += self.eta0 * (2 * prediction - 1)
            self.estimators_.append(tree)
 
            if self.verbose: 
                progress_bar.update (1)
                    
        if self.verbose: 
            progress_bar.close() 
            
        return self

    def predict(self, X):
        """
        Predict class labels for samples in `X`.
    
        This method predicts the class labels for the input samples in `X` using 
        the trained Boosted Decision Tree Classifier. It aggregates the predictions 
        from all the individual trees in the ensemble, applies a logistic 
        transformation, and thresholds the result to obtain binary class labels.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column 
            represents a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels for the input samples. Each element in the 
            array represents the predicted class label for the corresponding sample 
            in `X`.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from gofast.estimators.tree import BoostingTreeClassifier
        >>> X, y = make_classification(n_samples=100, n_features=4)
        >>> clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
        >>> clf.fit(X, y)
        >>> y_pred = clf.predict(X)
        >>> print(y_pred)
    
        Notes
        -----
        The prediction process involves aggregating the predictions from all the 
        trees in the ensemble, applying the logistic function to convert the 
        aggregated prediction to probabilities, and then thresholding the 
        probabilities to obtain binary class labels.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
           X,
           accept_large_sparse=True,
           accept_sparse= True,
           to_frame=False, 
           )
        
        y_pred = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        
        for tree in self.estimators_:
            y_pred += self.eta0 * (2 * tree.predict(X) - 1)

        return np.where(1 / (1 + np.exp(-y_pred)) > 0.5, 1, 0)
  
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in `X`.
    
        This method predicts the class probabilities for the input samples in `X` 
        using the trained Boosted Decision Tree Classifier. It aggregates the 
        predictions from all the individual trees in the ensemble and applies a 
        logistic transformation to obtain the probabilities for each class.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column 
            represents a feature.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples. Column 0 contains the 
            probabilities of the negative class, and column 1 contains the 
            probabilities of the positive class.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from gofast.estimators.tree import BoostingTreeClassifier
        >>> X, y = make_classification(n_samples=100, n_features=4)
        >>> clf = BoostingTreeClassifier(n_estimators=100, max_depth=3, eta0=0.1)
        >>> clf.fit(X, y)
        >>> proba = clf.predict_proba(X)
        >>> print(proba)
    
        Notes
        -----
        The prediction process involves aggregating the predictions from all the 
        trees in the ensemble, applying the logistic function to convert the 
        aggregated prediction to probabilities. The returned probabilities 
        correspond to the negative and positive classes.
        """

        check_is_fitted(self, 'estimators_')
        X = check_array(X,accept_sparse= True,to_frame=False, 
            )
        
        y_pred = np.full(X.shape[0], self.initial_prediction_, dtype=float)
        
        for tree in self.estimators_:
            y_pred += self.eta0 * (2 * tree.predict(X) - 1)
        
        proba_positive_class = 1 / (1 + np.exp(-y_pred))  # Sigmoid function
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

class HybridBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Boosting Classifier.

    The Hybrid Boosting Classifier combines the strengths of Decision Tree
    Classifiers and Gradient Boosting Classifiers to provide a robust
    classification model. The model first fits a Decision Tree Classifier, then
    uses the residuals from this fit to train a Gradient Boosting Classifier.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of boosting stages to be run. Gradient boosting
        is fairly robust to over-fitting, so a large number usually results
        in better performance.

    eta0 : float, default=0.1
        Learning rate shrinks the contribution of each tree by `eta0`.
        There is a trade-off between learning rate and the number of
        estimators.

    max_depth : int, default=3
        The maximum depth of the individual classification estimators.
        The maximum depth limits the number of nodes in the tree. 
        Tune this parameter for best performance; the best value depends 
        on the interaction of the input features.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria
        are "gini" for the Gini impurity and "entropy" for the information
        gain.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best
        random split.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum number of
          samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split
        point at any depth will only be considered if it leaves at least
        `min_samples_leaf` training samples in each of the left and right
        branches. This may have the effect of smoothing the model, especially
        in classification.

    min_weight_fraction_leaf : float, default=0.
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node. Samples have equal weight
        when sample_weight is not provided.

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None, then unlimited number
        of leaf nodes.

    min_impurity_decrease : float, default=0.
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form `{class_label: weight}`.
        If None, all classes are supposed to have weight one. If "balanced",
        the class weights will be adjusted inversely proportional to class
        frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        `ccp_alpha` will be chosen. By default, no pruning is performed.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When `max_features` < n_features,
        the algorithm will select `max_features` at random at each split
        before finding the best split among them. Pass an int for
        reproducible output across multiple function calls.
        
    verbose : int, default=False
        Controls the verbosity when fitting
        
    Attributes
    ----------
    decision_tree_ : DecisionTreeClassifier
        The fitted decision tree classifier.
    
    gradient_boosting_ : GradientBoostingClassifier
        The fitted gradient boosting classifier.

    Examples
    --------
    >>> from gofast.estimators.boosting import HybridBoostingClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    >>> clf = HybridBoostingClassifier(n_estimators=50, eta0=0.1, max_depth=3, random_state=42)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    
    Notes
    -----
    The Hybrid Boosting Classifier uses a combination of Decision Tree Classifiers
    and Gradient Boosting Classifiers. The procedure is as follows:
    
    1. Train a Decision Tree Classifier on the input data.
    2. Train a Gradient Boosting Classifier on the input data.

    The final prediction is obtained by summing the predictions of the decision
    tree and the gradient boosting classifier.

    Mathematically, the final prediction :math:`\hat{y}` can be represented as:

    .. math::
        \hat{y} = \text{DT}(X) + \text{GB}(X)

    where:
    - :math:`\text{DT}(X)` is the prediction from the Decision Tree.
    - :math:`\text{GB}(X)` is the prediction from the Gradient Boosting model.

    """
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
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        class_weight=None, 
        ccp_alpha=0.,
        random_state=None, 
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
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.verbose = verbose 


    def fit(self, X, y, sample_weight =None ):
        """
        Fit the Hybrid Boosting Classifier model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True, estimator=self )
        
        if self.verbose:
            print("Starting fit of Hybrid Boosting Classifier")
            print("Fitting Decision Tree Classifier...")
    
        # First layer: Decision Tree Classifier
        self.decision_tree_ = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha
        )
        self.decision_tree_.fit(X, y, sample_weight=sample_weight)
    
        if self.verbose:
            print("Decision Tree Classifier fitted successfully.")
            print("Fitting Gradient Boosting Classifier...")
    
        # Second layer: Gradient Boosting Classifier
        if self.verbose:
            with tqdm(total=self.n_estimators, ascii=True,ncols= 100,
                      desc='Fitting GradientBoostingClassifier ') as pbar:
                for i in range(self.n_estimators):
                    self.gradient_boosting_ = GradientBoostingClassifier(
                        learning_rate=self.eta0,
                        # Incremental fitting by increasing the number of estimators
                        n_estimators=i+1,  
                        max_depth=self.max_depth,
                        random_state=self.random_state
                    )
                    self.gradient_boosting_.fit(X, y, sample_weight=sample_weight)
                    pbar.update(1)
        else:
            self.gradient_boosting_ = GradientBoostingClassifier(
                learning_rate=self.eta0,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.gradient_boosting_.fit(X, y, sample_weight=sample_weight)
    
        if self.verbose:
            print("\nGradient Boosting Classifier fitted successfully.")
            print("Hybrid Boosting Classifier fit completed.")
    
        return self

    def predict(self, X):
        """
        Predict class labels for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'decision_tree_')
        check_is_fitted(self, 'gradient_boosting_')
        X = check_array(X)
        
        dt_predictions = self.decision_tree_.predict(X)
        gb_predictions = self.gradient_boosting_.predict(X)
        
        # Combine predictions
        combined_predictions = dt_predictions + gb_predictions
        return np.where(combined_predictions > 1, 1, 0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'decision_tree_')
        check_is_fitted(self, 'gradient_boosting_')
        X = check_array(X)
        
        dt_proba = self.decision_tree_.predict_proba(X)
        gb_proba = self.gradient_boosting_.predict_proba(X)
        
        combined_proba = (dt_proba + gb_proba) / 2  # Normalize probabilities
        
        return combined_proba

class HybridBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Boosting Regressor.

    The Hybrid Boosting Regressor combines the strengths of Decision Tree
    Regressors and Gradient Boosting Regressors to provide a robust
    regression model. The model first fits a Decision Tree Regressor, then
    uses the residuals from this fit to train a Gradient Boosting Regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to be run. Gradient boosting
        is fairly robust to over-fitting, so a large number usually results
        in better performance.
    
    eta0 : float, default=0.1
        Learning rate shrinks the contribution of each tree by `eta0`.
        There is a trade-off between learning rate and the number of
        estimators.

    max_depth : int, default=3
        The maximum depth of the individual regression estimators.
        The maximum depth limits the number of nodes in the tree. 
        Tune this parameter for best performance; the best value depends 
        on the interaction of the input features.

    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
        default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion, "friedman_mse" for
        mean squared error with improvement score by Friedman, "absolute_error"
        for the mean absolute error, and "poisson" which uses reduction in
        Poisson deviance to find splits.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best
        random split.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum number of
          samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split
        point at any depth will only be considered if it leaves at least
        `min_samples_leaf` training samples in each of the left and right
        branches. This may have the effect of smoothing the model, especially
        in regression.

    min_weight_fraction_leaf : float, default=0.
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node. Samples have equal weight
        when sample_weight is not provided.

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None, then unlimited number
        of leaf nodes.

    min_impurity_decrease : float, default=0.
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        `ccp_alpha` will be chosen. By default, no pruning is performed.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When `max_features` < n_features,
        the algorithm will select `max_features` at random at each split
        before finding the best split among them. Pass an int for
        reproducible output across multiple function calls.
        
    verbose : int, default=False
        Controls the verbosity when fitting
        
    Attributes
    ----------
    decision_tree_ : DecisionTreeRegressor
        The fitted decision tree regressor.
    
    gradient_boosting_ : GradientBoostingRegressor
        The fitted gradient boosting regressor.

    Examples
    --------
    >>> from gofast.estimators.boosting import HybridBoostingRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    >>> reg = HybridBoostingRegressor(n_estimators=100, eta0=0.1, max_depth=3, random_state=42)
    >>> reg.fit(X, y)
    >>> y_pred = reg.predict(X)
    
    Notes
    -----
    The Hybrid Boosting Regressor uses a combination of Decision Tree Regressors
    and Gradient Boosting Regressors. The procedure is as follows:
    
    1. Train a Decision Tree Regressor on the input data.
    2. Calculate the residuals, which are the differences between the true values
       and the predictions of the decision tree.
    3. Train a Gradient Boosting Regressor on these residuals.
    
    The final prediction is obtained by summing the predictions of the decision
    tree and the gradient boosting regressor.
    
    The decision tree model is fitted first to capture the main trends in the
    data, and then the gradient boosting model is used to fit the residuals,
    thereby improving the accuracy of the model.

    Mathematically, the final prediction :math:`\hat{y}` can be represented as:

    .. math::
        \hat{y} = \text{DT}(X) + \text{GB}(X, y - \text{DT}(X))

    where:
    - :math:`\text{DT}(X)` is the prediction from the Decision Tree.
    - :math:`\text{GB}(X, y - \text{DT}(X))` is the prediction from the
      Gradient Boosting model trained on the residuals.

    """
    def __init__(
        self, 
        n_estimators=100, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        ccp_alpha=0.,
        random_state=None,
        verbose=0
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
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.verbose=verbose 

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Hybrid Boosting Regressor model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
    
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(
            X, y, 
            accept_sparse=True, 
            accept_large_sparse=True, 
            y_numeric=True, 
            estimator=self.__class__.__name__
        )
        if self.verbose:
            print("Starting fit of Hybrid Boosting Regressor")
            print("Fitting Decision Tree Regressor...")
    
        # First layer: Decision Tree Regressor
        self.decision_tree_ = DecisionTreeRegressor(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha
        )
    
        self.decision_tree_.fit(X, y, sample_weight=sample_weight)
        dt_predictions = self.decision_tree_.predict(X)
    
        if self.verbose:
            print("Decision Tree Regressor fitted successfully.")
            print("Calculating residuals...")
    
        # Calculate residuals for Gradient Boosting
        residuals = y - dt_predictions
    
        if self.verbose:
            print("Fitting Gradient Boosting Regressor...")
    
        # Second layer: Gradient Boosting Regressor
        self.gradient_boosting_ = GradientBoostingRegressor(
            learning_rate=self.eta0,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
    
        self.gradient_boosting_.fit(X, residuals, sample_weight=sample_weight)
        if self.verbose:
            with tqdm(total=self.n_estimators, ascii=True, ncols=100,
                      desc='Fitting GradientBoostingRegressor') as pbar:
                for i in range(self.n_estimators):
                    # self.gradient_boosting_.n_estimators = i + 1
                    # self.gradient_boosting_.fit(X, residuals)
                    pbar.update(1)
                    
        gb_predictions = self.gradient_boosting_.predict(X)
    
        if self.verbose:
            print("\nGradient Boosting Regressor fitted successfully.")
            print("Calculating weights for combination...")
    
        # Calculate weights based on performance (using mean squared error)
        dt_mse = mean_squared_error(y, dt_predictions)
        gb_mse = mean_squared_error(y, gb_predictions)
        
        self.dt_weight_ = 1 / dt_mse
        self.gb_weight_ = 1 / gb_mse
    
        # Normalize weights
        total_weight = self.dt_weight_ + self.gb_weight_
        self.dt_weight_ /= total_weight
        self.gb_weight_ /= total_weight
    
        if self.verbose:
            print("Hybrid Boosting Regressor fit completed.")
    
        return self
    
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
        """
        check_is_fitted(self, 'decision_tree_')
        check_is_fitted(self, 'gradient_boosting_')
        check_is_fitted(self, 'dt_weight_')
        check_is_fitted(self, 'gb_weight_')
        
        X = check_array(
            X, accept_large_sparse=True, 
            accept_sparse=True, 
            estimator=self.__class__.__name__, 
            to_frame=False, 
            input_name='X'
        )
        
        dt_predictions = self.decision_tree_.predict(X)
        gb_predictions = self.gradient_boosting_.predict(X)
        
        # Combine predictions using the calculated weights
        combined_predictions = (self.dt_weight_ * dt_predictions) + (
            self.gb_weight_ * gb_predictions)
        return combined_predictions



    












