# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
from numbers import Integral, Real
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils._param_validation import Interval, StrOptions

from ._tree import  BaseDTB, BaseWeightedTree 
from ..tools.validator import check_array 
from ..tools.validator import check_is_fitted


__all__=[ "DTBRegressor", "DTBClassifier", 
         "WeightedTreeClassifier", "WeightedTreeRegressor",]

class DTBRegressor(BaseDTB, RegressorMixin):
    """
    Decision Tree-based Regressor for Regression Tasks.

    The `DTBRegressor` employs an ensemble approach, combining multiple 
    Decision Regression Trees to form a more robust regression model. Each 
    tree in the ensemble independently predicts the outcome, and the final 
    prediction is derived by averaging these individual predictions. This 
    method is effective in reducing variance and improving prediction accuracy 
    over a single decision tree.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble.
        
    max_depth : int, default=None
        The maximum depth of each regression tree. If `None`, then nodes are 
        expanded until all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.
        
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
        default="squared_error"
        The function to measure the quality of a split. Supported criteria 
        are "squared_error" for mean squared error, "friedman_mse" for 
        mean squared error with improvement score by Friedman, "absolute_error" 
        for mean absolute error, and "poisson" for Poisson deviance.
        
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
        the input samples) required to be at a leaf node.
        
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
        
    subsample : float, default=1.0
        The fraction of samples to be used for fitting each base learner.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling 
        without replacement is performed.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    oob_score_ : float
        Out-of-bag score for the training dataset.

    Examples
    --------
    Here's an example of how to use the `DTBRegressor` on a dataset:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.tree import DTBRegressor

    >>> X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=0)
    
    >>> reg = DTBRegressor(n_estimators=50, max_depth=3, verbose=1)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)

    Notes
    -----
    This model effectively combines the predictive power of multiple trees 
    through averaging, reducing variance and improving accuracy over a 
    single decision tree. The ensemble prediction is computed by averaging 
    the predictions from each individual Regression Tree within the ensemble, 
    as follows:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from 
      all trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the 
      :math:`i`-th Regression Tree.

    The use of bootstrap sampling and subsampling enhances model robustness 
    and provides an estimate of model performance through out-of-bag (OOB) 
    score when subsampling is enabled.

    See Also
    --------
    sklearn.ensemble.RandomForestRegressor : A popular ensemble method 
        based on decision trees for regression tasks.
    sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as 
        base learners in ensemble methods.
    sklearn.metrics.mean_squared_error : A common metric for evaluating 
        regression models.
    gofast.estimators.tree.BoostedRegressionTree : An enhanced BRT.
    """
    
    _parameter_constraints: dict = {
        **BaseDTB._parameter_constraints,
        "criterion": [StrOptions(
            {"squared_error", "friedman_mse", "absolute_error", "poisson"})],
        "subsample": [Interval(Real, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), "boolean"],
        "bootstrap": ["boolean"],
    }
    
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=None, 
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
        subsample=1.0, 
        bootstrap=True,
        verbose=0
    ):
        super().__init__(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            criterion=criterion, 
            splitter=splitter, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            ccp_alpha=ccp_alpha, 
            verbose=verbose
        )
        self.subsample = subsample
        self.bootstrap = bootstrap
        
    def _create_tree(self):
        """
        Create a new instance of `DecisionTreeRegressor`.
    
        This method initializes a `DecisionTreeRegressor` with the parameters
        specified in the `DTBRegressor` instance. The created decision tree
        will be used as a base estimator within the ensemble.
    
        Returns
        -------
        tree : DecisionTreeRegressor
            A new instance of a decision tree regressor.
    
        Parameters
        ----------
        max_depth : int, default=None
            The maximum depth of the tree. If `None`, then nodes are expanded
            until all leaves are pure or until all leaves contain less than
            `min_samples_split` samples.
    
        criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
            default="squared_error"
            The function to measure the quality of a split. Supported criteria are
            "squared_error" for mean squared error, "friedman_mse" for mean squared
            error with improvement score by Friedman, "absolute_error" for mean absolute
            error, and "poisson" for Poisson deviance.
    
        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported strategies
            are "best" to choose the best split and "random" to choose the best
            random split.
    
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and 
            `ceil(min_samples_split * n_samples)` are the minimum number of samples
            for each split.
    
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node:
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and 
            `ceil(min_samples_leaf * n_samples)` are the minimum number of samples
            for each node.
    
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights required to 
            be at a leaf node. Samples have equal weight when sample_weight is 
            not provided.
    
        max_features : int, float, str or None, default=None
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and 
            `int(max_features * n_features)` features are considered at each split.
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
    
        Examples
        --------
        Here's an example of how to use the `_create_tree` method within `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBRegressor
        >>> reg = DTBRegressor(max_depth=4, criterion='absolute_error', random_state=42)
        >>> tree = reg._create_tree()
        >>> print("Created tree:", tree)
    
        See Also
        --------
        sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as
            the base learner in the ensemble.
    
        References
        ----------
        .. [1] Breiman, L. "Random forests." Machine learning 45.1 (2001): 5-32.
        .. [2] Quinlan, J. R. "C4.5: programs for machine learning." Elsevier, 2014.
        """
        return DecisionTreeRegressor(
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
    
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from multiple decision trees.
    
        This method combines the predictions from all individual trees in
        the ensemble. For regression tasks, the predictions are averaged
        to form the final output.
    
        Parameters
        ----------
        predictions : array-like of shape (n_estimators, n_samples)
            The predictions from all individual trees.
    
        Returns
        -------
        aggregated_predictions : array-like of shape (n_samples,)
            The aggregated predictions.
    
        Notes
        -----
        The aggregation of predictions for regression tasks can be mathematically
        described as:
    
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
        Here's an example of how to use the `_aggregate_predictions` 
        method within `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBRegressor
        >>> import numpy as np
        >>> reg = DTBRegressor(n_estimators=3)
        >>> predictions = np.array([[2.0, 3.0], [2.5, 2.5], [3.0, 3.5]])
        >>> aggregated_predictions = reg._aggregate_predictions(predictions)
        >>> print("Aggregated predictions:", aggregated_predictions)
    
        See Also
        --------
        sklearn.ensemble.BaggingRegressor : A bagging regressor that also uses
            aggregation of predictions from multiple base estimators.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
        """
        return np.mean(predictions, axis=0)

class DTBClassifier(BaseDTB, ClassifierMixin):
    """
    Decision Tree-Based Classifier.

    This classifier leverages an ensemble of decision trees to enhance prediction 
    accuracy and robustness over individual tree performance. It aggregates the 
    predictions from multiple DecisionTreeClassifier instances, using majority 
    voting to determine the final classification. This method is particularly 
    effective at reducing overfitting and increasing the generalization ability 
    of the model.

    The fit method trains multiple DecisionTreeClassifier models on the entire dataset. 
    The predict method then aggregates their predictions through majority voting 
    to determine the final class labels. The ensemble's behavior can be customized 
    through parameters controlling the number and depth of trees.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the ensemble.
        
    max_depth : int, default=3
        The maximum depth of each tree in the ensemble.
    random_state : int, optional
        Controls the randomness of the tree building process and the bootstrap 
        sampling of the data points (if bootstrapping is used).
    
    criterion : str, default="gini"
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
        (min_samples_split * n_samples) are the minimum number of samples 
        for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split 
        point at any depth will only be considered if it leaves at least 
        min_samples_leaf training samples in each of the left and right branches.
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and ceil
        (min_samples_leaf * n_samples) are the minimum number of samples for 
        each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input 
        samples) required to be at a leaf node. Samples have equal weight when 
        sample_weight is not provided.
        
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
        Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None, then unlimited 
        number of leaf nodes.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.
        
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than ccp_alpha
        will be chosen.
    
    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        A list containing the fitted decision tree classifiers.

    Notes
    -----
    The final classification decision is made by aggregating the predictions 
    from all the decision trees and selecting the class with the majority vote:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input \(x\).
    - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the 
      occurrence of the most frequent class label predicted by the \(i\)-th tree.
    - :math:`n` is the number of decision trees in the ensemble.
    - :math:`C_i(x)` is the class label predicted by the \(i\)-th decision tree.

    This ensemble approach significantly reduces the likelihood of overfitting and 
    increases the predictive performance by leveraging the diverse predictive 
    capabilities of multiple models.

    Example
    -------
    Here's an example of how to use the `DTBClassifier`:

    >>> from gofast.estimators.tree import DTBClassifier
    >>> rtec = DTBClassifier(n_estimators=100, max_depth=3,
    ...                                     random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> rtec.fit(X, y)
    >>> y_pred = rtec.predict(X)

    Applications
    ------------
    The Decision Tree-Based Classifier is suitable for various classification
    tasks, such as spam detection, sentiment analysis, and medical diagnosis.

    Performance
    -----------
    The model's performance depends on the quality of the data, the number of
    estimators (trees), and the depth of each tree. Hyperparameter tuning may
    be necessary to optimize performance.

    See Also
    --------
    - `sklearn.ensemble.RandomForestClassifier`: Scikit-learn's Random Forest
      Classifier for ensemble-based classification.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.

    """
    _parameter_constraints: dict = {
        **BaseDTB._parameter_constraints,
        "criterion": [StrOptions({"gini", "entropy"})],
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
        "subsample": [Interval(Real, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
        "bootstrap": [bool],
    }
    
    def __init__(
        self, 
        n_estimators=100, 
        max_depth=None, 
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
        subsample=1.0,
        bootstrap=True,
        verbose=0 
      ):
        super().__init__(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            criterion=criterion, 
            splitter=splitter, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            ccp_alpha=ccp_alpha, 
            verbose=verbose
        )
        self.class_weight = class_weight
        self.subsample=subsample 
        self.bootstrap = bootstrap

    def _create_tree(self):
        """
        This method initializes a new `DecisionTreeClassifier` instance with the
        parameters specified in the class constructor. It is used to create the
        individual decision trees that will form the ensemble.

        Returns
        -------
        tree : `DecisionTreeClassifier`
            A new instance of `DecisionTreeClassifier` configured with the parameters
            provided in the class constructor.

        Notes
        -----
        The `DecisionTreeClassifier` is a fundamental component of the ensemble,
        and its configuration affects the overall performance of the classifier.
        Key parameters include the maximum depth of the tree (`max_depth`), 
        the criterion for splitting nodes (`criterion`), and others as specified
        in the constructor.

        .. math::
            H(X) = -\sum_{i=1}^{k} p_i \log(p_i)

        where:
        - :math:`H(X)` is the entropy of the node.
        - :math:`p_i` is the proportion of samples belonging to class :math:`i`
        at the node.

        Examples
        --------
        >>> from gofast.estimators.ensemble import _create_tree
        >>> dtc = _create_tree()
        >>> print(dtc)

        See Also
        --------
        `sklearn.tree.DecisionTreeClassifier` :
            Scikit-learn's Decision Tree Classifier.

        References
        ----------
        .. [1] Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
        """
        return DecisionTreeClassifier(
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

    def _aggregate_predictions(self, predictions):
        """
        This method aggregates the predictions from multiple decision trees 
        through majority voting. 
        
        Each decision tree in the ensemble makes a prediction for each sample,
         and the final class label is determined by the majority vote of
        these predictions.

        Parameters
        ----------
        predictions : array-like of shape (n_estimators, n_samples)
            Predictions from each decision tree in the ensemble.

        Returns
        -------
        majority_vote : ndarray of shape (n_samples,)
            The final class labels determined by majority voting.

        Notes
        -----
        Majority voting is a robust method for combining predictions from an ensemble
        of classifiers. It reduces the variance of the final prediction by averaging
        out the individual biases of the base classifiers.

        .. math::
            C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

        where:
        - :math:`C_{\text{final}}(x)` is the final predicted class label for
          input :math:`x`.
        - :math:`\delta(C_i(x), \text{mode})` is an indicator function that 
          counts the occurrence of the most frequent class label predicted by
          the :math:`i`-th tree.
        - :math:`n` is the number of decision trees in the ensemble.
        - :math:`C_i(x)` is the class label predicted by the :math:`i`-th 
          decision tree.

        Examples
        --------
        >>> from gofast.estimators.ensemble import _aggregate_predictions
        >>> predictions = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
        >>> majority_vote = _aggregate_predictions(predictions)
        >>> print(majority_vote)

        See Also
        --------
        `sklearn.ensemble.RandomForestClassifier` : Scikit-learn's Random Forest
          Classifier for ensemble-based classification.

        References
        ----------
        .. [1] Breiman, L. (1996). Bagging predictors. Machine learning, 24(2), 123-140.
        """
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_vote
    
    def predict_proba(self, X):
        """
        Predict Probabilities
    
        Predict class probabilities for the input samples using the 
        ensemble of decision trees.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
    
        Notes
        -----
        The predicted probabilities are obtained by averaging the probabilities 
        predicted by each decision tree in the ensemble. This approach leverages 
        the strengths of each individual classifier to provide a more robust 
        probability estimate.
    
        .. math::
            P(C_k | x) = \frac{1}{n} \sum_{i=1}^{n} P(C_k | x, \theta_i)
    
        where:
        - :math:`P(C_k | x)` is the predicted probability of class :math:`k` 
          given input :math:`x`.
        - :math:`n` is the number of decision trees in the ensemble.
        - :math:`P(C_k | x, \theta_i)` is the probability predicted by the 
          :math:`i`-th decision tree with parameters :math:`\theta_i`.
    
        Examples
        --------
        >>> from gofast.estimators.ensemble import DTBClassifier
        >>> X = np.random.rand(10, 4)
        >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
        >>> clf.fit(X, np.random.randint(0, 2, 10))
        >>> probas = clf.predict_proba(X)
        >>> print(probas)
    
        See Also
        --------
        `sklearn.ensemble.RandomForestClassifier.predict_proba` : Predict class
          probabilities with a Random Forest classifier.
    
        References
        ----------
        .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements
           of statistical learning. Springer series in statistics, 1(10).
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse=True,
            to_frame=False, 
        )
        probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

class WeightedTreeClassifier(BaseWeightedTree, ClassifierMixin):
    """
    Weighted Tree Classifier.

    The `WeightedTreeClassifier` is an ensemble learning model that 
    combines multiple decision trees using gradient boosting techniques 
    to tackle binary classification tasks. By integrating multiple 
    decision trees with varying weights, this classifier achieves high 
    accuracy and reduces the risk of overfitting.

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
        equal weight when `sample_weight` is not provided.

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

    verbose : bool, default=False
        Controls the verbosity of the fitting process. If True, the progress
        of the fitting process is displayed.

    Attributes
    ----------
    base_estimators_ : list of DecisionTreeClassifier
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

    **Initialization**:
    .. math::
        F_0(x) = 0

    **Iteration for each tree**:
    1. Compute the weighted error:
       .. math::
           \epsilon_m = \frac{\sum_{i=1}^{n} w_i \cdot I(y_i \neq h_m(x_i))}
                        {\sum_{i=1}^{n} w_i}

    2. Compute the weight of the tree:
       .. math::
           \alpha_m = \eta_0 \cdot \log\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)

    3. Update the sample weights:
       .. math::
           w_i = w_i \cdot \exp(\alpha_m \cdot I(y_i \neq h_m(x_i)))

    where:
    - :math:`F_m(x)` is the prediction after `m` trees.
    - :math:`\eta_0` is the learning rate.
    - :math:`h_m(x)` is the prediction of the `m`-th tree.
    - :math:`\epsilon_m` is the weighted error.
    - :math:`\alpha_m` is the weight of the `m`-th tree.
    - :math:`w_i` is the weight of sample `i`.

    Examples
    --------
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
        **BaseWeightedTree._parameter_constraints,
        "criterion": [StrOptions({"gini", "entropy"})],
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
    }
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
        class_weight=None, 
        ccp_alpha=0., 
        verbose=False
        ):
        super().__init__(
            n_estimators=n_estimators, 
            eta0=eta0, 
            max_depth=max_depth, 
            criterion=criterion, 
            splitter=splitter, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            verbose=verbose
        )
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
    
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
    
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        return self
        
    def _make_estimator(self):
        return DecisionTreeClassifier(
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

    def _is_classifier(self):
        return True

class WeightedTreeRegressor(BaseWeightedTree, RegressorMixin):
    """
    Weighted Tree Regressor.

    The `WeightedTreeRegressor` is an ensemble learning model that combines
    multiple decision trees using gradient boosting techniques to tackle
    regression tasks. By integrating multiple decision trees with varying
    weights, this regressor achieves high accuracy and reduces the risk of
    overfitting.

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

    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
        default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are:
        - "squared_error": Mean squared error, which is equal to variance 
          reduction as feature selection criterion.
        - "friedman_mse": Mean squared error with improvement score by Friedman.
        - "absolute_error": Mean absolute error.
        - "poisson": Poisson deviance as feature selection criterion.

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
        equal weight when `sample_weight` is not provided.

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
    base_estimators_ : list of DecisionTreeRegressor
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

    **Initialization**:
    .. math::
        F_0(x) = 0

    **Iteration for each tree**:
    .. math::
        r_i = y_i - F_{m-1}(x_i)

    .. math::
        F_m(x) = F_{m-1}(x) + \eta_0 \cdot f_m(x)

    where:
    - :math:`F_m(x)` is the prediction after `m` trees.
    - :math:`\eta_0` is the learning rate.
    - :math:`f_m(x)` is the prediction of the `m`-th tree.
    - :math:`r_i` is the residual for sample :math:`i`.

    Examples
    --------
    >>> from gofast.estimators.tree import WeightedTreeRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split

    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> reg = WeightedTreeRegressor(n_estimators=10, eta0=0.1, max_depth=3)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)

    

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: Scikit-learn's Gradient
      Boosting Regressor for comparison.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.

    References
    ----------
    .. [1] Friedman, J.H. "Greedy Function Approximation: A Gradient Boosting
           Machine," The Annals of Statistics, 2001.
    .. [2] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
           Python," Journal of Machine Learning Research, 12:2825-2830.
    """
    _parameter_constraints: dict = {
        **BaseWeightedTree._parameter_constraints,
        "criterion": [StrOptions(
            {"squared_error", "friedman_mse", "absolute_error", "poisson"})],
        "verbose": [Interval(Integral, 0, None, closed="left"), bool],
    }
    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
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
        verbose=False
        ):
        super().__init__(
            n_estimators=n_estimators, 
            eta0=eta0, 
            max_depth=max_depth, 
            criterion=criterion, 
            splitter=splitter, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf, 
            min_weight_fraction_leaf=min_weight_fraction_leaf, 
            max_features=max_features, 
            random_state=random_state, 
            max_leaf_nodes=max_leaf_nodes, 
            min_impurity_decrease=min_impurity_decrease, 
            verbose=verbose
        )

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
        The fitting process involves different steps for regression tasks:
    
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
        """
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            check_input=check_input,
        )
        return self
    
    def _make_estimator(self):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            criterion=self.criterion, 
            splitter=self.splitter, 
            min_samples_split=self.min_samples_split, 
            min_samples_leaf=self.min_samples_leaf, 
            min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
            max_features=self.max_features, 
            random_state=self.random_state, 
            max_leaf_nodes=self.max_leaf_nodes, 
            min_impurity_decrease=self.min_impurity_decrease
        )

    def _is_classifier(self):
        return False


















