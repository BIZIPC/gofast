# -*- coding: utf-8 -*-

from abc import abstractmethod
from numbers import Integral, Real
from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions
from ..api.property import LearnerMeta 

class BaseBoostingTree(BaseEstimator, metaclass=LearnerMeta):
    """
    Base class for Boosting Tree algorithms.

    The `BaseBoostingTree` serves as an abstract base class for implementing 
    boosting algorithms with decision trees. Boosting is an ensemble technique 
    that iteratively adds weak learners (typically decision trees) to correct 
    the errors made by the previous models. This abstract class provides the 
    foundational structure, including parameter validation, for specific 
    boosting implementations like `BoostingTreeRegressor` and 
    `BoostingTreeClassifier`.

    The boosting process can be mathematically represented as follows:

    .. math::
        F_{t+1}(x) = F_t(x) + \eta \cdot h_t(x),

    where:
    - :math:`F_t(x)` is the current prediction at iteration `t`.
    - :math:`\eta` is the learning rate (also known as ``eta0`` in the code).
    - :math:`h_t(x)` is the weak learner's prediction at iteration `t`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to be run, i.e., the number of trees 
        to be included in the ensemble. Increasing `n_estimators` improves 
        model accuracy but can lead to overfitting.

    eta0 : float, default=0.1
        Learning rate that scales the contribution of each weak learner. 
        A smaller `eta0` requires more trees to achieve the same effect 
        but can lead to better generalization.

    max_depth : int or None, default=3
        The maximum depth of each decision tree. A greater `max_depth` 
        allows the tree to capture more complex patterns but increases the 
        risk of overfitting. If `None`, the tree depth is unlimited.

    criterion : {"squared_error", "friedman_mse", "gini", "entropy"}, 
                default="squared_error"
        The function to measure the quality of a split:
        - ``squared_error`` and ``friedman_mse`` are used for regression tasks.
        - ``gini`` and ``entropy`` are used for classification tasks.
        This criterion helps the model to determine the best split points 
        within each tree.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies:
        - ``best``: Selects the best possible split.
        - ``random``: Selects the best random split. Using a random splitter 
          can increase model robustness by adding variance.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node. 
        If an integer is provided, it's interpreted as the absolute number 
        of samples. If a float is given, it's interpreted as a fraction 
        of the total number of samples.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. Like 
        `min_samples_split`, if a float is given, it's interpreted as a 
        fraction of the total number of samples.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all 
        the input samples) required to be at a leaf node.

    max_features : int, float, str, or None, default=None
        The number of features to consider when looking for the best split:
        - If int, considers `max_features` features at each split.
        - If float, `max_features` is a fraction, and int(`max_features` * 
          n_features) features are considered at each split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If `None`, then `max_features=n_features`.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the estimator. If an integer is provided, 
        it ensures reproducible results across different runs.

    max_leaf_nodes : int or None, default=None
        Limit the number of leaf nodes in the tree. If `None`, the number 
        of leaf nodes is unlimited.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than 
        `ccp_alpha` will be chosen.

    verbose : int, default=0
        Controls the verbosity of the output. If `verbose=1`, the fitting 
        process will display progress messages.

    Notes
    -----
    Boosting is highly effective for reducing both bias and variance in 
    models, making it a powerful method for regression and classification 
    tasks. It sequentially builds an ensemble of weak learners, adjusting 
    for the errors made by previous models.

    References
    ----------
    .. [1] Y. Freund and R. E. Schapire, "A Decision-Theoretic Generalization 
           of On-Line Learning and an Application to Boosting," 1997.
    .. [2] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting 
           Machine," Annals of Statistics, 2001.
    .. [3] T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical 
           Learning," Springer, 2009.

    Examples
    --------
    >>> from gofast.estimators._boosting import BaseBoostingTree
    >>> class ExampleBoostingTree(BaseBoostingTree):
    ...     def __init__(self, n_estimators=50, eta0=0.1):
    ...         super().__init__(n_estimators=n_estimators, eta0=eta0)
    >>> model = ExampleBoostingTree(n_estimators=10, eta0=0.05)
    >>> model.n_estimators
    10
    >>> model.eta0
    0.05

    See Also
    --------
    sklearn.ensemble.GradientBoostingRegressor : Gradient Boosting for 
        regression tasks.
    sklearn.ensemble.AdaBoostClassifier : Adaptive Boosting for classification.

    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "eta0": [Interval(Real, 0.0, 1.0, closed="both")],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "splitter": [StrOptions({"best", "random"})],
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
            StrOptions({"auto", "sqrt", "log2"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
        "verbose": [bool, Interval(Integral, 0, None, closed="left")],
    }

    @abstractmethod
    def __init__(
        self,
        n_estimators=100,
        eta0=0.1,
        max_depth=3,
        criterion="squared_error",
        splitter="best",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        verbose=False,
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
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose
