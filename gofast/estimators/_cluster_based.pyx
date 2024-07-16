# -*- coding: utf-8 -*-

from numbers import Integral, Real
from abc import ABCMeta
from abc import abstractmethod

from sklearn.base import BaseEstimator, clone
from sklearn.utils._param_validation import Hidden, HasMethods 
from sklearn.utils._param_validation  import Interval, StrOptions
from ..tools.validator import check_is_fitted
from ..transformers import KMeansFeaturizer
from .util import fit_with_estimator, select_best_model 

class BaseKMF(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for KMeansFeaturizer estimators.

    This class provides the foundation for KMeansFeaturizer-based 
    classifiers and regressors. It combines k-means clustering with a 
    base machine learning estimator to enhance the performance of the 
    estimator by leveraging the additional structure introduced by the 
    clustering process.

    Parameters
    ----------
    n_clusters : int, default=7
        The number of clusters to form in the k-means clustering process.
        This parameter controls the granularity of the clustering and can
        significantly impact the model performance.

    target_scale : float, default=1.0
        Scaling factor for the target variable when included in the k-means
        clustering process. This is used to adjust the influence of the target
        variable in the clustering process. A higher value increases the
        impact of the target variable.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the centroid initialization and data 
        shuffling, ensuring reproducibility of the results. If an int is
        provided, it is used as the seed for the random number generator.

    n_components : int, optional
        Number of principal components to retain for dimensionality reduction
        before applying k-means clustering. If not set, no dimensionality 
        reduction is performed. This helps in reducing the complexity of
        the input data and can improve clustering performance.

    init : str, callable or array-like, default='k-means++'
        Method for initialization of the k-means centroids. Options include:
        - 'k-means++': selects initial cluster centers in a smart way to 
          speed up convergence.
        - 'random': chooses `n_clusters` observations (rows) at random 
          from the data for the initial centroids.

    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm will be run with different 
        centroid seeds. The best output in terms of inertia is chosen. 
        Setting this to a higher value increases the robustness of the 
        clustering.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
        This controls the convergence of the algorithm. If the algorithm 
        does not converge within the specified number of iterations, it stops.

    tol : float, default=1e-4
        Relative tolerance with regards to the Frobenius norm of the difference
        in the cluster centers to declare convergence. This determines the 
        stopping criterion of the k-means algorithm.

    copy_x : bool, default=True
        If True, the input data is centered before computation. Centering the 
        data is known to improve the numerical stability of the k-means 
        algorithm.

    verbose : int, default=0
        Verbosity mode for logging the process of the k-means algorithm. Higher
        values result in more detailed logging.

    algorithm : str, default='lloyd'
        K-means algorithm variant to use. The options include:
        - 'lloyd': standard k-means algorithm.
        - 'elkan': more efficient variant using the triangle inequality.

    estimator : estimator object, default=None
        The base machine learning estimator to fit on the transformed data. 
        This estimator should follow the scikit-learn estimator interface. 
        If None, a default estimator is used.

    to_sparse : bool, default=False
        If True, the input data `X` will be converted to a sparse matrix before
        applying the transformation. This is useful for handling large datasets
        more efficiently. If False, the data format of `X` is preserved.
        
     encoding : {'onehot', 'bin-counting', 'label', 'frequency', 'mean_target'},\
         default='onehot'
         Encoding strategy for cluster labels:
         - 'onehot': One-hot encoding of the categorical variables. This creates 
           a binary column for each category and assigns a 1 or 0 based on 
           whether the category is present in the sample.
         - 'bin-counting': Probabilistic bin-counting encoding. This converts 
           categorical values into probabilities based on their frequency of 
           occurrence in the dataset.
         - 'label': Label encoding of the categorical variables. This assigns 
           a unique integer to each category.
         - 'frequency': Frequency encoding of the categorical variables. This 
           assigns the frequency of each category's occurrence in the dataset 
           as the encoded value.
         - 'mean_target': Mean target encoding based on target values provided 
           during fit. This assigns the mean of the target variable for each 
           category.

    Notes
    -----
    The k-means clustering algorithm involves minimizing the inertia, or 
    within-cluster sum-of-squares criterion:

    .. math::

        \text{Inertia} = \sum_{i=1}^{n} (x_i - \mu_k)^2

    where :math:`x_i` represents a data point, :math:`\mu_k` is the centroid 
    of the cluster k, and the sum is taken over all the data points in the 
    cluster.

    When the target variable `y` is included, the feature space is augmented 
    by stacking `y` with `X`. The KMeansFeaturizer then minimizes a modified 
    objective function:

    .. math::

        \min_{C_1,\cdots,C_k, \mu'_1,\cdots,\mu'_k} \sum_{i=1}^{k}\\
            \left( \sum_{x y \in C_i} \|x y - \mu'_i \|_2^2 \right)

    where:
    - :math:`x y` denotes the augmented data point, a combination of the 
      feature vector `x` and the target `y`. This combination guides the 
      clustering process by considering both feature characteristics and 
      target variable information.
    - :math:`\mu'_i` is the centroid of the augmented data points in the 
      cluster :math:`C_i`, including target variable information, aligning
      the clusters with the underlying class structure.
    - The first summation :math:`\sum_{i=1}^{k}` sums over all k-clusters, 
      focusing on minimizing the variance within each cluster.
    - The inner summation :math:`\sum_{x y \in C_i}` represents the sum over 
      all augmented data points :math:`x y` in the cluster :math:`C_i`.
    - :math:`\|x y - \mu'_i \|_2^2` is the squared Euclidean distance between
      each augmented data point :math:`x y` and its cluster centroid 
      :math:`\mu'_i`. Minimizing this distance ensures points are as close 
      as possible to their cluster center.

    - The effectiveness of the KMeansFeaturizer depends on the choice of the 
      base estimator and the suitability of k-means clustering for the given 
      dataset.
    - The KMeansFeaturizer can optionally scale the target variable and perform 
      PCA for dimensionality reduction, which can significantly affect the 
      clustering results and hence the final prediction accuracy.
      
    Examples
    --------
    >>> from gofast.estimators._base import BaseKMF
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> class MyKMFClassifier(BaseKMF):
    ...     estimator = DecisionTreeClassifier()
    >>> X, y = make_classification(n_samples=100, n_features=20, 
                                   random_state=42)
    >>> model = MyKMFClassifier(n_clusters=5)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)

    See Also
    --------
    sklearn.cluster.KMeans : The k-means clustering implementation used.
    sklearn.base.BaseEstimator : The base class for all estimators in scikit-learn.

    References
    ----------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
           K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
           Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
    .. [2] MacQueen, J. (1967). Some methods for classification and analysis 
           of multivariate observations. Proceedings of the 5th Berkeley 
           Symposium on Mathematical Statistics and Probability. 1:281-297.
    .. [3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
           Python. Journal of Machine Learning Research. 12:2825-2830.
    """
    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"k-means++", "random"}), callable, "array-like"],
        "n_init": [
            StrOptions({"auto"}),
            Hidden(StrOptions({"warn"})),
            Interval(Integral, 1, None, closed="left"),
        ],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "verbose": [Interval(Integral, 0, None, closed="left"), "boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "random_state": ["random_state"],
        "copy_x": [bool],
        "algorithm": [
            StrOptions({"lloyd", "elkan", "auto", "full"}, deprecated={"auto", "full"})],
        "estimator": [HasMethods(["fit", "predict"]), None],
        "to_sparse": ["boolean"], 
        "encoding": [ StrOptions(
            {'onehot', 'bin-counting', 'label', 'frequency', 'mean_target'}), 
            None],
        }
    
    @abstractmethod
    def __init__(
        self,
        n_clusters=7,
        target_scale=1.0,
        random_state=None,
        n_components=None,
        init='k-means++',
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        copy_x=True,
        verbose=0,
        algorithm='lloyd',
        estimator=None,
        to_sparse=False,
        encoding=None
    ):
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.copy_x = copy_x
        self.verbose = verbose
        self.algorithm = algorithm
        self.estimator = estimator
        self.to_sparse = to_sparse
        self.encoding=encoding 
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the KMeansFeaturizer and the base estimator.
    
        This method first validates the input data `X` and `y`. It then fits 
        the `KMeansFeaturizer` to transform the input data and subsequently 
        fits the base estimator on the transformed data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and 
            `n_features` is the number of features. Each row corresponds to 
            a single sample, and each column corresponds to a feature.
    
        y : array-like of shape (n_samples,)
            Target values. Each element in the array corresponds to the target 
            value for a sample.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. This 
            parameter can be used to give more importance to certain samples 
            during fitting.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fitting process involves two main steps:
    
        1. **Transformation**:
           The input data `X` is transformed into cluster memberships using the 
           k-means clustering algorithm:
           .. math::
               \text{Cluster Assignments} = \arg\min_k \| x_i - \mu_k \|_2
           where :math:`x_i` is a data point, and :math:`\mu_k` is the centroid 
           of cluster `k`.
    
        2. **Estimation**:
           The transformed data is used to fit the base estimator:
           .. math::
               \hat{y} = \text{BaseEstimator}(\text{Transformed Data})
       
        - The `fit` method must be called before `predict` or `predict_proba`.
        - Input data `X` and `y` are validated to ensure they meet the required 
          format and conditions.
        - The effectiveness of the fitting process depends on both the clustering 
          quality and the suitability of the base estimator for the given dataset.
          
        Examples
        --------
        >>> from gofast.estimators.cluster_based import KMFClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=100, n_features=20, 
                                       random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> kmf_classifier = KMFClassifier(n_clusters=5)
        >>> kmf_classifier.fit(X_train, y_train)

        See Also
        --------
        BaseKMF._fit_featurizer : Fit the KMeansFeaturizer on the input data.
        BaseKMF._fit_estimator : Fit the base estimator on the transformed data.
        KMFClassifier.predict : Predict class labels for samples in `X`.
        KMFClassifier.predict_proba : Predict class probabilities for samples in `X`.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        .. [2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
               Python. Journal of Machine Learning Research. 12:2825-2830.
        """
        self._validate_params() 
        check_X_params = dict(accept_sparse="csc")
        check_y_params = dict(ensure_2d=False, dtype=None)
        X, y = self._validate_data(
            X, y, validate_separately=(check_X_params, check_y_params)
        )
        # Hide the default estimator so make a copy instead. 
        self.estimator_ = self.estimator 
        if self.estimator_ is None: 
            self.estimator_ = select_best_model(
                X, y, self.estimator_,
                problem = ( 
                    "regression" if self._estimator_type=='regressor' 
                    else "classification") 
                )
        X_transformed = self._fit_featurizer(X, y)
        self._fit_estimator(X_transformed, y, sample_weight )
        return self
    
    def predict(self, X):
        """
        Predict class labels or target values for samples in `X`.
    
        The `predict` method transforms the input data using the 
        `KMeansFeaturizer` and then applies the base estimator to predict the 
        class labels or target values.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples. Each row corresponds to a single sample, and each 
            column corresponds to a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels or target values for each sample.
    
        Notes
        -----
        The prediction process involves two main steps:
    
        1. **Transformation**:
           The input data `X` is transformed into cluster memberships using the 
           k-means clustering algorithm:
           .. math::
               \text{Cluster Assignments} = \arg\min_k \| x_i - \mu_k \|_2
           where :math:`x_i` is a data point, and :math:`\mu_k` is the centroid 
           of cluster `k`.
    
        2. **Prediction**:
           The transformed data is used to predict the class labels or target 
           values using the base estimator:
           .. math::
               \hat{y} = \text{BaseEstimator}(\text{Transformed Data})

        - Ensure that the `fit` method has been called before invoking `predict`.
        - The performance of the predictions depends on both the clustering 
          quality and the effectiveness of the base estimator.
        - This method checks if the model is fitted and raises a `NotFittedError` 
          if the model is not fitted.
          
        Examples
        --------
        >>> from gofast.estimators.cluster_based import KMFClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=100, n_features=20, 
                                       random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> kmf_classifier = KMFClassifier(n_clusters=5)
        >>> kmf_classifier.fit(X_train, y_train)
        >>> y_pred = kmf_classifier.predict(X_test)
    
    
        See Also
        --------
        BaseKMF.fit : Fit the KMeansFeaturizer and the base estimator.
        KMFClassifier.predict_proba : Predict class probabilities for samples in `X`.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        .. [2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
               Python. Journal of Machine Learning Research. 12:2825-2830.
        """
        check_is_fitted(self, ["estimator_", "featurizer_"])
        X_transformed = self.featurizer_.transform(X)
        return self.estimator_.predict(X_transformed)
    
    def _fit_featurizer(self, X, y):
        """
        Fit the KMeansFeaturizer on the input data.
    
        This method initializes and fits the `KMeansFeaturizer` to transform 
        the input data `X` and target variable `y` into cluster memberships. 
        The featurizer enhances the input data by capturing the inherent 
        clustering structure.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and 
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_clusters)
            Transformed input data, where `n_clusters` is the number of clusters 
            formed by the k-means algorithm.
    
        Mathematical Formulation
        ------------------------
        The k-means clustering algorithm involves minimizing the inertia, or 
        within-cluster sum-of-squares criterion:
    
        .. math::
    
            \text{Inertia} = \sum_{i=1}^{n} (x_i - \mu_k)^2
    
        where :math:`x_i` represents a data point, and :math:`\mu_k` is the 
        centroid of the cluster k.
    
        Examples
        --------
        >>> from gofast.estimators.cluster_based import BaseKMF
        >>> X, y = make_classification(n_samples=100, n_features=20, 
                                       random_state=42)
        >>> base_kmf = BaseKMF(n_clusters=5)
        >>> X_transformed = base_kmf._fit_featurizer(X, y)
    
        Notes
        -----
        - The effectiveness of the transformation depends on the number of 
          clusters and the quality of the k-means clustering.
    
        See Also
        --------
        sklearn.cluster.KMeans : The k-means clustering implementation used.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        """
        self.featurizer_ = KMeansFeaturizer(
            n_clusters=self.n_clusters,
            target_scale=self.target_scale,
            random_state=self.random_state,
            n_components=self.n_components,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            copy_x=self.copy_x,
            verbose=self.verbose,
            algorithm=self.algorithm,
            to_sparse=self.to_sparse,
            encoding= self.encoding
        )
        return self.featurizer_.fit_transform(X, y)

    def _fit_estimator(self, X_transformed, y, sample_weight=None):
        """
        Fit the base estimator on the transformed data.
    
        This method initializes and fits the base estimator on the data 
        transformed by the `KMeansFeaturizer`. If no estimator is specified, 
        it defaults to the `base_estimator`.
    
        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_clusters)
            Transformed input data.
    
        y : array-like of shape (n_samples,)
            Target values.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. This 
            parameter can be used to give more importance to certain samples 
            during fitting.
            
        Returns
        -------
        None
    
        Examples
        --------
        >>> from gofast.estimators.cluster_based import BaseKMF
        >>> X, y = make_classification(n_samples=100, n_features=20, 
                                       random_state=42)
        >>> base_kmf = BaseKMF(n_clusters=5)
        >>> X_transformed = base_kmf._fit_featurizer(X, y)
        >>> base_kmf._fit_estimator(X_transformed, y)
    
        Notes
        -----
        - Ensure that the transformed data is correctly generated by the 
          `KMeansFeaturizer` before fitting the estimator.
    
        See Also
        --------
        sklearn.base.BaseEstimator : The base class for all estimators in 
            scikit-learn.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        """
        self.estimator_ = clone(self.estimator_)
        self.estimator_ = fit_with_estimator(
            self.estimator_, X_transformed, y, sample_weight= sample_weight )
        # self.estimator_.fit(X_transformed, y)
