# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
`cluster_based` module offers cluster-based machine learning models
leveraging clustering techniques to enhance predictive performance.
"""

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score

from ..tools.validator import check_is_fitted
from ._cluster_based import BaseKMF

__all__=["KMFClassifier", "KMFRegressor"]


class KMFClassifier(BaseKMF, ClassifierMixin):
    """
    KMeans Featurizer Classifier (KMFClassifier).

    A classifier that integrates k-means clustering with a base machine 
    learning estimator. The `KMFClassifier` first employs the 
    `KMeansFeaturizer` algorithm to transform the input data into cluster 
    memberships based on k-means clustering [1]_. Each data point is 
    represented by its closest cluster center. This transformed data, 
    capturing the inherent clustering structure, is then used to train a 
    specified base estimator [2]_. The approach aims to enhance the 
    performance of the base estimator by leveraging the additional structure 
    introduced by the clustering process.

    See more in :ref:`User Guide`. 
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form in the k-means clustering process. This 
        parameter controls the granularity of the clustering and can 
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
        reduction is performed. This helps in reducing the complexity of the 
        input data and can improve clustering performance.

    init : str, callable or array-like, default='k-means++'
        Method for initialization of the k-means centroids. Options include:
        - 'k-means++': selects initial cluster centers in a smart way to speed 
          up convergence.
        - 'random': chooses `n_clusters` observations (rows) at random from the 
          data for the initial centroids.

    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm will be run with different 
        centroid seeds. The best output in terms of inertia is chosen. Setting 
        this to a higher value increases the robustness of the clustering.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run. 
        This controls the convergence of the algorithm. If the algorithm does 
        not converge within the specified number of iterations, it stops.

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
        This estimator should follow the scikit-learn estimator interface. If 
        None, a default `DecisionTreeClassifier` is used.

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
    - The effectiveness of the KMFClassifier depends on the choice of the base 
      estimator and the suitability of k-means clustering for the given 
      dataset.
    - The KMeansFeaturizer can optionally scale the target variable and 
      perform PCA for dimensionality reduction, which can significantly affect 
      the clustering results and hence the final prediction accuracy.

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
      :math:`\mu'_i`. Minimizing this distance ensures points are as close as 
      possible to their cluster center.

    Examples
    --------
    >>> from gofast.estimators.cluster_based import KMFClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> kmf_classifier = KMFClassifier(n_clusters=5)
    >>> kmf_classifier.fit(X_train, y_train)
    >>> y_pred = kmf_classifier.predict(X_test)
    >>> kmf_classifier.score (X_test, y_test)

    See Also
    --------
    sklearn.cluster.KMeans : The k-means clustering implementation used.
    sklearn.base.BaseEstimator : The base class for all estimators in 
        scikit-learn.

    References
    ----------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
           K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
           Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
    .. [2] MacQueen, J. (1967). Some methods for classification and analysis of 
           multivariate observations. Proceedings of the 5th Berkeley Symposium 
           on Mathematical Statistics and Probability. 1:281-297.
    .. [3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
           Python. Journal of Machine Learning Research. 12:2825-2830.
    """

    def __init__(
        self,
        n_clusters=3,
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
        super().__init__(
            n_clusters=n_clusters,
            target_scale=target_scale,
            random_state=random_state,
            n_components=n_components,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            copy_x=copy_x,
            verbose=verbose,
            algorithm=algorithm,
            estimator=estimator,
            to_sparse=to_sparse, 
            encoding =encoding
        )

    def predict(self, X):
        """
        Predict class labels for samples in `X`.
    
        The `predict` method transforms the input data using the 
        `KMeansFeaturizer` and then applies the base classifier to predict the 
        class labels.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples. Each row corresponds to a single sample, and each 
            column corresponds to a feature.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for each sample.
    
        Notes
        -----
        - Ensure that the `fit` method has been called before invoking `predict`.
        - The performance of the predictions depends on both the clustering 
          quality and the effectiveness of the base classifier.

        The prediction process involves two main steps:
        1. **Transformation**:
           The input data `X` is transformed into cluster memberships using the 
           k-means clustering algorithm:
           .. math::
               \text{Cluster Assignments} = \arg\min_k \| x_i - \mu_k \|_2
           where :math:`x_i` is a data point, and :math:`\mu_k` is the centroid 
           of cluster `k`.
    
        2. **Prediction**:
           The transformed data is used to predict the class labels using the 
           base classifier:
           .. math::
               \hat{y} = \text{BaseClassifier}(\text{Transformed Data})
    
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
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in `X`.
    
        The `predict_proba` method transforms the input data using the 
        `KMeansFeaturizer` and then applies the base classifier to predict the 
        class probabilities.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples. Each row corresponds to a single sample, and each 
            column corresponds to a feature.
    
        Returns
        -------
        y_prob : array-like of shape (n_samples, n_classes)
            Predicted class probabilities for each sample. The columns correspond 
            to the probability of each class.
    
        Notes
        -----
        - Ensure that the `fit` method has been called before invoking 
          `predict_proba`.
        - The base classifier must support the `predict_proba` method. If the base 
          classifier does not support probability predictions, a 
          `NotImplementedError` is raised.

        The probability prediction process involves two main steps:
        1. **Transformation**:
           The input data `X` is transformed into cluster memberships using the 
           k-means clustering algorithm:
           .. math::
               \text{Cluster Assignments} = \arg\min_k \| x_i - \mu_k \|_2
           where :math:`x_i` is a data point, and :math:`\mu_k` is the centroid 
           of cluster `k`.
    
        2. **Probability Prediction**:
           The transformed data is used to predict the class probabilities using 
           the base classifier:
           .. math::
               P(\hat{y} = c | x_i) = \text{BaseClassifier.predict_proba}(\text{Transformed Data})
    
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
        >>> y_prob = kmf_classifier.predict_proba(X_test)
    
        
        See Also
        --------
        KMFClassifier.predict : Predict class labels for samples in `X`.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        .. [2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
               Python. Journal of Machine Learning Research. 12:2825-2830.
        """
        check_is_fitted(self, ["estimator_", "featurizer_"])
        if hasattr(self.estimator_, "predict_proba"):
            X_transformed = self.featurizer_.transform(X)
            return self.estimator_.predict_proba(X_transformed)
        else:
            raise NotImplementedError(
                f"{self.estimator_.__class__.__name__} does not support predict_proba"
            )
            
class KMFRegressor(BaseKMF, RegressorMixin):
    """
    K-Means Featurizer Regressor (KMFRegressor).

    A regressor that integrates k-means clustering with a base machine 
    learning estimator. The `KMFRegressor` first employs the 
    `KMeansFeaturizer` algorithm to transform the input data into cluster 
    memberships based on k-means clustering [1]_. Each data point is 
    represented by its closest cluster center. This transformed data, 
    capturing the inherent clustering structure, is then used to train a 
    specified base estimator [2]_. The approach aims to enhance the 
    performance of the base estimator by leveraging the additional structure 
    introduced by the clustering process.
    
    See More in :ref:`User Guide`. 
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of clusters to form in the k-means clustering process. This 
        parameter controls the granularity of the clustering and can 
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
        reduction is performed. This helps in reducing the complexity of the 
        input data and can improve clustering performance.

    init : str, callable or array-like, default='k-means++'
        Method for initialization of the k-means centroids. Options include:
        - 'k-means++': selects initial cluster centers in a smart way to speed 
          up convergence.
        - 'random': chooses `n_clusters` observations (rows) at random from the 
          data for the initial centroids.

    n_init : int or 'auto', default='auto'
        Number of times the k-means algorithm will be run with different 
        centroid seeds. The best output in terms of inertia is chosen. Setting 
        this to a higher value increases the robustness of the clustering.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run. 
        This controls the convergence of the algorithm. If the algorithm does 
        not converge within the specified number of iterations, it stops.

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
        This estimator should follow the scikit-learn estimator interface. If 
        None, a default `DecisionTreeRegressor` is used.

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
    - The effectiveness of the KMFRegressor depends on the choice of the base 
      estimator and the suitability of k-means clustering for the given 
      dataset.
    - The KMeansFeaturizer can optionally scale the target variable and 
      perform PCA for dimensionality reduction, which can significantly affect 
      the clustering results and hence the final prediction accuracy.
 
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
      :math:`\mu'_i`. Minimizing this distance ensures points are as close as 
      possible to their cluster center.

    Examples
    --------
    >>> from gofast.estimators.cluster_based import KMFRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> kmf_regressor = KMFRegressor(n_clusters=5)
    >>> kmf_regressor.fit(X_train, y_train)
    >>> kmf_regressor.score (X_test, y_test)
    
    See Also
    --------
    sklearn.cluster.KMeans : The k-means clustering implementation used.
    sklearn.base.BaseEstimator : The base class for all estimators in 
        scikit-learn.

    References
    ----------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
           K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
           Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
    .. [2] MacQueen, J. (1967). Some methods for classification and analysis of 
           multivariate observations. Proceedings of the 5th Berkeley Symposium 
           on Mathematical Statistics and Probability. 1:281-297.
    .. [3] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
           Python. Journal of Machine Learning Research. 12:2825-2830.
    """

    def __init__(
        self,
        n_clusters=3,
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
        super().__init__(
            n_clusters=n_clusters,
            target_scale=target_scale,
            random_state=random_state,
            n_components=n_components,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            copy_x=copy_x,
            verbose=verbose,
            algorithm=algorithm,
            estimator=estimator,
            to_sparse=to_sparse,
            encoding=encoding
        )
        
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.
    
        The `score` method computes the R^2 score, also known as the 
        coefficient of determination, which provides a measure of how well the 
        predicted values match the actual values. The best possible score is 
        1.0, which indicates perfect prediction. This method first makes 
        predictions on the input data `X` using the trained model and then 
        compares these predictions with the actual values `y` to compute the 
        R^2 score.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples, where `n_samples` is the number of samples and 
            `n_features` is the number of features. Each row corresponds to a 
            single sample, and each column corresponds to a feature.
    
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`. Each element in the array corresponds to the 
            actual value for a sample.
    
        Returns
        -------
        score : float
            R^2 of `self.predict(X)` with respect to `y`. The R^2 score is a 
            statistical measure that represents the proportion of the variance 
            for a dependent variable that's explained by an independent variable 
            or variables in a regression model.
    
        Notes
        -----
        The R^2 score, or coefficient of determination, is computed as follows:
    
        .. math::
    
            R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}
    
        where:
        - :math:`y_i` are the true values,
        - :math:`\\hat{y}_i` are the predicted values,
        - :math:`\\bar{y}` is the mean of the true values,
        - :math:`n` is the number of samples.
  
        - The R^2 score used here is not a symmetric function and hence can be 
          negative if the model is worse than a simple mean predictor.
        - The score method evaluates the performance of the model on the test 
          dataset. Higher scores indicate a model that better captures the 
          variability of the dataset.
        - Ensure that the `fit` method has been called before invoking `score`.
    
        Examples
        --------
        >>> from gofast.estimators.cluster_based import KMFRegressor
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> kmf_regressor = KMFRegressor(n_clusters=5)
        >>> kmf_regressor.fit(X_train, y_train)
        >>> r2 = kmf_regressor.score(X_test, y_test)
    
        See Also
        --------
        sklearn.metrics.r2_score : The function to compute the R^2 score.
        KMFRegressor.fit : Fit the KMeansFeaturizer and the base estimator.
        KMFRegressor.predict : Predict target values for samples in `X`.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
               K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
               Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
        .. [2] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in 
               Python. Journal of Machine Learning Research. 12:2825-2830.
        """
        check_is_fitted(self, ["estimator_", "featurizer_"])
        predictions = self.predict(X)
        return r2_score(y, predictions, sample_weight= sample_weight )






