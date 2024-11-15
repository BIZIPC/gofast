.. _cluster_based:

Cluster-Based Models
===================

.. currentmodule:: gofast.estimators.cluster_based

The :mod:`gofast.estimators.cluster_based` module implements machine learning models that leverage clustering techniques for classification and regression tasks. These models combine clustering algorithms with supervised learning approaches to create powerful hybrid models that can capture complex data patterns.

Background & Motivation
---------------------
Cluster-based models represent a unique approach to machine learning that combines unsupervised and supervised learning techniques. By first identifying natural groupings in the data through clustering, these models can:

1. **Capture Local Patterns**:
   - Learn different patterns for different data regions
   - Adapt to heterogeneous data distributions
   - Handle non-linear relationships effectively

2. **Improve Generalization**:
   - Reduce impact of outliers
   - Handle multimodal distributions
   - Better model complex decision boundaries

3. **Enhance Interpretability**:
   - Provide insights into data structure
   - Allow analysis of cluster-specific patterns
   - Support localized feature importance analysis

Theoretical Foundation
--------------------

The cluster-based approach follows a two-step process:

1. **Clustering Phase**:
   Partition the input space into K clusters using a clustering algorithm:

   .. math::

      C = \{C_1, C_2, ..., C_K\}

   where each cluster Cᵢ represents a subset of the input space.

2. **Local Model Phase**:
   For each cluster, train a local model:

   .. math::

      f_k(x) = \text{LocalModel}_k(x), \quad x \in C_k

The final prediction combines local models:

.. math::

   F(x) = \sum_{k=1}^K w_k(x)f_k(x)

where w_k(x) are combining weights based on cluster membership or distance.

Key Features
-----------
- **Adaptive Learning**: Models adapt to local data characteristics
- **Flexibility**: Supports various clustering and local model combinations
- **Scalability**: Can be parallelized across clusters
- **Interpretability**: Provides insights through cluster analysis

Classes Overview
--------------

KMFClassifier
~~~~~~~~~~~~
K-Means Based Fuzzy Classifier that combines k-means clustering with local classification models.

Parameters:
    - n_clusters (int): Number of clusters (default: 3)
    - base_estimator (object): Base classifier for local models
    - membership_type (str): Type of fuzzy membership ('distance', 'probability')
    - alpha (float): Membership smoothing parameter
    - random_state (int): Random seed for reproducibility

Attributes:
    - cluster_centers_ (array): Learned cluster centers
    - local_models_ (list): List of fitted local classifiers
    - cluster_sizes_ (array): Number of samples in each cluster

Examples:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from gofast.estimators.cluster_based import KMFClassifier

    # Example 1: Basic Usage
    # Generate non-linearly separable data
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=3,
        random_state=42
    )

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create and train model
    kmf = KMFClassifier(
        n_clusters=5,
        base_estimator=LogisticRegression(),
        membership_type='distance',
        random_state=42
    )
    kmf.fit(X_train, y_train)

    # Evaluate
    train_score = kmf.score(X_train, y_train)
    test_score = kmf.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Example 2: Analyzing Cluster Properties
    print("\nCluster Analysis:")
    for i, size in enumerate(kmf.cluster_sizes_):
        local_score = kmf.local_models_[i].score(
            kmf.get_cluster_data(X_test, i),
            kmf.get_cluster_labels(y_test, i)
        )
        print(f"Cluster {i}:")
        print(f"  Size: {size}")
        print(f"  Local Model Accuracy: {local_score:.4f}")

KMFRegressor
~~~~~~~~~~~
K-Means Based Fuzzy Regressor for complex regression tasks using local models.

Parameters:
    - n_clusters (int): Number of clusters
    - base_estimator (object): Base regressor for local models
    - membership_type (str): Type of fuzzy membership
    - smoothing (float): Prediction smoothing parameter
    - random_state (int): Random seed

Mathematical Expression:

The final prediction is computed as:

.. math::

    \hat{y}(x) = \frac{\sum_{k=1}^K \mu_k(x)f_k(x)}{\sum_{k=1}^K \mu_k(x)}

where:
- μₖ(x) is the membership degree of x to cluster k
- fₖ(x) is the prediction of the k-th local model

Examples:

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from gofast.estimators.cluster_based import KMFRegressor

    # Example 3: Complex Regression Task
    # Generate non-linear regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        random_state=42
    )
    y = np.sin(y) + 0.1 * np.random.randn(1000)  # Add non-linearity

    # Preprocess and split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create and train model
    kmf_reg = KMFRegressor(
        n_clusters=5,
        base_estimator=Ridge(),
        membership_type='probability',
        smoothing=0.5,
        random_state=42
    )
    kmf_reg.fit(X_train, y_train)

    # Evaluate
    train_r2 = kmf_reg.score(X_train, y_train)
    test_r2 = kmf_reg.score(X_test, y_test)
    print(f"\nTraining R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

    # Example 4: Local Model Analysis
    from sklearn.metrics import mean_squared_error

    print("\nLocal Model Performance:")
    for i in range(kmf_reg.n_clusters):
        X_cluster = kmf_reg.get_cluster_data(X_test, i)
        y_cluster = kmf_reg.get_cluster_labels(y_test, i)
        if len(X_cluster) > 0:
            y_pred = kmf_reg.local_models_[i].predict(X_cluster)
            mse = mean_squared_error(y_cluster, y_pred)
            print(f"Cluster {i} MSE: {mse:.4f}")

Implementation Details
--------------------

1. **Clustering Strategy**:
   - K-means is used for initial clustering
   - Clusters are balanced using size constraints
   - Empty clusters are handled gracefully

2. **Membership Computation**:
   - Distance-based: Using inverse distance weighting
   - Probability-based: Using softmax of distances
   - Adaptive weighting based on cluster sizes

3. **Local Model Management**:
   - Parallel training of local models
   - Handling of small clusters
   - Model persistence and updating

Best Practices
-------------

1. **Model Selection**:
   - Choose appropriate number of clusters
   - Select suitable base estimators
   - Consider data distribution

2. **Parameter Tuning**:
   - Optimize cluster count
   - Adjust membership parameters
   - Fine-tune base estimators

3. **Performance Optimization**:
   - Scale features appropriately
   - Handle outliers
   - Use cross-validation

Advantages and Limitations
------------------------

Advantages:
   - Handles non-linear patterns
   - Adapts to local data structure
   - Provides interpretable results

Limitations:
   - Sensitive to cluster count
   - Requires sufficient data per cluster
   - Computational complexity

See Also
--------
- :mod:`gofast.estimators.ensemble`: Ensemble learning methods
- :mod:`gofast.cluster`: Additional clustering algorithms
- :mod:`gofast.metrics`: Performance evaluation metrics

References
----------
.. [1] Pedrycz, W. (2005). Knowledge-Based Clustering: From Data to 
       Information Granules. John Wiley & Sons.

.. [2] Yang, M. S., & Nataliani, Y. (2017). Robust-learning fuzzy c-means 
       clustering algorithm with unknown number of clusters. Pattern 
       Recognition, 71, 45-59.

.. [3] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements 
       of Statistical Learning. Springer.