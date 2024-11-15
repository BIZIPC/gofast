.. _ensemble:

Ensemble Learning Models
======================

.. currentmodule:: gofast.estimators.ensemble

The :mod:`gofast.estimators.ensemble` module provides a comprehensive suite of ensemble learning methods, combining multiple base estimators to create robust and high-performing models. These ensemble methods leverage different combination strategies to improve model performance and reliability.

Background & Motivation
---------------------
Ensemble learning is a powerful approach that combines multiple models to create a stronger predictor. The key advantages include:

1. **Reduced Overfitting**:
   - Averaging reduces variance
   - Combines different model biases
   - More robust generalization

2. **Improved Accuracy**:
   - Leverages strengths of multiple models
   - Handles complex patterns
   - Better decision boundaries

3. **Enhanced Stability**:
   - Reduced sensitivity to noise
   - More reliable predictions
   - Better handling of outliers

Theoretical Foundation
--------------------

The ensemble prediction can be expressed as:

For classification (majority voting):
.. math::

    \hat{y} = \text{mode}\{f_1(x), f_2(x), ..., f_M(x)\}

For regression (weighted average):
.. math::

    \hat{y} = \sum_{i=1}^M w_i f_i(x)

where:
- M is the number of base estimators
- wᵢ are the weights
- fᵢ(x) are individual model predictions

Classes Overview
--------------

MajorityVoteClassifier
~~~~~~~~~~~~~~~~~~~~
Implements majority voting for classification tasks.

Parameters:
    - classifiers (list): List of classifier objects
    - voting (str): Voting strategy ('hard' or 'soft')
    - weights (array-like): Model weights for weighted voting
    - n_jobs (int): Number of parallel jobs

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from gofast.estimators.ensemble import MajorityVoteClassifier

    # Example 1: Basic Majority Voting
    # Generate dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create base classifiers
    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    clf3 = SVC(probability=True, random_state=42)

    # Create and train ensemble
    mv_clf = MajorityVoteClassifier(
        classifiers=[clf1, clf2, clf3],
        voting='soft'
    )
    mv_clf.fit(X_train, y_train)

    # Evaluate
    for clf, name in zip([clf1, clf2, clf3, mv_clf],
                        ['LR', 'DT', 'SVC', 'Ensemble']):
        print(f"{name} Test Score: {clf.score(X_test, y_test):.4f}")

WeightedAverageRegressor
~~~~~~~~~~~~~~~~~~~~~~
Implements weighted averaging for regression tasks.

Parameters:
    - regressors (list): List of regressor objects
    - weights (array-like): Model weights
    - n_jobs (int): Number of parallel jobs

Examples:

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from gofast.estimators.ensemble import WeightedAverageRegressor

    # Example 2: Weighted Average Regression
    # Generate dataset
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )

    # Preprocess and split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create base regressors
    reg1 = Ridge(alpha=1.0)
    reg2 = Lasso(alpha=1.0)
    reg3 = ElasticNet(alpha=1.0, l1_ratio=0.5)

    # Create and train ensemble
    wa_reg = WeightedAverageRegressor(
        regressors=[reg1, reg2, reg3],
        weights=[0.4, 0.3, 0.3]
    )
    wa_reg.fit(X_train, y_train)

    # Evaluate
    for reg, name in zip([reg1, reg2, reg3, wa_reg],
                        ['Ridge', 'Lasso', 'ElasticNet', 'Ensemble']):
        print(f"{name} Test R²: {reg.score(X_test, y_test):.4f}")

SimpleAverageEnsemble
~~~~~~~~~~~~~~~~~~~
Implements simple averaging for both classification and regression.

Parameters:
    - estimators (list): List of estimator objects
    - task (str): 'classification' or 'regression'
    - n_jobs (int): Number of parallel jobs

Examples:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from gofast.estimators.ensemble import SimpleAverageEnsemble

    # Example 3: Simple Average Ensemble
    # Create base estimators
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(random_state=42)

    # Create and train ensemble
    sa_ensemble = SimpleAverageEnsemble(
        estimators=[rf, gb, lr],
        task='classification'
    )
    sa_ensemble.fit(X_train, y_train)

    # Evaluate with cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(
        sa_ensemble, X_scaled, y,
        cv=5, scoring='accuracy'
    )
    print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

DynamicWeightedEnsemble
~~~~~~~~~~~~~~~~~~~~~
Implements dynamic weight adjustment based on recent performance.

Parameters:
    - base_estimators (list): List of estimator objects
    - window_size (int): Size of performance history window
    - adaptation_rate (float): Weight adaptation rate
    - task (str): 'classification' or 'regression'

Examples:

.. code-block:: python

    from sklearn.neural_network import MLPClassifier
    from gofast.estimators.ensemble import DynamicWeightedEnsemble

    # Example 4: Dynamic Weighted Ensemble
    # Create diverse base estimators
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(100,50), random_state=42))
    ]

    # Create and train ensemble
    dw_ensemble = DynamicWeightedEnsemble(
        base_estimators=estimators,
        window_size=10,
        adaptation_rate=0.1
    )
    dw_ensemble.fit(X_train, y_train)

    # Monitor weight evolution
    print("\nFinal Estimator Weights:")
    for est, weight in zip(estimators, dw_ensemble.current_weights_):
        print(f"{est[0]}: {weight:.4f}")


Implementation Details
--------------------

1. **Weight Management**:
   - Static weights specification
   - Dynamic weight adaptation
   - Performance-based weighting

2. **Parallel Processing**:
   - Efficient parallel prediction
   - Shared memory management
   - Load balancing

3. **Error Handling**:
   - Model compatibility checks
   - Prediction shape validation
   - Missing value handling

Best Practices
-------------

1. **Ensemble Design**:
   - Use diverse base models
   - Balance model complexity
   - Consider resource constraints

2. **Weight Selection**:
   - Start with equal weights
   - Use cross-validation for optimization
   - Monitor weight stability

3. **Performance Optimization**:
   - Feature preprocessing
   - Parameter tuning
   - Regular retraining

Advanced Usage
------------

1. **Custom Base Estimators**:

.. code-block:: python

    from sklearn.base import BaseEstimator, ClassifierMixin
    
    class CustomEstimator(BaseEstimator, ClassifierMixin):
        def __init__(self, param1=1):
            self.param1 = param1
            
        def fit(self, X, y):
            # Implementation
            return self
            
        def predict(self, X):
            # Implementation
            return predictions

    # Use in ensemble
    custom_ensemble = MajorityVoteClassifier(
        classifiers=[CustomEstimator(), clf1, clf2]
    )

2. **Weight Optimization**:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV
    
    # Define weight grid
    param_grid = {
        'weights': [[0.2, 0.3, 0.5],
                   [0.3, 0.3, 0.4],
                   [0.4, 0.3, 0.3]]
    }
    
    # Optimize weights
    grid_search = GridSearchCV(
        WeightedAverageRegressor(regressors=[reg1, reg2, reg3]),
        param_grid,
        cv=5
    )
    grid_search.fit(X_train, y_train)

3. **Performance Monitoring**:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Track individual model performances
    performances = []
    for clf in mv_clf.classifiers_:
        scores = cross_val_score(clf, X_scaled, y, cv=5)
        performances.append(scores)
    
    # Visualize
    plt.boxplot(performances, labels=['CLF1', 'CLF2', 'CLF3'])
    plt.title('Model Performance Distribution')
    plt.ylabel('Accuracy')
    plt.show()

Advantages and Limitations
------------------------

Advantages:
   - Improved prediction stability
   - Better generalization
   - Reduced overfitting risk

Limitations:
   - Increased computational cost
   - More complex model management
   - Potential for redundancy

See Also
--------
- :mod:`gofast.estimators.boosting`: Boosting algorithms
- :mod:`gofast.model_selection`: Model selection utilities
- :mod:`gofast.metrics`: Performance evaluation metrics

References
----------
.. [1] Zhou, Z. H. (2012). Ensemble Methods: Foundations and Algorithms. 
       Chapman and Hall/CRC.

.. [2] Kuncheva, L. I. (2014). Combining Pattern Classifiers: Methods 
       and Algorithms. Wiley.

.. [3] Rokach, L. (2010). Ensemble-based classifiers. Artificial 
       Intelligence Review, 33(1-2), 1-39.