.. _adaline:

AdalineClassifier
~~~~~~~~~~~~~~~
Binary classification using the ADALINE algorithm with batch gradient descent.

Parameters:
    - eta0 (float): Initial learning rate (default: 0.01)
    - max_iter (int): Maximum number of iterations (default: 1000)
    - tol (float): Convergence tolerance (default: 1e-4)
    - early_stopping (bool): Whether to use early stopping (default: False)
    - validation_fraction (float): Fraction of training data for validation
    - random_state (int): Random seed for reproducibility

Examples:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from gofast.estimators.adaline import AdalineClassifier

    # Example 1: Basic Binary Classification
    # Generate linearly separable data
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
        class_sep=2
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Create and train model
    adaline = AdalineClassifier(
        eta0=0.01,
        max_iter=100,
        random_state=42
    )
    adaline.fit(X_train, y_train)

    # Evaluate
    train_score = adaline.score(X_train, y_train)
    test_score = adaline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Example 2: With Early Stopping
    # Generate more challenging data
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=5,
        random_state=42
    )

    # Preprocess and split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Train with early stopping
    adaline = AdalineClassifier(
        eta0=0.01,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        tol=1e-4,
        random_state=42
    )
    adaline.fit(X_train, y_train)

    print(f"Convergence achieved at iteration: {adaline.n_iter_}")
    print(f"Final validation score: {adaline.validation_scores_[-1]:.4f}")

AdalineRegressor
~~~~~~~~~~~~~~
ADALINE implementation for regression tasks with continuous output values.

Parameters:
    - eta0 (float): Initial learning rate (default: 0.01)
    - max_iter (int): Maximum number of iterations (default: 1000)
    - tol (float): Convergence tolerance (default: 1e-4)
    - learning_rate (str): Learning rate schedule ('constant', 'adaptive')
    - eta0_decay (float): Learning rate decay factor

Examples:

.. code-block:: python

    from sklearn.datasets import make_regression
    from gofast.estimators.adaline import AdalineRegressor

    # Example 3: Regression with Adaptive Learning Rate
    # Generate regression data
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.1,
        random_state=42
    )

    # Scale features and split data
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Create and train model with adaptive learning rate
    regressor = AdalineRegressor(
        eta0=0.1,
        max_iter=500,
        learning_rate='adaptive',
        eta0_decay=0.99
    )
    regressor.fit(X_train, y_train)

    # Evaluate
    train_r2 = regressor.score(X_train, y_train)
    test_r2 = regressor.score(X_test, y_test)
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

AdalineStochasticClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~
Stochastic version of ADALINE for classification, updating weights after each sample.

Parameters:
    - eta0 (float): Initial learning rate
    - max_iter (int): Maximum number of iterations
    - shuffle (bool): Whether to shuffle training data
    - random_state (int): Random seed

Examples:

.. code-block:: python

    from gofast.estimators.adaline import AdalineStochasticClassifier
    
    # Example 4: Online Learning with Stochastic ADALINE
    # Generate streaming data simulation
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    true_weights = np.array([1, -0.5, 0.25, -0.1, 0.2])
    y = np.sign(X.dot(true_weights) + np.random.randn(n_samples) * 0.1)

    # Initialize model
    stochastic_adaline = AdalineStochasticClassifier(
        eta0=0.01,
        shuffle=True,
        random_state=42
    )

    # Simulate online learning
    batch_size = 50
    scores = []

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        if i > 0:  # Test on batch before training
            scores.append(stochastic_adaline.score(X_batch, y_batch))
            
        stochastic_adaline.partial_fit(X_batch, y_batch)

    print("Online learning performance:")
    print(f"Average accuracy: {np.mean(scores):.4f}")
    print(f"Final accuracy: {scores[-1]:.4f}")

Implementation Notes
------------------

1. **Feature Scaling**:
   - Always scale features before training
   - Use StandardScaler or MinMaxScaler
   - Apply same scaling to training and test data

2. **Learning Rate Selection**:
   - Start with small learning rate (0.01)
   - Use adaptive learning rate for complex problems
   - Monitor convergence behavior

3. **Convergence Monitoring**:
   - Use early stopping for large datasets
   - Monitor validation scores
   - Check for oscillation in training error

4. **Performance Optimization**:
   - Use stochastic versions for large datasets
   - Implement mini-batch processing
   - Consider parallel processing for large-scale applications

See Also
--------
- :mod:`gofast.estimators.perceptron`: Basic perceptron implementation
- :mod:`gofast.preprocessing`: Data preprocessing utilities
- :mod:`gofast.metrics`: Performance metrics

References
----------
.. [1] Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits.
       IRE WESCON Convention Record, Part 4, 96-104.

.. [2] Widrow, B., & Lehr, M. A. (1990). 30 years of adaptive neural networks:
       Perceptron, madaline, and backpropagation. Proceedings of the IEEE, 78(9), 1415-1442.

.. [3] Zhang, X. (2019). A comprehensive review of stability analysis of 
       continuous-time recurrent neural networks. IEEE Transactions on Neural Networks
       and Learning Systems, 30(4), 1229-1262.