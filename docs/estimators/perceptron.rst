.. _perceptron:

Perceptron Models
================

.. currentmodule:: gofast.estimators.perceptron

The :mod:`gofast.estimators.perceptron` module implements various perceptron-based learning algorithms, including standard perceptron, gradient descent variants, and specialized perceptron models for both classification and regression tasks.

Background & Motivation
---------------------
The perceptron, introduced by Frank Rosenblatt in 1957, is one of the foundational algorithms in machine learning. It represents the first artificial neural network model and provides the basis for many modern neural architectures. Its importance stems from:

1. **Historical Significance**:
   - First trainable neural network model
   - Pioneered online learning concepts
   - Foundation for deep learning

2. **Theoretical Importance**:
   - Demonstrates linear separability
   - Provides convergence guarantees
   - Illustrates gradient-based learning

3. **Practical Applications**:
   - Binary classification
   - Online learning scenarios
   - Feature learning foundations

Theoretical Foundation
--------------------

The perceptron model computes its output using:

.. math::

    f(x) = \begin{cases} 
    1 & \text{if } w^T x + b > 0 \\
    -1 & \text{otherwise}
    \end{cases}

The weight update rule is:

.. math::

    w_{t+1} = w_t + \eta y_i x_i

where:
- w is the weight vector
- η is the learning rate
- y_i is the true label
- x_i is the input vector

For the gradient descent variant:

.. math::

    w_{t+1} = w_t - \eta \nabla E(w_t)

where E(w) is the error function.

Classes Overview
--------------

Perceptron
~~~~~~~~~
Basic perceptron implementation with various learning options.

Parameters:
    - eta0 (float): Initial learning rate
    - max_iter (int): Maximum number of iterations
    - tol (float): Convergence tolerance
    - early_stopping (bool): Whether to use early stopping
    - validation_fraction (float): Fraction of training data for validation
    - n_iter_no_change (int): Number of iterations with no improvement
    - random_state (int): Random seed for reproducibility

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from gofast.estimators.perceptron import Perceptron

    # Example 1: Basic Perceptron Classification
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

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create and train perceptron
    ppn = Perceptron(
        eta0=0.01,
        max_iter=100,
        random_state=42
    )
    ppn.fit(X_train, y_train)

    # Evaluate
    y_pred = ppn.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Example 2: Perceptron with Early Stopping
    # Generate more challenging data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=10,
        random_state=42
    )

    # Preprocess and split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train with early stopping
    ppn = Perceptron(
        eta0=0.01,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
        random_state=42
    )
    ppn.fit(X_train, y_train)

    print(f"\nConverged in {ppn.n_iter_} iterations")
    print(f"Final validation score: {ppn.validation_scores_[-1]:.4f}")

LightGDPerceptron
~~~~~~~~~~~~~~~
Light gradient descent perceptron with optimized learning rate scheduling.

Parameters:
    - eta0 (float): Initial learning rate
    - max_iter (int): Maximum iterations
    - batch_size (int): Mini-batch size
    - learning_rate (str): Learning rate schedule
    - momentum (float): Momentum coefficient
    - random_state (int): Random seed

Examples:

.. code-block:: python

    from gofast.estimators.perceptron import LightGDPerceptron
    import matplotlib.pyplot as plt

    # Example 3: GD Perceptron with Learning Rate Scheduling
    # Generate nonlinearly separable data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42
    )

    # Preprocess and split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train with different learning rate schedules
    schedules = ['constant', 'adaptive', 'inverse_scaling']
    results = {}

    for schedule in schedules:
        model = LightGDPerceptron(
            eta0=0.1,
            learning_rate=schedule,
            batch_size=32,
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        results[schedule] = {
            'train_scores': model.train_scores_,
            'test_score': model.score(X_test, y_test)
        }

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    for schedule in schedules:
        plt.plot(
            results[schedule]['train_scores'],
            label=f'{schedule} (test={results[schedule]["test_score"]:.4f})'
        )
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Learning Curves for Different Schedules')
    plt.legend()
    plt.grid(True)
    plt.show()

MultiLayerPerceptron
~~~~~~~~~~~~~~~~~~
Multi-layer perceptron implementation with customizable architecture.

Parameters:
    - hidden_layers (list): Sizes of hidden layers
    - activation (str): Activation function
    - solver (str): Optimization algorithm
    - learning_rate (float): Learning rate
    - max_iter (int): Maximum iterations
    - random_state (int): Random seed

Examples:

.. code-block:: python

    from gofast.estimators.perceptron import MultiLayerPerceptron
    from sklearn.preprocessing import LabelBinarizer

    # Example 4: Multi-class Classification with MLP
    # Generate multi-class data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    # Preprocess data
    X_scaled = StandardScaler().fit_transform(X)
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    # Create and train MLP
    mlp = MultiLayerPerceptron(
        hidden_layers=[64, 32],
        activation='relu',
        solver='adam',
        learning_rate=0.001,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_train, y_train)

    # Evaluate
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(mlp.loss_curve_, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

Implementation Details
--------------------

1. **Weight Initialization**:
   - Zero initialization
   - Random initialization
   - Xavier/Glorot initialization

2. **Learning Rate Scheduling**:
   - Constant learning rate
   - Adaptive learning rate
   - Time-based decay

3. **Optimization Techniques**:
   - Stochastic gradient descent
   - Mini-batch processing
   - Momentum optimization

Best Practices
-------------

1. **Data Preparation**:
   - Feature scaling
   - Shuffling training data
   - Handling missing values

2. **Model Selection**:
   - Architecture optimization
   - Learning rate tuning
   - Early stopping criteria

3. **Performance Optimization**:
   - Batch size selection
   - Learning rate scheduling
   - Regularization techniques

Advanced Usage
------------

1. **Custom Activation Functions**:

.. code-block:: python

    def custom_activation(x):
        return np.tanh(x) * np.sigmoid(x)

    mlp = MultiLayerPerceptron(
        hidden_layers=[64, 32],
        activation=custom_activation
    )

2. **Learning Rate Scheduling**:

.. code-block:: python

    def custom_lr_schedule(initial_lr, epoch):
        return initial_lr / (1 + epoch * 0.01)

    ppn = LightGDPerceptron(
        eta0=0.1,
        learning_rate=custom_lr_schedule
    )

3. **Custom Loss Functions**:

.. code-block:: python

    def custom_loss(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    mlp = MultiLayerPerceptron(
        hidden_layers=[64, 32],
        loss=custom_loss
    )

Advantages and Limitations
------------------------

Advantages:
   - Simple and interpretable
   - Fast training on linear problems
   - Online learning capability
   - Guaranteed convergence for linearly separable data

Limitations:
   - Limited to linear decision boundaries (single layer)
   - Sensitive to feature scaling
   - No probabilistic outputs
   - May not converge for non-linearly separable data

Visualization Tools
-----------------

1. **Decision Boundary Visualization**:

.. code-block:: python

    def plot_decision_boundary(X, y, model):
        h = 0.02  # step size
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h),
            np.arange(y_min, y_max, h)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary')
        plt.show()

2. **Learning Curve Visualization**:

.. code-block:: python

    from sklearn.model_selection import learning_curve

    def plot_learning_curves(X, y, model):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 
                label='Training Score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 
                label='Validation Score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

See Also
--------
- :mod:`gofast.estimators.adaline`: ADALINE implementation
- :mod:`gofast.neural`: Neural network components
- :mod:`gofast.optimization`: Optimization algorithms

References
----------
.. [1] Rosenblatt, F. (1958). The perceptron: a probabilistic model for
       information storage and organization in the brain. 
       Psychological Review, 65(6), 386.

.. [2] Minsky, M., & Papert, S. (1969). Perceptrons: An Introduction to
       Computational Geometry. MIT Press.

.. [3] Widrow, B., & Lehr, M. A. (1990). 30 years of adaptive neural networks:
       Perceptron, Madaline, and backpropagation. 
       Proceedings of the IEEE, 78(9), 1415-1442.