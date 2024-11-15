.. _dynamic_system:

Dynamic System Models
===================

.. currentmodule:: gofast.estimators.dynamic_system

The :mod:`gofast.estimators.dynamic_system` module implements dynamic system identification models, particularly focusing on Hammerstein-Wiener models for both classification and regression tasks. These models are especially powerful for systems with both static nonlinearities and linear dynamics.

Background & Motivation
---------------------
Dynamic system models are crucial for identifying and modeling systems where the output depends not only on the current input but also on the system's history. The Hammerstein-Wiener model is particularly important because it can:

1. **Capture Complex Dynamics**:
   - Model both input and output nonlinearities
   - Account for system memory effects
   - Handle time-varying relationships

2. **System Identification**:
   - Identify nonlinear dynamic systems
   - Separate static and dynamic components
   - Model various real-world phenomena

3. **Versatile Applications**:
   - Process control systems
   - Biological system modeling
   - Signal processing
   - Time series prediction

Theoretical Foundation
--------------------

The Hammerstein-Wiener model consists of three components in series:

1. **Input Nonlinearity (Hammerstein)**:
   
   .. math::

      w(t) = f(u(t))

   where f(·) is a static nonlinear function

2. **Linear Dynamics**:

   .. math::

      x(t) = \sum_{k=1}^n b_k w(t-k) - \sum_{k=1}^m a_k x(t-k)

   where {aₖ} and {bₖ} are the system coefficients

3. **Output Nonlinearity (Wiener)**:

   .. math::

      y(t) = h(x(t))

   where h(·) is another static nonlinear function

The complete model can be expressed as:

.. math::

   y(t) = h\left(\sum_{k=1}^n b_k f(u(t-k)) - \sum_{k=1}^m a_k x(t-k)\right)

Classes Overview
--------------

HammersteinWienerClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~
A classifier based on the Hammerstein-Wiener model structure.

Parameters:
    - memory_depth (int): Number of past samples to consider
    - hidden_neurons (list): Architecture of neural networks
    - activation (str): Activation function for neural networks
    - optimizer (str): Optimization algorithm
    - learning_rate (float): Learning rate for optimization
    - max_iter (int): Maximum training iterations
    - random_state (int): Random seed for reproducibility

Examples:

.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from gofast.estimators.dynamic_system import HammersteinWienerClassifier

    # Example 1: Time Series Classification
    # Generate synthetic dynamic data
    def generate_dynamic_data(n_samples, memory_depth):
        X = np.random.randn(n_samples, 5)
        # Add temporal dependencies
        y = np.zeros(n_samples)
        for i in range(memory_depth, n_samples):
            features = X[i-memory_depth:i+1].flatten()
            y[i] = 1 if np.sum(features) > 0 else 0
        return X, y

    X, y = generate_dynamic_data(1000, memory_depth=3)

    # Preprocess and split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create and train model
    hw_clf = HammersteinWienerClassifier(
        memory_depth=3,
        hidden_neurons=[10, 5],
        activation='relu',
        learning_rate=0.01,
        random_state=42
    )
    hw_clf.fit(X_train, y_train)

    # Evaluate
    train_score = hw_clf.score(X_train, y_train)
    test_score = hw_clf.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

HammersteinWienerRegressor
~~~~~~~~~~~~~~~~~~~~~~~~
A regression model based on the Hammerstein-Wiener architecture.

Parameters:
    - memory_depth (int): System memory depth
    - hidden_neurons (list): Neural network architecture
    - activation (str): Activation function
    - optimizer (str): Optimization algorithm
    - learning_rate (float): Learning rate
    - max_iter (int): Maximum iterations
    - random_state (int): Random seed

Examples:

.. code-block:: python

    from gofast.estimators.dynamic_system import HammersteinWienerRegressor

    # Example 2: Dynamic System Identification
    # Generate nonlinear dynamic system data
    def generate_nonlinear_system_data(n_samples):
        t = np.linspace(0, 10, n_samples)
        u = np.sin(t) + 0.5 * np.sin(2*t)
        
        # System with nonlinear dynamics
        x = np.zeros_like(u)
        y = np.zeros_like(u)
        
        for i in range(1, len(t)):
            x[i] = 0.8*x[i-1] + np.tanh(u[i])
            y[i] = x[i]**2 + 0.1*np.random.randn()
            
        return u.reshape(-1, 1), y

    X, y = generate_nonlinear_system_data(1000)

    # Split and preprocess
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train model
    hw_reg = HammersteinWienerRegressor(
        memory_depth=2,
        hidden_neurons=[20, 10],
        activation='tanh',
        learning_rate=0.01,
        random_state=42
    )
    hw_reg.fit(X_train, y_train)

    # Evaluate
    train_r2 = hw_reg.score(X_train, y_train)
    test_r2 = hw_reg.score(X_test, y_test)
    print(f"\nTraining R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")

EnsembleHWClassifier
~~~~~~~~~~~~~~~~~~
Ensemble of Hammerstein-Wiener classifiers for improved robustness.

Parameters:
    - n_estimators (int): Number of base estimators
    - memory_depth (int): Memory depth for each estimator
    - hidden_neurons (list): Neural network architecture
    - voting (str): Voting method ('hard', 'soft')

Examples:

.. code-block:: python

    from gofast.estimators.dynamic_system import EnsembleHWClassifier

    # Example 3: Ensemble Dynamic Classification
    # Create and train ensemble
    ensemble_clf = EnsembleHWClassifier(
        n_estimators=5,
        memory_depth=3,
        hidden_neurons=[10, 5],
        voting='soft',
        random_state=42
    )
    ensemble_clf.fit(X_train, y_train)

    # Compare with single model
    ensemble_score = ensemble_clf.score(X_test, y_test)
    print(f"\nEnsemble Test Accuracy: {ensemble_score:.4f}")

Implementation Details
--------------------

1. **Memory Management**:
   - Efficient storage of temporal sequences
   - Sliding window implementation
   - Memory depth optimization

2. **Model Architecture**:
   - Flexible neural network design
   - Multiple activation functions
   - Various optimization algorithms

3. **Training Process**:
   - Mini-batch processing
   - Early stopping criteria
   - Learning rate scheduling

Best Practices
-------------

1. **Data Preparation**:
   - Proper temporal alignment
   - Feature scaling
   - Handling missing values

2. **Model Selection**:
   - Memory depth tuning
   - Architecture optimization
   - Ensemble size selection

3. **Training Strategy**:
   - Cross-validation with time series
   - Monitoring convergence
   - Preventing overfitting

Advanced Concepts
---------------

1. **System Identification**:
   - Model structure selection
   - Parameter estimation
   - Validation techniques

2. **Nonlinearity Handling**:
   - Input nonlinearity estimation
   - Output nonlinearity approximation
   - Basis function selection

3. **Stability Analysis**:
   - BIBO stability checks
   - Parameter constraints
   - Robustness analysis

See Also
--------
- :mod:`gofast.estimators.ensemble`: Ensemble methods
- :mod:`gofast.timeseries`: Time series utilities
- :mod:`gofast.neural`: Neural network components

References
----------
.. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification: 
       A User-Oriented Road Map. IEEE Control Systems Magazine.

.. [2] Giri, F., & Bai, E. W. (2010). Block-oriented Nonlinear System 
       Identification. Springer.

.. [3] Billings, S. A. (2013). Nonlinear System Identification: NARMAX 
       Methods in the Time, Frequency, and Spatio-Temporal Domains. Wiley.