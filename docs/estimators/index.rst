.. _estimators:

**********
Estimators
**********

.. currentmodule:: gofast.estimators

The :mod:`gofast.estimators` module provides a comprehensive collection of machine learning estimators and models. This module implements various classical and advanced algorithms for classification, regression, and clustering tasks, following a scikit-learn compatible API design.

Submodules
----------

- :mod:`~gofast.estimators.adaline`: Adaptive Linear Neuron implementations
    - AdalineClassifier
    - AdalineMixte
    - AdalineRegressor
    - AdalineStochasticRegressor
    - AdalineStochasticClassifier

- :mod:`~gofast.estimators.base`: Base classes and fundamental estimators
    - DecisionStumpRegressor

- :mod:`~gofast.estimators.benchmark`: Benchmark models for comparison
    - BenchmarkRegressor
    - BenchmarkClassifier

- :mod:`~gofast.estimators.boosting`: Boosting algorithm implementations
    - HybridBoostingClassifier
    - HybridBoostingRegressor

- :mod:`~gofast.estimators.cluster_based`: Clustering-based models
    - KMFClassifier
    - KMFRegressor

- :mod:`~gofast.estimators.dynamic_system`: Dynamic system models
    - HammersteinWienerClassifier
    - HammersteinWienerRegressor
    - EnsembleHWClassifier
    - EnsembleHWRegressor

- :mod:`~gofast.estimators.ensemble`: Ensemble learning methods
    - MajorityVoteClassifier
    - SimpleAverageClassifier
    - WeightedAverageClassifier
    - SimpleAverageRegressor
    - WeightedAverageRegressor
    - EnsembleClassifier
    - EnsembleRegressor

- :mod:`~gofast.estimators.perceptron`: Neural network components
    - Perceptron

Key Features
------------

- **Flexible API Design**: 
    All estimators follow a consistent interface with standard methods:
    
    - fit(X, y): Train the model
    - predict(X): Make predictions
    - score(X, y): Evaluate model performance
    - get_params(): Get model parameters
    - set_params(): Set model parameters

- **Comprehensive Algorithms**:
    Wide range of algorithms for different machine learning tasks:
    
    - Classification
    - Regression
    - Clustering
    - Ensemble Methods
    - Dynamic Systems

- **Performance Optimization**:
    Implementations focused on computational efficiency:
    
    - Vectorized operations
    - Parallel processing support
    - Memory-efficient designs

- **Extensibility**:
    Easy extension and customization through:
    
    - Base classes for new estimators
    - Mixins for common functionality
    - Flexible parameter handling

Common Usage Examples
-------------------

1. Basic Classification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gofast.estimators.adaline import AdalineClassifier
    from gofast.datasets import load_example_data

    # Load data
    X, y = load_example_data()
    
    # Create and train classifier
    clf = AdalineClassifier(learning_rate=0.01, n_iter=100)
    clf.fit(X, y)
    
    # Make predictions
    predictions = clf.predict(X)

2. Ensemble Learning
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gofast.estimators.ensemble import WeightedAverageClassifier
    from gofast.estimators.adaline import AdalineClassifier
    from gofast.estimators.perceptron import Perceptron

    # Create base models
    estimators = [
        ('adaline', AdalineClassifier()),
        ('perceptron', Perceptron())
    ]
    
    # Create ensemble
    ensemble = WeightedAverageClassifier(estimators=estimators)
    ensemble.fit(X, y)

3. Dynamic System Modeling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gofast.estimators.dynamic_system import HammersteinWienerRegressor

    # Create and train model
    model = HammersteinWienerRegressor()
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)

Base Classes
-----------

The module provides several base classes that define the interface for estimators:

- **BaseEstimator**: Root class providing parameter handling
- **ClassifierMixin**: Interface for classification tasks
- **RegressorMixin**: Interface for regression tasks
- **ClusterMixin**: Interface for clustering tasks

Guidelines for Custom Estimators
------------------------------

When creating custom estimators:

1. Inherit from appropriate base classes
2. Implement required methods (fit, predict, etc.)
3. Follow scikit-learn API conventions
4. Include parameter validation
5. Add proper documentation

Example:

.. code-block:: python

    from gofast.estimators.base import BaseEstimator, ClassifierMixin

    class CustomClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, param1=1, param2='value'):
            self.param1 = param1
            self.param2 = param2
            
        def fit(self, X, y):
            # Implementation
            return self
            
        def predict(self, X):
            # Implementation
            return predictions

.. toctree::
   :maxdepth: 2
   :titlesonly:

   adaline
   base
   benchmark
   boosting
   cluster_based
   dynamic_system
   ensemble
   perceptron

See Also
--------
- :mod:`gofast.metrics`: Evaluation metrics
- :mod:`gofast.preprocessing`: Data preprocessing tools
- :mod:`gofast.model_selection`: Model selection utilities