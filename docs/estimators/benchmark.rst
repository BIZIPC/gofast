.. _benchmark:

Benchmark Models
===============

.. currentmodule:: gofast.estimators.benchmark

The :mod:`gofast.estimators.benchmark` module provides benchmark models for evaluating machine learning performance. These models serve as baseline references for comparing more complex algorithms, offering standardized performance metrics and helping to establish minimum acceptable performance levels.

Background & Importance
---------------------
Benchmark models are crucial in machine learning for several reasons:

1. **Performance Baseline**:
   - Establish minimum performance thresholds
   - Provide reference points for model comparison
   - Help identify when complex models are unnecessary

2. **Model Validation**:
   - Verify if sophisticated models offer significant improvements
   - Detect overfitting or underfitting
   - Validate the need for complex architectures

3. **Cost-Benefit Analysis**:
   - Evaluate trade-offs between model complexity and performance
   - Assess computational resource requirements
   - Guide model selection decisions

Key Features
-----------
- **Standardized Metrics**: Consistent performance evaluation across different models
- **Multiple Algorithms**: Various baseline algorithms for different tasks
- **Scalability**: Efficient implementation for large datasets
- **Compatibility**: Scikit-learn compatible API
- **Visualization**: Built-in plotting capabilities for model comparison

Classes Overview
--------------

BenchmarkClassifier
~~~~~~~~~~~~~~~~~
A benchmark classifier that implements multiple baseline classification strategies.

Parameters:
    - strategy (str): Benchmark strategy to use
        - 'most_frequent': Always predicts most frequent class
        - 'stratified': Generates predictions based on class distribution
        - 'uniform': Generates predictions uniformly at random
        - 'constant': Always predicts a specified constant class
    - random_state (int): Random number generator seed
    - constant (int): Value to predict when strategy='constant'

Attributes:
    - classes_ (array): Unique classes in training data
    - n_classes_ (int): Number of classes
    - class_prior_ (array): Class probability estimates

Mathematical Foundation:

For 'most_frequent' strategy:
.. math::

    \hat{y} = \arg\max_{c \in C} \sum_{i=1}^n \mathbb{1}(y_i = c)

For 'stratified' strategy:
.. math::

    P(\hat{y} = c) = \frac{\sum_{i=1}^n \mathbb{1}(y_i = c)}{n}

Examples:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from gofast.estimators.benchmark import BenchmarkClassifier

    # Example 1: Most Frequent Strategy
    # Generate imbalanced classification data
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        weights=[0.6, 0.3, 0.1],
        n_informative=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and evaluate most_frequent benchmark
    clf_mf = BenchmarkClassifier(strategy='most_frequent')
    clf_mf.fit(X_train, y_train)
    score_mf = clf_mf.score(X_test, y_test)
    print(f"Most Frequent Strategy Accuracy: {score_mf:.4f}")

    # Example 2: Stratified Strategy with Visualization
    clf_strat = BenchmarkClassifier(
        strategy='stratified',
        random_state=42
    )
    clf_strat.fit(X_train, y_train)
    
    # Compare predictions distribution with true distribution
    pred_dist = np.bincount(clf_strat.predict(X_test)) / len(y_test)
    true_dist = np.bincount(y_test) / len(y_test)
    
    print("\nClass Distribution Comparison:")
    for i, (pred, true) in enumerate(zip(pred_dist, true_dist)):
        print(f"Class {i}: Predicted={pred:.3f}, True={true:.3f}")

BenchmarkRegressor
~~~~~~~~~~~~~~~~
A benchmark regressor implementing various baseline regression strategies.

Parameters:
    - strategy (str): Benchmark strategy to use
        - 'mean': Always predicts mean of training target
        - 'median': Always predicts median of training target
        - 'quantile': Predicts specified quantile of training target
        - 'constant': Always predicts specified constant value
    - quantile (float): Quantile to predict when strategy='quantile'
    - constant (float): Value to predict when strategy='constant'

Attributes:
    - constant_ (float): Stored prediction value
    - n_samples_ (int): Number of training samples
    - n_features_ (int): Number of features

Mathematical Foundation:

For 'mean' strategy:
.. math::

    \hat{y} = \frac{1}{n}\sum_{i=1}^n y_i

For 'quantile' strategy:
.. math::

    \hat{y} = Q_{\alpha}(y) = \inf\{x : P(Y \leq x) \geq \alpha\}

Examples:

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.model_selection import cross_val_score
    from gofast.estimators.benchmark import BenchmarkRegressor

    # Example 3: Comparing Different Regression Strategies
    # Generate regression data with different distributions
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        noise=0.5,
        random_state=42
    )
    
    # Add some outliers to make median potentially better
    y[::100] *= 5

    # Compare different strategies
    strategies = ['mean', 'median', 'quantile']
    results = {}

    for strategy in strategies:
        reg = BenchmarkRegressor(
            strategy=strategy,
            quantile=0.5 if strategy == 'quantile' else None
        )
        
        # Perform cross-validation
        scores = cross_val_score(
            reg, X, y,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        
        results[strategy] = -scores.mean()

    print("\nMean Squared Error by Strategy:")
    for strategy, mse in results.items():
        print(f"{strategy}: {mse:.4f}")

    # Example 4: Quantile Regression for Prediction Intervals
    # Generate data with heteroscedastic noise
    n_samples = 1000
    X = np.random.randn(n_samples, 1)
    y = X.ravel() + np.random.randn(n_samples) * np.abs(X.ravel())

    # Fit multiple quantile regressors
    quantiles = [0.1, 0.5, 0.9]
    quantile_models = {}

    for q in quantiles:
        reg = BenchmarkRegressor(strategy='quantile', quantile=q)
        reg.fit(X, y)
        quantile_models[q] = reg

    # Make predictions
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    predictions = {
        q: model.predict(X_test)
        for q, model in quantile_models.items()
    }

    print("\nPrediction Intervals Example:")
    print(f"Lower Bound (10%): {predictions[0.1][50]:.4f}")
    print(f"Median (50%): {predictions[0.5][50]:.4f}")
    print(f"Upper Bound (90%): {predictions[0.9][50]:.4f}")

Implementation Notes
------------------

1. **Strategy Selection**:
   - Choose strategy based on data distribution
   - Consider computational resources
   - Account for presence of outliers

2. **Performance Evaluation**:
   - Use appropriate metrics for task
   - Consider multiple evaluation criteria
   - Compare with domain-specific benchmarks

3. **Common Use Cases**:
   - Model comparison baseline
   - Quick prototyping
   - Sanity checks

4. **Limitations**:
   - No feature learning capability
   - May underperform on complex patterns
   - Limited to basic statistical properties

Best Practices
-------------
1. **Baseline Establishment**:
   - Always start with simple benchmarks
   - Document benchmark performance
   - Use multiple benchmark strategies

2. **Model Comparison**:
   - Use consistent evaluation metrics
   - Consider statistical significance
   - Account for computational costs

3. **Reporting**:
   - Include benchmark results in analysis
   - Document strategy selection rationale
   - Report confidence intervals

See Also
--------
- :mod:`gofast.metrics`: Evaluation metrics
- :mod:`gofast.visualize`: Visualization tools
- :mod:`gofast.stats`: Statistical utilities

References
----------
.. [1] Flach, P. (2012). Machine Learning: The Art and Science of Algorithms That 
       Make Sense of Data. Cambridge University Press.

.. [2] Koenker, R. (2005). Quantile Regression. Cambridge University Press.

.. [3] Henderson, P., et al. (2018). Deep Reinforcement Learning That Matters. 
       AAAI Conference on Artificial Intelligence.