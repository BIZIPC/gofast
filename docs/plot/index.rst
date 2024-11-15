.. _plot:

***********
Plot Module
***********

.. currentmodule:: gofast.plot

The :mod:`gofast.plot` module provides a comprehensive suite of visualization tools for data analysis, model evaluation, and exploratory data analysis. This module combines various plotting capabilities to create insightful and publication-ready visualizations.

Module Structure
--------------

The plot module is organized into several submodules, each focusing on specific visualization needs:

- :mod:`~gofast.plot.explore`: Exploratory visualization tools
- :mod:`~gofast.plot.eval`: Model evaluation plots
- :mod:`~gofast.plot.ts`: Time series visualization
- :mod:`~gofast.plot.utils`: Plotting utilities and helpers

Key Classes
----------

.. currentmodule:: gofast

EasyPlotter
~~~~~~~~~~
Base plotting class providing common visualization functionality.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.EasyPlotter

EvalPlotter
~~~~~~~~~~
Specialized class for model evaluation visualizations.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.EvalPlotter
   plot.EvalPlotter.plotPCA
   plot.EvalPlotter.plotPR
   plot.EvalPlotter.plotROC
   plot.EvalPlotter.plotRobustPCA
   plot.EvalPlotter.plotConfusionMatrix
   plot.EvalPlotter.plotFeatureImportance
   plot.EvalPlotter.plotHeatmap
   plot.EvalPlotter.plotBox
   plot.EvalPlotter.plotHistogram
   plot.EvalPlotter.plot2d

MetricPlotter
~~~~~~~~~~~~
Class for visualizing various performance metrics.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.MetricPlotter
   plot.MetricPlotter.plotConfusionMatrix
   plot.MetricPlotter.plotRocCurve
   plot.MetricPlotter.plotPrecisionRecallCurve
   plot.MetricPlotter.plotLearningCurve
   plot.MetricPlotter.plotSilhouette
   plot.MetricPlotter.plotLiftCurve
   plot.MetricPlotter.plotCumulativeGain
   plot.MetricPlotter.plotPrecisionRecallPerClass
   plot.MetricPlotter.plotActualVSPredicted

QuestPlotter
~~~~~~~~~~~
Class for questionnaire and survey data visualization.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.QuestPlotter

TimeSeriesPlotter
~~~~~~~~~~~~~~~
Specialized class for time series visualization.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.TimeSeriesPlotter

Plot Types
---------

Categorical Plots
~~~~~~~~~~~~~~~
Tools for visualizing categorical data relationships:

- stripplot: Plot a categorical scatter with jitter
- swarmplot: Plot a categorical scatter with non-overlapping points
- violinplot: Draw an enhanced boxplot using kernel density estimation
- pointplot: Plot point estimates and CIs using markers and lines
- boxplot: Draw an enhanced boxplot

Multiple Variable Plots
~~~~~~~~~~~~~~~~~~~~
Tools for exploring relationships between multiple variables:

- jointplot: Draw a bivariate plot with univariate marginal distributions
- pairplot: Draw multiple bivariate plots with univariate marginal distributions
- jointgrid: Set up a figure with joint and marginal views on bivariate data
- pairgrid: Set up a figure with joint and marginal views on multiple variables

Usage Examples
------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.plot import EasyPlotter

    # Create sample data
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, 100),
        'y': np.random.normal(0, 1, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Initialize plotter
    plotter = EasyPlotter()

    # Create basic visualization
    plotter.scatter(data['x'], data['y'], c=data['category'])

Model Evaluation
~~~~~~~~~~~~~~

.. code-block:: python

    from gofast.plot import EvalPlotter
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification

    # Generate classification data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Initialize evaluation plotter
    eval_plotter = EvalPlotter()

    # Plot ROC curve
    eval_plotter.plotROC(y_test, y_pred_proba)

Time Series Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from gofast.plot import TimeSeriesPlotter
    import pandas as pd

    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100)
    values = np.random.cumsum(np.random.randn(100))
    ts_data = pd.Series(values, index=dates)

    # Initialize time series plotter
    ts_plotter = TimeSeriesPlotter()

    # Create time series visualization
    ts_plotter.plot(ts_data)

Advanced Features
---------------

Customization Options
~~~~~~~~~~~~~~~~~~~
All plotting classes support extensive customization:

- Color schemes and palettes
- Figure size and layout
- Axis properties and labels
- Legend positioning
- Grid styles
- Font properties

Style Management
~~~~~~~~~~~~~~
The module provides tools for consistent styling:

.. code-block:: python

    from gofast.plot.utils import set_style, reset_style

    # Set custom style
    set_style('publication')

    # Create plots...

    # Reset to default style
    reset_style()

Best Practices
------------

1. **Data Preparation**:
   - Clean and preprocess data before plotting
   - Handle missing values appropriately
   - Scale data when necessary

2. **Style Consistency**:
   - Use consistent color schemes
   - Maintain uniform figure sizes
   - Apply consistent formatting

3. **Performance Optimization**:
   - Use appropriate plot types for data size
   - Consider memory usage for large datasets
   - Utilize downsampling when needed

See Also
--------
- :mod:`gofast.visualization`: Additional visualization tools
- :mod:`gofast.metrics`: Performance metrics calculation
- :mod:`gofast.preprocessing`: Data preprocessing utilities

References
----------
.. [1] Wilke, C. O. (2019). Fundamentals of Data Visualization. O'Reilly Media.

.. [2] Tufte, E. R. (2001). The Visual Display of Quantitative Information. 
       Graphics Press.

.. toctree::
   :maxdepth: 2
   :titlesonly:

   charts
   clusters
   dimensionality
   evaluate
   explore
   feature_analysis
   inspection
   mlviz
   ts
   utils