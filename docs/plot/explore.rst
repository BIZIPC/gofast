.. _explore:

Exploratory Visualization
=======================

.. currentmodule:: gofast.plot.explore

The :mod:`gofast.plot.explore` module provides comprehensive tools for exploratory data analysis through visualization. This module implements various plotting classes that help understand data distributions, relationships, and patterns.

Key Concepts
-----------
The module follows a "fit-then-plot" pattern where:

1. First initialize the plotter object with style parameters
2. Fit the plotter with data using the `fit` method
3. Use various plotting methods to visualize different aspects of the data

This design allows for:
- Consistent data handling across visualizations
- Memory efficiency for large datasets
- Flexible plot customization
- State preservation between plots

Classes Overview
--------------

EasyPlotter
~~~~~~~~~~
Base class providing fundamental plotting capabilities with an emphasis on ease of use.

Parameters:
    - fig_size (tuple): Figure dimensions (width, height)
    - style (str): Plot style ('seaborn', 'ggplot', etc.)
    - palette (str): Color palette name
    - title_size (int): Title font size
    - label_size (int): Label font size
    - tick_size (int): Tick label font size

Methods require fitting the data first using the `fit` method.

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from gofast.plot.explore import EasyPlotter

    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(2, 1.5, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })

    # Initialize and fit plotter
    plotter = EasyPlotter(
        fig_size=(10, 6),
        style='seaborn',
        palette='Set2'
    )
    plotter.fit(data)

    # Create various plots
    # 1. Histogram
    plotter.plotHistogram('feature1', bins=30)
    
    # 2. Box plot
    plotter.plotBox('feature1', by='category')
    
    # 3. Scatter plot
    plotter.plotScatter('feature1', 'feature2', hue='category')

QuestPlotter
~~~~~~~~~~~
Specialized class for exploratory analysis with advanced visualization capabilities.

Parameters:
    - fig_size (tuple): Figure dimensions
    - savefig (bool): Whether to save figures
    - features_to_drop (list): Features to exclude
    - random_state (int): Random seed
    - verbose (int): Verbosity level

Key Methods
----------

1. plotMissingPatterns
~~~~~~~~~~~~~~~~~~~~
Visualizes patterns of missing data in the dataset.

Parameters:
    - figsize (tuple): Figure size
    - rotx (int): X-axis label rotation
    - roty (int): Y-axis label rotation
    - fontsize (int): Font size
    - view (bool): Whether to display plot

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.plot.explore import QuestPlotter

    # Create data with missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, np.nan, 3, 4, 5],
        'D': [1, 2, 3, 4, np.nan]
    })

    # Initialize and fit plotter
    plotter = QuestPlotter(fig_size=(10, 6))
    plotter.fit(data)

    # Plot missing patterns
    plotter.plotMissingPatterns(
        figsize=(12, 6),
        rotx=45,
        fontsize=12,
        view=True
    )

2. plotHistCatDistribution
~~~~~~~~~~~~~~~~~~~~~~~~
Plots histogram distribution for categorical variables.

Parameters:
    - column (str): Column name to plot
    - bins (int): Number of bins
    - kde (bool): Whether to show kernel density estimate
    - norm_hist (bool): Whether to normalize histogram

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.plot.explore import QuestPlotter

    # Create categorical data
    data = pd.DataFrame({
        'category': np.random.choice(['Low', 'Medium', 'High'], 1000),
        'values': np.random.normal(0, 1, 1000),
        'group': np.random.choice(['A', 'B'], 1000)
    })

    # Initialize and fit plotter
    plotter = QuestPlotter(fig_size=(12, 6))
    plotter.fit(data)

    # Plot categorical distribution
    plotter.plotHistCatDistribution(
        'category',
        bins=20,
        kde=True,
        norm_hist=True
    )

3. plotBarCatDistribution
~~~~~~~~~~~~~~~~~~~~~~~
Creates bar plots for categorical variable distributions.

Parameters:
    - column (str): Column to plot
    - horizontal (bool): Whether to create horizontal bars
    - sort_bars (bool): Whether to sort bars by height
    - annotate (bool): Whether to add value annotations

Examples:

.. code-block:: python

    # Using previous data
    plotter.plotBarCatDistribution(
        'category',
        horizontal=True,
        sort_bars=True,
        annotate=True
    )

4. plotMultiCatDistribution
~~~~~~~~~~~~~~~~~~~~~~~~~
Visualizes multiple categorical distributions simultaneously.

Parameters:
    - columns (list): Columns to plot
    - figsize (tuple): Figure size
    - plot_type (str): Type of plot ('bar', 'count', 'box')

Examples:

.. code-block:: python

    # Create multi-categorical data
    data = pd.DataFrame({
        'category1': np.random.choice(['A', 'B', 'C'], 1000),
        'category2': np.random.choice(['X', 'Y', 'Z'], 1000),
        'category3': np.random.choice(['Low', 'High'], 1000),
        'values': np.random.normal(0, 1, 1000)
    })

    plotter = QuestPlotter(fig_size=(15, 8))
    plotter.fit(data)

    # Plot multiple distributions
    plotter.plotMultiCatDistribution(
        columns=['category1', 'category2', 'category3'],
        plot_type='bar'
    )
    
5. plotCorrMatrix
~~~~~~~~~~~~~~~
Visualizes correlation matrix with customizable features.

Parameters:
    - method (str): Correlation method ('pearson', 'spearman', 'kendall')
    - annot (bool): Whether to annotate cells
    - cmap (str): Color map for heatmap
    - mask_upper (bool): Whether to mask upper triangle

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast.plot.explore import QuestPlotter

    # Create correlated data
    n = 1000
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, n),
        'B': np.random.normal(0, 1, n),
        'C': np.random.normal(0, 1, n)
    })
    data['D'] = data['A'] * 0.5 + data['B'] * 0.3 + np.random.normal(0, 0.1, n)

    # Initialize and fit plotter
    plotter = QuestPlotter(fig_size=(10, 8))
    plotter.fit(data)

    # Plot correlation matrix
    plotter.plotCorrMatrix(
        method='pearson',
        annot=True,
        cmap='coolwarm',
        mask_upper=True
    )

6. plotNumFeatures
~~~~~~~~~~~~~~~~
Creates distribution plots for numerical features.

Parameters:
    - columns (list): Features to plot
    - n_cols (int): Number of columns in subplot grid
    - plot_type (str): Type of plot ('hist', 'kde', 'box')

Examples:

.. code-block:: python

    # Create numerical data
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(2, 1000),
        'feature3': np.random.gamma(2, 2, 1000),
        'feature4': np.random.lognormal(0, 1, 1000)
    })

    plotter = QuestPlotter(fig_size=(15, 10))
    plotter.fit(data)

    # Plot numerical distributions
    plotter.plotNumFeatures(
        columns=data.columns,
        n_cols=2,
        plot_type='hist'
    )

7. plotJoint2Features
~~~~~~~~~~~~~~~~~~~
Creates joint plot for two features showing marginal distributions.

Parameters:
    - x (str): First feature name
    - y (str): Second feature name
    - kind (str): Plot type ('scatter', 'hex', 'kde')
    - hue (str): Variable for color coding

Examples:

.. code-block:: python

    # Create bivariate data
    n = 1000
    data = pd.DataFrame({
        'x': np.random.normal(0, 1, n),
        'y': np.random.normal(0, 1, n),
        'group': np.random.choice(['A', 'B'], n)
    })
    data['y'] += 0.5 * data['x']  # Add correlation

    plotter = QuestPlotter(fig_size=(10, 10))
    plotter.fit(data)

    # Create joint plot
    plotter.plotJoint2Features(
        x='x',
        y='y',
        kind='scatter',
        hue='group'
    )

Advanced Usage
---------------

1. Custom Style Configuration:

.. code-block:: python

    # Create plotter with custom style
    plotter = QuestPlotter(
        fig_size=(12, 8),
        style={
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        }
    )

2. Complex Multi-Feature Analysis:

.. code-block:: python

    # Create complex dataset
    data = pd.DataFrame({
        'numeric1': np.random.normal(0, 1, 1000),
        'numeric2': np.random.exponential(2, 1000),
        'category1': np.random.choice(['A', 'B', 'C'], 1000),
        'category2': np.random.choice(['X', 'Y'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })

    plotter = QuestPlotter(fig_size=(15, 10))
    plotter.fit(data)

    # Multiple analysis in sequence
    plotter.plotMissingPatterns()
    plotter.plotMultiCatDistribution(
        columns=['category1', 'category2']
    )
    plotter.plotNumFeatures(
        columns=['numeric1', 'numeric2']
    )
    plotter.plotCorrMatrix()

3. Interactive Analysis Pipeline:

.. code-block:: python

    class DataAnalyzer:
        def __init__(self, data):
            self.data = data
            self.plotter = QuestPlotter(fig_size=(12, 8))
            self.plotter.fit(data)
        
        def analyze_distributions(self):
            # Analyze numerical distributions
            num_cols = self.data.select_dtypes(
                include=[np.number]
            ).columns
            self.plotter.plotNumFeatures(columns=num_cols)
            
            # Analyze categorical distributions
            cat_cols = self.data.select_dtypes(
                include=['category', 'object']
            ).columns
            self.plotter.plotMultiCatDistribution(
                columns=cat_cols
            )
        
        def analyze_correlations(self):
            self.plotter.plotCorrMatrix(
                annot=True,
                mask_upper=True
            )
        
        def analyze_missing_data(self):
            self.plotter.plotMissingPatterns()

    # Usage
    analyzer = DataAnalyzer(data)
    analyzer.analyze_distributions()
    analyzer.analyze_correlations()
    analyzer.analyze_missing_data()


4. Time Series with Categorical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from gofast.plot.explore import QuestPlotter

    # Create complex time series data with categories
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(0, 1, 1000) + np.sin(np.arange(1000) * 2 * np.pi / 365),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'season': pd.cut(pd.DatetimeIndex(dates).month, 
                        bins=[0, 3, 6, 9, 12],
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])
    })

    # Add some missing values
    data.loc[np.random.choice(data.index, 50), 'value'] = np.nan

    # Initialize plotter with custom settings
    plotter = QuestPlotter(
        fig_size=(15, 10),
        style='seaborn-whitegrid',
        features_to_drop=['date']
    )
    plotter.fit(data)

    # 1. Analyze temporal patterns with categories
    plt.figure(figsize=(15, 10))
    for cat in data['category'].unique():
        subset = data[data['category'] == cat]
        plt.plot(subset['date'], subset['value'], 
                label=f'Category {cat}', alpha=0.6)
    plt.title('Time Series by Category')
    plt.legend()
    plt.show()

    # 2. Seasonal distribution analysis
    plotter.plotMultiCatDistribution(
        columns=['season', 'category'],
        figsize=(12, 6),
        plot_type='count'
    )

    # 3. Value distribution by season and category
    g = sns.FacetGrid(data, col='season', row='category', height=3)
    g.map(sns.histplot, 'value', kde=True)
    plt.show()

5. Complex Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create dataset with complex relationships
    n_samples = 1000
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n_samples),
        'x2': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })

    # Add derived features with different relationships
    data['y1'] = 2 * data['x1'] + np.random.normal(0, 0.5, n_samples)  # Linear
    data['y2'] = data['x1']**2 + np.random.normal(0, 0.5, n_samples)   # Quadratic
    data['y3'] = np.sin(data['x1']) + np.random.normal(0, 0.2, n_samples)  # Sinusoidal
    
    # Add categorical interaction
    data.loc[data['category'] == 'A', 'y4'] = data.loc[data['category'] == 'A', 'x1'] * 2
    data.loc[data['category'] == 'B', 'y4'] = data.loc[data['category'] == 'B', 'x1'] * 0.5
    data.loc[data['category'] == 'C', 'y4'] = data.loc[data['category'] == 'C', 'x1'] * -1

    # Initialize plotter
    plotter = QuestPlotter(fig_size=(15, 12))
    plotter.fit(data)

    # 1. Basic correlation matrix
    plotter.plotCorrMatrix(
        method='spearman',  # Use Spearman for non-linear relationships
        annot=True,
        cmap='RdBu',
        mask_upper=True
    )

    # 2. Category-wise correlation analysis
    for cat in data['category'].unique():
        subset = data[data['category'] == cat].drop('category', axis=1)
        plt.figure(figsize=(8, 6))
        sns.heatmap(subset.corr(), annot=True, cmap='RdBu')
        plt.title(f'Correlation Matrix for Category {cat}')
        plt.show()

    # 3. Relationship visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    relationships = ['y1', 'y2', 'y3', 'y4']
    
    for ax, y_var in zip(axes.flat, relationships):
        sns.scatterplot(data=data, x='x1', y=y_var, 
                       hue='category', alpha=0.6, ax=ax)
        ax.set_title(f'x1 vs {y_var}')
    plt.tight_layout()
    plt.show()

6. Multi-Feature Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create complex dataset with mixed distributions
    n_samples = 1000
    data = pd.DataFrame({
        'normal': np.random.normal(0, 1, n_samples),
        'exponential': np.random.exponential(2, n_samples),
        'lognormal': np.random.lognormal(0, 0.5, n_samples),
        'bimodal': np.concatenate([
            np.random.normal(-2, 0.5, n_samples//2),
            np.random.normal(2, 0.5, n_samples//2)
        ]),
        'uniform': np.random.uniform(-3, 3, n_samples),
        'category': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    })

    # Add some outliers
    data.loc[np.random.choice(data.index, 20), 'normal'] = np.random.normal(10, 2, 20)

    # Initialize plotter
    plotter = QuestPlotter(fig_size=(15, 10))
    plotter.fit(data)

    # 1. Distribution comparison
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    plotter.plotNumFeatures(
        columns=numeric_cols,
        n_cols=3,
        plot_type='hist'
    )

    # 2. Box plots with category
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for ax, col in zip(axes.flat, numeric_cols[:-1]):
        sns.boxplot(data=data, x='category', y=col, ax=ax)
        ax.set_title(f'{col} by Category')
    plt.tight_layout()
    plt.show()

    # 3. Kernel Density Estimation
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for ax, col in zip(axes.flat, numeric_cols[:-1]):
        sns.kdeplot(data=data, x=col, hue='category', ax=ax)
        ax.set_title(f'KDE of {col} by Category')
    plt.tight_layout()
    plt.show()

7. Advanced Missing Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create dataset with structured missing patterns
    n_samples = 1000
    base_data = pd.DataFrame({
        'id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })

    # Create missing patterns
    # Pattern 1: Missing completely at random
    base_data.loc[np.random.choice(n_samples, 50), 'feature1'] = np.nan

    # Pattern 2: Missing based on category
    base_data.loc[(base_data['category'] == 'A') & 
                 (np.random.random(n_samples) < 0.3), 'feature2'] = np.nan

    # Pattern 3: Missing based on value
    base_data.loc[base_data['feature3'] > 1, 'feature3'] = np.nan

    # Initialize plotter
    plotter = QuestPlotter(fig_size=(12, 8))
    plotter.fit(base_data)

    # 1. Basic missing pattern analysis
    plotter.plotMissingPatterns(
        figsize=(10, 6),
        rotx=45,
        fontsize=12
    )

    # 2. Missing data correlation
    missing_matrix = base_data.isna().astype(int)
    plt.figure(figsize=(10, 8))
    sns.heatmap(missing_matrix.corr(), annot=True, cmap='coolwarm')
    plt.title('Missing Data Correlation Matrix')
    plt.show()

    # 3. Category-wise missing data analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, feature in enumerate(['feature1', 'feature2', 'feature3']):
        missing_by_cat = base_data.groupby('category')[feature].apply(
            lambda x: x.isna().mean()
        )
        missing_by_cat.plot(kind='bar', ax=axes[idx])
        axes[idx].set_title(f'Missing Rate in {feature} by Category')
        axes[idx].set_ylabel('Missing Rate')
    plt.tight_layout()
    plt.show()

8. Advanced Categorical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from gofast.plot.explore import QuestPlotter

    # Create complex categorical dataset
    n_samples = 1000
    data = pd.DataFrame({
        'primary_category': np.random.choice(['A', 'B', 'C'], n_samples),
        'secondary_category': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'ordinal_feature': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'numeric_value': np.random.normal(0, 1, n_samples),
        'count_value': np.random.poisson(5, n_samples)
    })

    # Add interaction effects
    for cat in ['A', 'B', 'C']:
        mask = data['primary_category'] == cat
        data.loc[mask, 'numeric_value'] += {'A': -1, 'B': 0, 'C': 1}[cat]

    # Initialize plotter
    plotter = QuestPlotter(fig_size=(15, 10))
    plotter.fit(data)

    # 1. Complex categorical distribution
    plotter.plotMultiCatDistribution(
        columns=['primary_category', 'secondary_category', 'ordinal_feature'],
        figsize=(15, 5),
        plot_type='count'
    )

    # 2. Categorical interaction analysis
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data,
        x='primary_category',
        y='numeric_value',
        hue='secondary_category'
    )
    plt.title('Value Distribution by Category Interaction')
    plt.show()

    # 3. Heatmap of category combinations
    pivot_table = pd.crosstab(
        data['primary_category'],
        data['secondary_category'],
        values=data['count_value'],
        aggfunc='mean'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
    plt.title('Average Count by Category Combinations')
    plt.show()

9. Interactive Analysis Class
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class InteractiveExplorer:
        def __init__(self, data):
            self.data = data
            self.plotter = QuestPlotter(fig_size=(12, 8))
            self.plotter.fit(data)
            self.numeric_cols = data.select_dtypes(include=[np.number]).columns
            self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns

        def analyze_distributions(self, plot_type='hist'):
            """Analyze all numeric distributions."""
            self.plotter.plotNumFeatures(
                columns=self.numeric_cols,
                plot_type=plot_type
            )

        def analyze_categories(self):
            """Analyze categorical distributions and relationships."""
            self.plotter.plotMultiCatDistribution(
                columns=self.categorical_cols
            )

        def analyze_relationships(self, target_col):
            """Analyze relationships with target variable."""
            # Numeric relationships
            fig, axes = plt.subplots(
                len(self.numeric_cols)-1, 1,
                figsize=(12, 4*(len(self.numeric_cols)-1))
            )
            for ax, col in zip(axes, self.numeric_cols):
                if col != target_col:
                    sns.scatterplot(
                        data=self.data,
                        x=col,
                        y=target_col,
                        ax=ax
                    )
            plt.tight_layout()
            plt.show()

            # Categorical relationships
            fig, axes = plt.subplots(
                len(self.categorical_cols), 1,
                figsize=(12, 4*len(self.categorical_cols))
            )
            for ax, col in zip(axes, self.categorical_cols):
                sns.boxplot(
                    data=self.data,
                    x=col,
                    y=target_col,
                    ax=ax
                )
            plt.tight_layout()
            plt.show()

        def analyze_correlations(self):
            """Analyze correlation patterns."""
            self.plotter.plotCorrMatrix(
                method='spearman',
                annot=True
            )

    # Usage example
    explorer = InteractiveExplorer(data)
    explorer.analyze_distributions()
    explorer.analyze_categories()
    explorer.analyze_relationships('numeric_value')
    explorer.analyze_correlations()


Best Practices
---------------

1. **Data Preparation**:
   - Clean data before visualization
   - Handle missing values appropriately
   - Convert data types as needed

2. **Plot Configuration**:
   - Choose appropriate plot types for data
   - Use consistent styling
   - Consider color blindness accessibility

3. **Performance Optimization**:
   - Use appropriate figure sizes
   - Consider memory usage for large datasets
   - Save figures when needed
   


Performance Optimization:

.. code-block:: python

    class OptimizedPlotter(QuestPlotter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cached_data = {}

        def fit(self, data):
            # Cache commonly used computations
            self.cached_data['numeric_cols'] = data.select_dtypes(
                include=[np.number]
            ).columns
            self.cached_data['categorical_cols'] = data.select_dtypes(
                include=['object', 'category']
            ).columns
            self.cached_data['correlations'] = data.corr()
            super().fit(data)


Customization Example:

.. code-block:: python

    def custom_visualization_pipeline(data, save_path=None):
        """
        Custom visualization pipeline with advanced features.
        """
        # Initialize plotter with custom settings
        plotter = QuestPlotter(
            fig_size=(15, 10),
            style='seaborn-whitegrid',
            palette='Set3'
        )
        plotter.fit(data)

        # 1. Distribution Analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        fig = plt.figure(figsize=(15, 5*len(numeric_cols)))
        gs = plt.GridSpec(len(numeric_cols), 2)

        for i, col in enumerate(numeric_cols):
            # Histogram
            ax1 = fig.add_subplot(gs[i, 0])
            sns.histplot(data[col], kde=True, ax=ax1)
            ax1.set_title(f'{col} Distribution')

            # Box plot
            ax2 = fig.add_subplot(gs[i, 1])
            sns.boxplot(y=data[col], ax=ax2)
            ax2.set_title(f'{col} Box Plot')

        plt.tight_layout()
        if save_path:
            plt.savefig(f'{save_path}/distributions.png')
        plt.show()

        # 2. Correlation Analysis
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(data.corr(), dtype=bool))
        sns.heatmap(
            data.corr(),
            mask=mask,
            annot=True,
            cmap='RdBu',
            center=0
        )
        plt.title('Correlation Matrix')
        if save_path:
            plt.savefig(f'{save_path}/correlations.png')
        plt.show()

        # 3. Category Analysis (if applicable)
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            fig = plt.figure(figsize=(15, 5*len(cat_cols)))
            gs = plt.GridSpec(len(cat_cols), 2)

            for i, col in enumerate(cat_cols):
                # Bar plot
                ax1 = fig.add_subplot(gs[i, 0])
                sns.countplot(data=data, x=col, ax=ax1)
                ax1.set_title(f'{col} Distribution')
                plt.xticks(rotation=45)

                # Pie chart
                ax2 = fig.add_subplot(gs[i, 1])
                data[col].value_counts().plot(
                    kind='pie',
                    autopct='%1.1f%%',
                    ax=ax2
                )
                ax2.set_title(f'{col} Proportion')

            plt.tight_layout()
            if save_path:
                plt.savefig(f'{save_path}/categories.png')
            plt.show()

    # Usage
    custom_visualization_pipeline(
        data,
        save_path='./visualizations'
    )

See Also
--------
- :mod:`gofast.plot.eval`: Evaluation plots
- :mod:`gofast.plot.ts`: Time series plots
- :mod:`gofast.plot.utils`: Plotting utilities

References
----------
.. [1] Wickham, H. (2016). ggplot2: Elegant Graphics for Data Analysis. 
       Springer-Verlag New York.

.. [2] McKinney, W. (2012). Python for Data Analysis: Data Wrangling with 
       Pandas, NumPy, and IPython. O'Reilly Media.