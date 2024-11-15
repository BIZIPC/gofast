.. _charts:

Charts Module
============

.. currentmodule:: gofast.plot.charts

The charts module provides specialized functions for creating pie charts and radar charts.

Functions
--------

plot_pie_charts
~~~~~~~~~~~~~
Creates pie charts for both categorical and numerical data in a DataFrame.

Mathematical Foundation:
For numerical data binning:

.. math::

    bins = \frac{max(x) - min(x)}{n}

where:
- x is the numerical data
- n is the number of bins

Parameters:
    - data (DataFrame): Input DataFrame containing the data
    - columns (str or List[str], optional): Specific columns to plot
    - bin_numerical (bool): Whether to bin numerical data
    - num_bins (int): Number of bins for numerical data
    - handle_missing (str): How to handle missing values ('exclude' or 'include')
    - explode (Tuple[float, ...] or str): Separation of wedges from center
    - shadow (bool): Whether to add shadow effect
    - startangle (int): Starting angle for first slice
    - cmap (str): Colormap name
    - autopct (str): Format string for wedge labels
    - verbose (int): Verbosity level

Examples:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from gofast.plot.charts import plot_pie_charts

    # Example 1: Basic categorical data
    df = pd.DataFrame({
        'Category': ['A', 'B', 'A', 'C', 'B', 'A'],
        'Values': [1, 2, 3, 4, 5, 6]
    })
    plot_pie_charts(df, columns=['Category'])

    # Example 2: Numerical data with binning
    df = pd.DataFrame({
        'Values': np.random.normal(0, 1, 100),
        'Scores': np.random.uniform(0, 100, 100)
    })
    plot_pie_charts(df, bin_numerical=True, num_bins=5)

    # Example 3: Mixed data with custom settings
    df = pd.DataFrame({
        'Category': ['A', 'B', 'C'] * 30,
        'Values': np.random.normal(0, 1, 90),
        'Ratings': np.random.choice(['High', 'Medium', 'Low'], 90)
    })
    plot_pie_charts(
        df,
        columns=['Category', 'Values', 'Ratings'],
        bin_numerical=True,
        num_bins=4,
        explode='auto',
        cmap='Set3'
    )

create_radar_chart
~~~~~~~~~~~~~~~
Creates a customizable radar chart for multivariate data visualization.

Mathematical Foundation:
The angles for each axis are computed using:

.. math::

    \theta_i = \frac{2\pi i}{n}

where:
- i is the category index
- n is the total number of categories

Parameters:
    - d (ArrayLike): 2D array of shape (n_clusters, n_variables)
    - categories (List[str]): Names of variables
    - cluster_labels (List[str]): Labels for clusters
    - title (str): Chart title
    - figsize (Tuple[int, int]): Figure dimensions
    - color_map (str or List[str]): Colors for clusters
    - alpha_fill (float): Fill transparency
    - linestyle (str): Line style
    - linewidth (int): Line width
    - yticks (Tuple[float, ...]): Y-axis tick positions
    - ytick_labels (List[str]): Y-axis tick labels
    - ylim (Tuple[float, float]): Y-axis limits
    - legend_loc (str): Legend position

Examples:

.. code-block:: python

    import numpy as np
    from gofast.plot.charts import create_radar_chart

    # Example 1: Basic radar chart
    data = np.random.rand(3, 5)  # 3 clusters, 5 variables
    categories = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']
    cluster_labels = ['Cluster 1', 'Cluster 2', 'Cluster 3']
    create_radar_chart(data, categories, cluster_labels)

    # Example 2: Customized radar chart
    data = np.random.rand(4, 6)
    categories = [f'Feature {i+1}' for i in range(6)]
    cluster_labels = [f'Group {i+1}' for i in range(4)]
    create_radar_chart(
        data,
        categories,
        cluster_labels,
        figsize=(10, 10),
        color_map='Set3',
        alpha_fill=0.3,
        linewidth=2,
        yticks=(0.2, 0.4, 0.6, 0.8, 1.0)
    )

    # Example 3: Complex visualization
    # Generate data with specific patterns
    data = np.zeros((3, 8))
    data[0] = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8])  # Cyclic pattern
    data[1] = np.array([0.3, 0.6, 0.9, 0.9, 0.9, 0.6, 0.3, 0.3])  # Peak pattern
    data[2] = np.random.uniform(0.4, 0.8, 8)  # Random pattern
    
    categories = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    cluster_labels = ['Pattern 1', 'Pattern 2', 'Pattern 3']
    create_radar_chart(
        data,
        categories,
        cluster_labels,
        title='Directional Patterns Analysis',
        color_map=['blue', 'red', 'green'],
        alpha_fill=0.2,
        yticks=(0.2, 0.4, 0.6, 0.8, 1.0),
        ytick_labels=['20%', '40%', '60%', '80%', '100%']
    )

create_base_radar_chart
~~~~~~~~~~~~~~~~~~~~
Creates a basic radar chart with default styling settings.

Uses the same mathematical foundation as create_radar_chart but with fixed styling parameters.

Parameters:
    - d (ArrayLike): 2D array of shape (n_clusters, n_variables)
    - categories (List[str]): Names of variables
    - cluster_labels (List[str]): Labels for clusters
    - title (str): Chart title

Examples:

.. code-block:: python

    import numpy as np
    from gofast.plot.charts import create_base_radar_chart

    # Example 1: Simple radar chart
    data = np.random.rand(3, 5)
    categories = ['A', 'B', 'C', 'D', 'E']
    cluster_labels = ['Group 1', 'Group 2', 'Group 3']
    create_base_radar_chart(data, categories, cluster_labels)

    # Example 2: Visualization with more variables
    data = np.random.rand(2, 8)
    categories = [f'Metric {i+1}' for i in range(8)]
    cluster_labels = ['Set A', 'Set B']
    create_base_radar_chart(
        data, 
        categories, 
        cluster_labels,
        title='Comparison of Metric Sets'
    )

    # Example 3: Pattern comparison
    # Create data with specific patterns
    pattern1 = np.array([0.8, 0.6, 0.9, 0.7, 0.8, 0.9])
    pattern2 = np.array([0.6, 0.8, 0.7, 0.9, 0.6, 0.7])
    data = np.vstack([pattern1, pattern2])
    
    categories = ['Speed', 'Accuracy', 'Recall', 'Precision', 'F1', 'ROC']
    cluster_labels = ['Model A', 'Model B']
    create_base_radar_chart(
        data,
        categories,
        cluster_labels,
        title='Model Performance Comparison'
    ).. _charts:

Charts and Advanced Visualizations
===============================

.. currentmodule:: gofast.plot.charts

The :mod:`gofast.plot.charts` module provides specialized charting capabilities for complex data visualization needs. This module implements various chart types with advanced customization options, focusing on clarity and insights in data presentation.

Key Features
-----------
- Interactive charts with customizable elements
- Multiple chart types for different data scenarios
- Advanced styling and formatting options
- Publication-ready output capabilities
- Responsive design for different display sizes

Chart Types Overview
------------------

1. Basic Charts
~~~~~~~~~~~~~
- Line charts with multiple series
- Bar charts (vertical, horizontal, stacked)
- Scatter plots with customizable markers
- Area charts with fill options

2. Statistical Charts
~~~~~~~~~~~~~~~~~~
- Box plots with outlier detection
- Violin plots for distribution visualization
- KDE (Kernel Density Estimation) plots
- Q-Q plots for distribution comparison

3. Categorical Charts
~~~~~~~~~~~~~~~~~~
- Grouped bar charts
- Stacked bar charts
- Dot plots
- Category comparison charts

4. Specialized Charts
~~~~~~~~~~~~~~~~~~
- Radar/Spider charts
- Polar charts
- Sunburst diagrams
- Treemaps

Classes and Functions
-------------------

RadarChart
~~~~~~~~~
Creates radar/spider charts for multivariate data comparison.

Parameters:
    - fig_size (tuple): Figure dimensions
    - n_levels (int): Number of concentric levels
    - plot_type (str): Type of radar plot ('polygon', 'circle')
    - fill (bool): Whether to fill the radar plot
    - alpha (float): Fill transparency

Examples:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from gofast.plot.charts import RadarChart

    # Example 1: Basic Radar Chart
    # Create sample data
    categories = ['Speed', 'Power', 'Agility', 'Intelligence', 'Stamina']
    values = [85, 92, 78, 95, 88]

    # Initialize and create radar chart
    radar = RadarChart(
        fig_size=(10, 10),
        n_levels=5,
        plot_type='polygon'
    )
    
    radar.plot(
        categories=categories,
        values=values,
        title='Character Attributes',
        color='blue',
        alpha=0.25
    )
    plt.show()

    # Example 2: Multiple Series Radar Chart
    # Create multiple series data
    data = pd.DataFrame({
        'Player1': [85, 92, 78, 95, 88],
        'Player2': [92, 85, 95, 78, 90],
        'Player3': [78, 88, 90, 85, 95]
    }, index=categories)

    # Plot multiple series
    radar = RadarChart(
        fig_size=(12, 12),
        n_levels=6,
        plot_type='circle'
    )
    
    colors = ['blue', 'red', 'green']
    for i, (name, values) in enumerate(data.items()):
        radar.plot(
            categories=categories,
            values=values,
            label=name,
            color=colors[i],
            alpha=0.2
        )
    
    plt.legend()
    plt.show()

SunburstChart
~~~~~~~~~~~
Creates hierarchical sunburst diagrams for nested categorical data.

Parameters:
    - fig_size (tuple): Figure dimensions
    - start_angle (float): Starting angle in degrees
    - direction (str): Direction of hierarchy ('inward', 'outward')
    - color_scheme (str): Color palette for segments

Examples:

.. code-block:: python

    from gofast.plot.charts import SunburstChart

    # Example 1: Basic Sunburst Chart
    # Create hierarchical data
    data = {
        'A': {
            'A1': 20,
            'A2': 30,
            'A3': {
                'A3a': 15,
                'A3b': 10
            }
        },
        'B': {
            'B1': 25,
            'B2': 20
        },
        'C': 30
    }

    # Create sunburst chart
    sunburst = SunburstChart(
        fig_size=(12, 12),
        start_angle=90,
        direction='outward'
    )
    
    sunburst.plot(
        data=data,
        title='Revenue Distribution',
        cmap='viridis'
    )
    plt.show()

    # Example 2: Complex Hierarchical Data
    # Create more complex nested data
    data = {
        'Products': {
            'Electronics': {
                'Phones': {
                    'iPhone': 300,
                    'Android': 250,
                    'Other': 50
                },
                'Laptops': {
                    'Windows': 200,
                    'MacBook': 180,
                    'Linux': 70
                }
            },
            'Clothing': {
                'Men': 400,
                'Women': 450,
                'Children': 200
            }
        },
        'Services': {
            'Maintenance': 300,
            'Installation': 200,
            'Support': {
                'Phone': 150,
                'Email': 100,
                'Chat': 80
            }
        }
    }

    sunburst = SunburstChart(
        fig_size=(15, 15),
        start_angle=90,
        direction='outward'
    )
    
    sunburst.plot(
        data=data,
        title='Company Revenue Breakdown',
        cmap='tab20c',
        label_threshold=50  # Only show labels for segments > 50
    )
    plt.show()

TreemapChart
~~~~~~~~~~
Creates treemap visualizations for hierarchical data representation.

Parameters:
    - fig_size (tuple): Figure dimensions
    - padding (float): Space between rectangles
    - color_scheme (str): Color palette
    - text_kwargs (dict): Text styling options

Examples:

.. code-block:: python

    from gofast.plot.charts import TreemapChart
    import numpy as np

    # Example 1: Basic Treemap
    # Create sample data
    data = {
        'A': {'A1': 100, 'A2': 200, 'A3': 150},
        'B': {'B1': 300, 'B2': 200},
        'C': {'C1': 150, 'C2': 100, 'C3': 100}
    }

    treemap = TreemapChart(
        fig_size=(12, 8),
        padding=0.02
    )
    
    treemap.plot(
        data=data,
        title='Market Share Distribution',
        cmap='Blues'
    )
    plt.show()

    # Example 2: Complex Treemap with Custom Styling
    # Create more complex data
    np.random.seed(42)
    sectors = ['Technology', 'Healthcare', 'Finance', 'Energy']
    companies = {
        sector: {
            f'Company{i}': np.random.randint(100, 1000)
            for i in range(1, 6)
        }
        for sector in sectors
    }

    # Custom text styling
    text_style = {
        'fontsize': 12,
        'fontweight': 'bold',
        'fontfamily': 'serif'
    }

    treemap = TreemapChart(
        fig_size=(15, 10),
        padding=0.03,
        text_kwargs=text_style
    )
    
    treemap.plot(
        data=companies,
        title='Market Capitalization by Sector',
        cmap='viridis',
        show_values=True,
        value_format='${:,.0f}M'
    )
    plt.show()

PolarChart
~~~~~~~~~
Creates polar charts for cyclic or periodic data visualization.

Parameters:
    - fig_size (tuple): Figure dimensions
    - gridlines (bool): Show/hide gridlines
    - theta_direction (int): Direction of angle increase
    - start_angle (float): Starting angle in degrees

Examples:

.. code-block:: python

    from gofast.plot.charts import PolarChart
    import numpy as np

    # Example 1: Basic Polar Plot
    # Create sample data
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.cos(3*theta)

    polar = PolarChart(
        fig_size=(10, 10),
        gridlines=True
    )
    
    polar.plot(
        theta=theta,
        r=r,
        title='Polar Pattern',
        color='blue'
    )
    plt.show()

    # Example 2: Multiple Polar Patterns
    # Create multiple patterns
    theta = np.linspace(0, 2*np.pi, 100)
    patterns = {
        'Pattern 1': np.cos(3*theta),
        'Pattern 2': np.sin(2*theta),
        'Pattern 3': np.abs(np.sin(4*theta))
    }

    polar = PolarChart(
        fig_size=(12, 12),
        gridlines=True,
        start_angle=90
    )
    
    colors = ['blue', 'red', 'green']
    for (name, r), color in zip(patterns.items(), colors):
        polar.plot(
            theta=theta,
            r=r,
            label=name,
            color=color,
            alpha=0.5
        )
    
    plt.legend()
    plt.show()

    # Example 3: Wind Rose Diagram
    # Create wind data
    directions = np.arange(0, 360, 30)
    speeds = np.random.normal(10, 2, len(directions))

    polar = PolarChart(
        fig_size=(12, 12),
        gridlines=True,
        start_angle=0
    )
    
    polar.plot_windrose(
        directions=directions,
        speeds=speeds,
        bins=5,
        cmap='viridis',
        title='Wind Speed Distribution'
    )
    plt.show()

Advanced Usage and Customization
----------------------------

1. Custom Color Schemes

.. code-block:: python

    import matplotlib.colors as mcolors

    # Create custom colormap
    def create_custom_cmap(start_color, end_color, n_steps=100):
        start_rgb = mcolors.to_rgb(start_color)
        end_rgb = mcolors.to_rgb(end_color)
        
        return mcolors.LinearSegmentedColormap.from_list(
            'custom_cmap',
            [start_rgb, end_rgb],
            N=n_steps
        )

    # Usage with charts
    custom_cmap = create_custom_cmap('lightblue', 'darkblue')
    
    radar = RadarChart(fig_size=(12, 12))
    radar.plot(
        categories=categories,
        values=values,
        cmap=custom_cmap
    )

2. Animation Support

.. code-block:: python

    import matplotlib.animation as animation

    def animate_radar(data_frames, categories):
        """Create animated radar chart."""
        fig = plt.figure(figsize=(12, 12))
        radar = RadarChart(fig=fig)
        
        def update(frame):
            plt.cla()
            values = data_frames[frame]
            radar.plot(categories=categories, values=values)
            return fig,
        
        ani = animation.FuncAnimation(
            fig, update,
            frames=len(data_frames),
            interval=200
        )
        return ani

    # Usage
    data_frames = [
        np.random.uniform(50, 100, len(categories))
        for _ in range(10)
    ]
    animation = animate_radar(data_frames, categories)
    plt.show()

3. Interactive Features

.. code-block:: python

    from matplotlib.widgets import Slider, Button

    class InteractiveTreemap:
        def __init__(self, data, fig_size=(12, 8)):
            self.data = data
            self.fig = plt.figure(figsize=fig_size)
            self.treemap = TreemapChart(fig=self.fig)
            self.setup_controls()
        
        def setup_controls(self):
            """Setup interactive controls."""
            ax_depth = plt.axes([0.2, 0.02, 0.6, 0.03])
            self.depth_slider = Slider(
                ax_depth, 'Depth', 1, 5,
                valinit=2, valstep=1
            )
            self.depth_slider.on_changed(self.update)
            
            ax_button = plt.axes([0.8, 0.02, 0.1, 0.03])
            self.reset_button = Button(ax_button, 'Reset')
            self.reset_button.on_clicked(self.reset)
        
        def update(self, val):
            """Update visualization based on controls."""
            plt.clf()
            depth = int(self.depth_slider.val)
            self.treemap.plot(
                self.data,
                max_depth=depth
            )
        
        def reset(self, event):
            """Reset to default view."""
            self.depth_slider.reset()
        
        def show(self):
            plt.show()

    # Usage
    interactive_treemap = InteractiveTreemap(data)
    interactive_treemap.show()

Best Practices
------------

1. Data Preparation
~~~~~~~~~~~~~~~~

.. code-block:: python

    def prepare_hierarchical_data(df, hierarchy_cols):
        """Prepare data for hierarchical visualizations."""
        result = {}
        for _, row in df.iterrows():
            current = result
            for col in hierarchy_cols[:-1]:
                key = row[col]
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[row[hierarchy_cols[-1]]] = row['value']
        return result

2. Style Management
~~~~~~~~~~~~~~~~

.. code-block:: python

    class ChartStyleManager:
        """Manage consistent styling across charts."""
        def __init__(self):
            self.style_dict = {
                'figure.figsize': (12, 8),
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'font.family': 'serif',
                'axes.grid': True,
                'grid.alpha': 0.3
            }
        
        def apply_style(self):
            plt.style.use(self.style_dict)
        
        def reset_style(self):
            plt.style.use('default')

3. Export Utilities
~~~~~~~~~~~~~~~~

.. code-block:: python

    class ChartExporter:
        """Handle chart export in various formats."""
        @staticmethod
        def save_chart(fig, filename, format='png', dpi=300):
            fig.savefig(
                f"{filename}.{format}",
                format=format,
                dpi=dpi,
                bbox_inches='tight'
            )
        
        @staticmethod
        def export_to_html(fig, filename):
            import mpld3
            html_str = mpld3.fig_to_html(fig)
            with open(f"{filename}.html", 'w') as f:
                f.write(html_str)

See Also
--------
- :mod:`gofast.plot.explore`: Exploratory visualization tools
- :mod:`gofast.plot.eval`: Evaluation plots
- :mod:`gofast.plot.utils`: Plotting utilities

References
----------
.. [1] Few, S. (2009). Now You See It: Simple Visualization Techniques for
       Quantitative Analysis. Analytics Press.

.. [2] Cairo, A. (2016). The Truthful Art: Data, Charts, and Maps for
       Communication. New Riders.

.. [3] Munzner, T. (2014). Visualization Analysis and Design. CRC Press.