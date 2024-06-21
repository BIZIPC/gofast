# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from ..api.types import ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Union 
from ..tools.coreutils import is_iterable 
from ..tools.validator import is_frame 

__all__=[
    "plot_pie_charts", "create_radar_chart", "create_base_radar_chart"
    ]

def plot_pie_charts(
    data: DataFrame, /, 
    columns: Optional[Union[str, List[str]]] = None,
    bin_numerical: bool = True,
    num_bins: int = 4,
    handle_missing: str = 'exclude',
    explode: Optional[Union[Tuple[float, ...], str]] = None,
    shadow: bool = True,
    startangle: int = 90,
    cmap: str = 'viridis',
    autopct: str = '%1.1f%%',
    verbose: int = 0
):
    """
    Plots pie charts for categorical and numerical columns in a DataFrame. 
    
    Function automatically detects and appropriately treats each type. 
    Numerical columns can be binned into categories.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data.
    columns : str or list of str, optional
        Specific columns to plot. If None, all columns are plotted.
    bin_numerical : bool, optional
        If True, numerical columns will be binned into categories before plotting.
    num_bins : int, optional
        Number of bins to use for numerical columns if bin_numerical is True.
    handle_missing : {'exclude', 'include'}, optional
        How to handle missing values in data. 'exclude' will ignore them,
        while 'include' will treat them as a separate category.
    explode, shadow, startangle, cmap, autopct : various
        Formatting options for the pie charts, similar to matplotlib's pie 
        chart configuration.
    verbose : int, optional
        Verbosity level. Higher values increase the amount of informational output.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.charts import plot_pie_charts
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'D'],
    ...     'Values': [1, 2, 3, 4, 5, 6, 7, 8]
    ... })
    >>> plot_pie_charts(df, bin_numerical=True, num_bins=3)
    """
    is_frame (data, df_only=True, raise_exception= True)
    if columns is None:
        columns = data.columns.tolist()
    columns = is_iterable(columns, exclude_string= True, transform=True )
    
    valid_columns = [col for col in columns if col in data.columns]
    n_cols = len(valid_columns)
    rows, cols = (n_cols // 4 + (n_cols % 4 > 0), min(n_cols, 4)
                  ) if n_cols > 1 else (1, 1)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    axs = np.array(axs).flatten()

    for idx, column in enumerate(valid_columns):
        column_data = data[column]
        if column_data.dtype.kind in 'iufc' and bin_numerical:  # Numerical data
            column_data = _handle_numerical_data(column_data, bin_numerical, num_bins)
        column_data = _handle_missing_data(column_data, handle_missing)
        labels, sizes = _compute_sizes_and_labels(column_data)

        _plot_pie_chart(axs[idx], labels, sizes, explode, shadow, startangle, 
                        cmap, autopct, f'{column} Distribution')

    plt.tight_layout()
    plt.show()
    
def _handle_numerical_data(column_data: pd.Series, bin_numerical: bool,
                           num_bins: int) -> pd.Series:
    """Bins numerical data into categories if specified."""
    if bin_numerical:
        return pd.cut(column_data, bins=num_bins, include_lowest=True)
    return column_data

def _handle_missing_data(
        column_data: pd.Series, handle_missing: str  ) -> pd.Series:
    """Handles missing data based on user preference."""
    if handle_missing == 'exclude':
        return column_data.dropna()
    return column_data.fillna('Missing')

def _compute_sizes_and_labels(
        column_data: pd.Series) -> Tuple[List[str], List[float]]:
    """Computes the sizes and labels for the pie chart."""
    sizes = column_data.value_counts(normalize=True)
    labels = sizes.index.astype(str).tolist()
    sizes = sizes.values.tolist()
    return labels, sizes

def _plot_pie_chart(
        ax, labels: List[str], sizes: List[float], 
        explode: Optional[Union[Tuple[float, ...], str]], shadow: bool, 
        startangle: int, cmap: str, autopct: str, title: str):
    """Plots a single pie chart on the given axis."""
    # Adjust explode based on the number of labels if 'auto' is selected
    if explode == 'auto':
        explode = [0.1 if i == sizes.index(max(sizes)) else 0 for i in range(len(labels))]
    elif explode is None:
        explode = (0,) * len(labels)
    else:
        # Ensure explode is correctly sized for the current pie chart
        if len(explode) != len(labels):
            raise ValueError(f"The length of 'explode' ({len(explode)}) does "
                             f"not match the number of categories ({len(labels)}).")
    
    ax.pie(sizes, explode=explode, labels=labels, autopct=autopct,
           shadow=shadow, startangle=startangle, colors=plt.get_cmap(cmap)(
               np.linspace(0, 1, len(labels))))
    ax.set_title(title)
    ax.axis('equal')  # Ensures the pie chart is drawn as a circle
    
def create_radar_chart(
    d: ArrayLike, /, categories: List[str], 
    cluster_labels: List[str], 
    title: str = "Radar plot Umatrix cluster properties",
    figsize: Tuple[int, int] = (6, 6), 
    color_map: Union[str, List[str]] = 'Set2', 
    alpha_fill: float = 0.25, 
    linestyle: str = 'solid', 
    linewidth: int = 1,
    yticks: Tuple[float, ...] = (0.5, 1, 1.5), 
    ytick_labels: Union[None, List[str]] = None,
    ylim: Tuple[float, float] = (0, 2),
    legend_loc: str = 'upper right'
   ) -> None:
    """
    Create a radar chart with one axis per variable.

    Parameters
    ----------
    d : array-like
        2D array with shape (n_clusters, n_variables), where each row 
        represents a different cluster and each column represents a 
        different variable.
    categories : list of str
        List of variable names corresponding to the columns in the data.
    cluster_labels : list of str
        List of labels for the different clusters.
    title : str, optional
        The title of the radar chart. Default is "Radar plot Umatrix cluster 
        properties".
    figsize : tuple, optional
        The size of the figure to plot (width, height in inches). Default is (6, 6).
    color_map : str or list, optional
        Colormap or list of colors for the different clusters. Default is 'Set2'.
    alpha_fill : float, optional
        Alpha value for the filled area under the plot. Default is 0.25.
    linestyle : str, optional
        The style of the line in the plot. Default is 'solid'.
    linewidth : int, optional
        The width of the lines. Default is 1.
    yticks : tuple, optional
        Tuple containing the y-ticks values. Default is (0.5, 1, 1.5).
    ytick_labels : list of str, optional
        List of labels for the y-ticks, must match the length of yticks. 
        If None, yticks will be used as labels. Default is None.
    ylim : tuple, optional
        Tuple containing the min and max values for the y-axis. Default is (0, 2).
    legend_loc : str, optional
        The location of the legend. Default is 'upper right'.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the radar chart.
    ax : Axes
        The matplotlib Axes object for the radar chart.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.charts import create_radar_chart
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> create_radar_chart(data, categories, cluster_labels)
    """

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, len(categories),
                         endpoint=False).tolist()
    # The plot is made in a circular (not polygon) space, so we need to 
    # "complete the loop"and append the start to the end.
    angles += angles[:1]  # complete the loop
    d= np.array(d)
    d = np.concatenate((d, d[:,[0]]), axis=1)
    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    plt.title(title, y=1.08)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    if ytick_labels is None:
        ytick_labels = [str(ytick) for ytick in yticks]
    plt.yticks(yticks, ytick_labels, color="grey", size=7)
    plt.ylim(*ylim)

    # Plot data and fill with color
    for idx, (row, label) in enumerate(zip(d, cluster_labels)):
        color = plt.get_cmap(color_map)(idx / len(d))
        ax.plot(angles, row, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)
        ax.fill(angles, row, color=color, alpha=alpha_fill)

    # Add a legend
    plt.legend(loc=legend_loc, bbox_to_anchor=(0.1, 0.1))

    plt.show()
    return fig, ax

def create_base_radar_chart(
    d: ArrayLike,/,   categories: List[str], 
    cluster_labels: List[str],
    title:str="Radar plot Umatrix cluster properties"
    ):
    """
    Create a radar chart with one axis per variable.

    Parameters
    ----------
    data : array-like
        2D array with shape (n_clusters, n_variables), where each row 
        represents a different
        cluster and each column represents a different variable.
    categories : list of str
        List of variable names corresponding to the columns in the data.
    cluster_labels : list of str
        List of labels for the different clusters.
    title : str, optional
        The title of the radar chart. Default is "Radar plot
        Umatrix cluster properties".

    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the radar chart.
    ax : Axes
        The matplotlib Axes object for the radar chart.
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.charts import create_base_radar_chart
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> create_base_radar_chart(data, categories, cluster_labels)
    """
    # Number of variables we're plotting.
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is made in a circular (not polygon) space, so we need to 
    # "complete the loop"and append the start to the end.
    angles += angles[:1]
    d = np.array(d)
    d = np.concatenate((d, d[:,[0]]), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.5, 1, 1.5], ["0.5", "1", "1.5"], color="grey", size=7)
    plt.ylim(0, 2)

    # Plot data
    for i in range(len(d)):
        ax.plot(angles, d[i], linewidth=1, linestyle='solid', 
                label=cluster_labels[i])

    # Fill area
    ax.fill(angles, d[0], color='blue', alpha=0.25)

    # Add a legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)

    plt.show()
    
    return fig, ax