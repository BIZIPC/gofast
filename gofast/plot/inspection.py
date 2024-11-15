# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `inspection` module offers a range of visualization tools for inspecting
and understanding machine learning models. It includes functions for
plotting learning inspections, projection plots, heatmaps, 
sunburst charts, Sankey diagrams, Euler diagrams, Venn diagrams, and more, 
enhancing model interpretability and analysis.
"""

from __future__ import annotations 
import os 
import warnings 
import numpy as np 
import pandas as pd 
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve 

from ..api.types import NDArray, ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Dict, Union, Any
from ..api.util import to_snake_case 
from ..decorators import Dataify
from ..exceptions import PlotError 
from ..tools.coreutils import is_iterable, to_numeric_dtypes 
from ..tools.coreutils import _assert_all_types, projection_validator
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import check_X_y, check_array, validate_square_matrix 
from ..tools.validator import get_estimator_name, _check_consistency_size 
from ..tools.validator import parameter_validator, validate_sets
from ._config import PlotConfig 
from .utils import _manage_plot_kws, pobj, savefigure

__all__=[
    'plot_learning_inspection',
    'plot_learning_inspections',
    'plot_loc_projection',
    'plot_matshow',
    'plot_heatmapx',
    'plot_matrix',
    'plot_sunburst',
    'plot_sankey',
    'plot_euler_diagram',
    'create_upset_plot',
    'plot_venn_diagram',
    'plot_set_matrix',
    'plot_woodland', 
    'plot_l_curve', 
    ]


@Dataify(auto_columns=True, prefix="feature_")  
def plot_woodland(
    data: DataFrame,*,
    quadrant: str="upper_left",
    compute_corr: bool=False,
    method: str='pearson',
    annot: bool=True,
    fig_size: Tuple [int, int]=(11, 9),
    fmt: str=".2f",
    linewidths: float=.5,
    xrot: int=90,
    yrot: int=0,
    cmap: Optional[str]=None,
    cbar: bool=True,
    ax: Optional[mpl.axes.Axes]=None
):
    """
    Plot a heatmap of the correlation matrix or a given matrix.

    Function requires the input data to be a square matrix if `compute_corr`
    is False. The function provides flexibility in the visual representation
    of the correlation matrix by allowing selection of specific quadrants to display.
    The `quadrant` parameter is particularly useful for focusing on parts of a 
    larger matrix while analyzing data.

    Parameters
    ----------
    data : DataFrame or array-like
        The input data. If `compute_corr` is True, it should be a DataFrame; 
        otherwise, it must be a square matrix.
    quadrant : str, optional
        Specifies the quadrant of the correlation matrix to display. 
        Valid options: "upper_left" (default), "upper_right", "bottom_left", 
        and "bottom_right".
    compute_corr : bool, optional
        Indicates whether to compute the correlation matrix from `data`. 
        If True, `data` should be a DataFrame.
    method : str, optional
        The method used for computing the correlation matrix if `compute_corr` 
        is True. Default is 'pearson'.
    annot : bool, optional
        Specifies whether to annotate the heatmap with correlation values. 
        Default is True.
    fig_size : tuple, optional
        The dimensions of the figure to be created. Default is (11, 9).
    fmt : str, optional
        The format for annotating the heatmap. Default is ".2f".
    linewidths : float, optional
        The width of the lines that will divide each cell in the heatmap. 
        Default is 0.5.
    xrot : float, optional
        Rotation angle in degrees for x-axis labels. Default is 90.
    yrot : float, optional
        Rotation angle in degrees for y-axis labels. Default is 0.
    cmap : str or Colormap, optional
        The colormap for the heatmap. If None, a default diverging palette 
        is used.
    cbar : bool, optional
        Specifies whether to draw a colorbar. Default is True.
    ax : matplotlib Axes, optional
        The axes upon which to draw the heatmap. If None, a new figure 
        and axes will be created.

    Notes
    -----
    The woodland plot function in Python draws inspiration from Baig Abdullah 
    Al Shoumik's work on the R statistical Software group on the Meta platform. 
    His approach to visualizing correlation matrices in R emphasizes clarity and 
    effectiveness, which serves as a foundation for this implementation. An example 
    of Al Shoumik's R script is provided below to illustrate his method:

    ```R
    # Calculate the correlation coefficients
    WW <- corr_coef(www)

    # Create the plot with upper triangle, custom text sizes, and titles
    plot(WW,
         type="upper",
         size.text.cor=4,
         size.text.signif=4) +
         ggtitle("Correlation") +
         theme(text=element_text(family="B", color="black", size=17),
               plot.title=element_text(hjust=0.5, vjust=2, face="bold"))
    ```

    This Python implementation strives to replicate the intuitive display and 
    flexible configuration of correlation matrices found in Al Shoumik's R-based 
    methods, adapting them for use with Python's data handling and visualization 
    capabilities.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.plot.inspection import plot_woodland
    >>> df = pd.DataFrame(np.random.rand(10, 10), columns=list('ABCDEFGHIJ'))
    >>> plot_woodland(df, compute_corr=True, quadrant='upper_left', 
    ...                  cbar=True, cmap='coolwarm')

    See Also
    --------
    gofast.tools.dataops.analyze_data_corr : 
        Analyze and elegantly display correlation tables. Provides options for 
        interpreting correlation data.
    gofast.stats.utils.correlation : 
        Compute correlation, optionally handling both categorical and numeric 
        data.
    pd.DataFrame.corr : 
        Compute correlation matrix for a pandas DataFrame.
    pd.DataFrame.go_corr : 
        Enhance pandas DataFrame with 'go_corr' function to compute correlation 
        matrices and optionally provide enhanced visual displays.
    """
    # Convert quadrant parameter to snake case
    quadrant = to_snake_case(quadrant)
    
    # Validate quadrant option
    valid_options = {"bottom_left", "upper_left", "upper_right", "bottom_right"}
    quadrant = parameter_validator("quadrant", target_strs=valid_options)(quadrant)
    
    data = to_numeric_dtypes(data) # validate 
    # Compute correlation matrix if needed
    if compute_corr:
        data = data.corr(method=method)
         
    # Validate that data is a square matrix suitable 
    data = validate_square_matrix(data, message=( 
        "Ensure the data is square matrix, or set `compute_corr` to `True`"
        " if data size adjustment is required.")
        )
    # Setup the plotting figure
    if ax is None:
        f, ax = plt.subplots(figsize=fig_size)

    # Initialize mask to True (hide all)
    mask = np.ones_like(data, dtype=bool)

    # Configure mask according to the quadrant
    if quadrant == "bottom_left":
        # Show lower triangle including diagonal
        mask[np.tril_indices_from(mask)]=False
    elif quadrant == "upper_right":
        # Show upper triangle including diagonal
        mask[np.triu_indices_from(mask)]=False 
    elif quadrant == "upper_left":
        # Show upper triangle and rotate 
        mask[np.triu_indices_from(mask)] = False 
        mask=np.rot90 (mask, 1)
    elif quadrant == "bottom_right":
        # Show lower triangle and rotate
        mask[np.tril_indices_from(mask)] = False
        mask=np.rot90 (mask, 1)

    # Set the colormap if none provided
    if cmap is None:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and specified options
    sns.heatmap(data, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=linewidths, 
                cbar_kws={"shrink": .5},
                annot=annot, fmt=fmt, cbar=cbar)

    # Adjust label positioning based on the quadrant
    if quadrant in ["upper_left", "upper_right"]:
        # X-axis labels on the top
        ax.xaxis.set_tick_params(
            top=True, bottom=False, labeltop=True,
            labelbottom=False)
        plt.xticks(rotation=xrot, ha='right')
        plt.yticks(rotation=yrot)
        if quadrant == "upper_right":
            # Y-axis labels on the right
            ax.yaxis.set_tick_params(
                left=False, right=True, labelleft=False,
                labelright=True)
    else:
        plt.xticks(rotation=xrot, ha='right')
        plt.yticks(rotation=yrot)
        if quadrant == "bottom_right":
            # Y-axis labels on the right
            ax.yaxis.set_tick_params(
                left=False, right=True, labelleft=False,
                labelright=True)
    # Display the plot
    plt.show()
   
def plot_l_curve(
    rms: Union[List[float], np.ndarray], 
    roughness: Union[List[float], np.ndarray], 
    tau: Optional[Union[List[float], np.ndarray]] = None, 
    hansen_point: Optional[Tuple[float, float]] = None, 
    rms_target: Optional[float] = None,
    view_tline: bool = False,
    hpoint_kws: Dict[str, Any] = dict(), 
    fig_size: Tuple[int, int] = (10, 4),
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None, 
    style: str = 'classic', 
    savefig: Optional[str] = None, 
    **plot_kws: Any
) -> plt.Axes:
    """
    Plot the Hansen L-curve.

    The L-curve criteria is used to determine the suitable model 
    after running multiple inversions with different :math:`\tau` values. 
    The function plots RMS vs. Roughness with an option to highlight a 
    specific point named Hansen point [1]_.

    The :math:`\tau` represents the measure of compromise between data fit and 
    model smoothness. To find out an appropriate :math:`\tau` value, the 
    inversion was carried out with different :math:`\tau` values. The RMS error 
    obtained from each inversion is plotted against model roughness [2]_.

    Parameters 
    ----------
    rms : Union[List[float], np.ndarray]
        Corresponding list or array-like of RMS values.

    roughness : Union[List[float], np.ndarray]
        List or array-like of roughness values.

    tau : Optional[Union[List[float], np.ndarray]], optional
        List of tau values to visualize as text mark in the plot. Default is None.

    hansen_point : Optional[Tuple[float, float]], optional
        The Hansen point to visualize in the plot. It can be determined 
        automatically if set to "auto". Default is None.

    rms_target : Optional[float], optional
        The root-mean-squared target. If set, and `view_tline` is ``False``, 
        the target value should be an axis limit. Default is None.

    view_tline : bool, default=False
        Whether to display the target line.

    hpoint_kws : Dict[str, Any], optional
        Keyword arguments to highlight the Hansen point in the figure.
        Default is an empty dict.

    fig_size : Tuple[int, int], optional
        Figure size in inches. Default is (10, 4).

    ax : Optional[plt.Axes], optional
        Axes to plot on. If None, a new figure and axes are created.
        Default is None.

    fig : Optional[plt.Figure], optional
        Figure to save automatically if provided. Default is None.

    style : str, optional
        Matplotlib style to use. Default is 'classic'.

    savefig : Optional[str], optional
        File path to save the figure. If None, the figure is not saved. 
        Default is None.

    plot_kws : Dict[str, Any]
        Additional keyword arguments for `ax.plot`.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes object with the plot.
        
    
    Examples
    --------
    >>> from gofast.tools.utils import plot_l_curve
    >>> roughness_data = [0, 50, 100, 150, 200, 250, 300, 350]
    >>> RMS_data = [3.16, 3.12, 3.1, 3.08, 3.06, 3.04, 3.02, 3]
    >>> highlight_data = (50, 3.12)
    >>> plot_l_curve(RMS_data, roughness_data, hansen_point=highlight_data)
    
    
    References
    ----------
    .. [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
           the regularization of discrete ill-posed problems. SIAM Journal
           on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
        
    .. [2] Kouadio, K.L, Jianxin Liu, lui R., Liu W. Boukhalfa Z. (2024). An integrated
           Approach for sewage diversion: Case of Huayuan Mine. Geophysics, 89(4). 
           https://doi.org/10.1190/geo2023-0332.1
           
    """
    rms= np.array (
        is_iterable(rms, exclude_string= True, transform =True ), 
                   dtype =float) 
    roughness = np.array( 
        is_iterable(roughness , exclude_string= True, transform =True 
                            ), dtype = float) 
    
    # Create the plot
    plt.style.use (style )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # manage the plot keyword argument and remove 
    # the default is given.
    plot_kws = _manage_plot_kws (plot_kws, dict(
        marker='o',linestyle='-', color='black')
                    )
    ax.plot(roughness, rms, **plot_kws
             )
    # Highlight the specific hansen point if "auto" 
    # option is provided.
    if str(hansen_point).lower().strip() =="auto": 
        hansen_point = _get_hansen_point(roughness, rms)
        
    if hansen_point is not None:
        if len(hansen_point)!=2: 
            raise ValueError("Hansen knee point needs a tuple of (x, y)."
                             f" Got {hansen_point}")
        hpoint_kws = _manage_plot_kws(hpoint_kws, 
                                         dict(marker='o', color='red'))
        ax.plot(hansen_point[0], hansen_point[1], **hpoint_kws)
        ax.annotate(str(hansen_point[0]), 
                     hansen_point, textcoords="offset points",
                     xytext=(0,10), ha='center'
                     )
    if tau is not None: 
        tau = is_iterable(tau, exclude_string= True, transform =True )
        rough_rms = np.hstack ((roughness, rms))
        for tvalues, text in zip ( rough_rms, tau): 
            if ( 
                    tvalues[0]==hansen_point[0] 
                    and tvalues[1]==hansen_point[1]
                    ): 
                # hansen point is found then skip it
                continue 
            ax.annotate(str(text), tvalues, textcoords="offset points",
                         xytext=(0,10), ha='center'
                         )
    if rms_target: 
        rms_target = float( _assert_all_types(
            rms_target, float, int, objname='RMS target')) 
        
    if view_tline: 
        if rms_target is None: 
            warnings.warn("Missing RMS target value. Could not display"
                          " the target line.")
        else:
            ax.axhline(y=float(rms_target), color='k', linestyle=':')

    if rms_target: 
        # get the extent value from the min 
        extent_value = abs ( rms_target - min(rms )) 
        ax.set_ylim ( [rms_target - extent_value  if view_tline else 0  ,
                       max(rms)+ extent_value]) # .3 extension limit

    # Setting the labels and title
    ax.set_xlabel('Roughness')
    ax.set_ylabel('RMS')
    ax.set_title('RMS vs. Roughness')

    # savefig 
    if savefig is not  None: savefigure (fig, savefig, dpi = 300)
    # Show the plot    
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def _get_hansen_point ( roughness, RMS): 
    """ Get the Hansen point automatically.
    
    An isolated part of :func:`~plot_l_curve`. 
    
    The L-curve criteria proposed by Hansen and O'Leary (1993)[1]_ and 
    Hansen (1998) [2]_, which suggests that the s value at the knee of 
    the curve is most appropriate, have been adopted.

    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
        
    [2] Hansen, P. C. (1998). Rank deficient and discrete Ill: Posed problems, 
        numerical aspects of linear inversion (p. 247). Philadelphia: SIAM
    """
    # Calculate the curvature of the plot
    # Using a simple method to estimate the 'corner' of the L-curve
    curvature = []
    for i in range(1, len(roughness) - 1):
        # Triangle area method to calculate curvature
        x1, y1 = roughness[i-1], RMS[i-1]
        x2, y2 = roughness[i], RMS[i]
        x3, y3 = roughness[i+1], RMS[i+1]

        # Lengths of the sides of the triangle
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Semiperimeter of the triangle
        s = (a + b + c) / 2

        # Area of the triangle
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Curvature is 4 * area divided by product of the sides
        # (Heron's formula for the area of a triangle)
        if a * b * c == 0:
            k = 0
        else:
            k = 4 * area / (a * b * c)
        curvature.append(k)

    # Find the index of the point with the maximum curvature
    # +1 due to curvature array being shorter
    max_curvature_index = np.argmax(curvature) + 1  
    return roughness[max_curvature_index], RMS[max_curvature_index]

@ensure_pkg(
    "matplotlib_venn", 
    extra="plot_venn_diagram requires the 'matplotlib_venn' package.", 
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
 )
def plot_venn_diagram(
    sets: Union[Dict[str, set], List[np.ndarray]],
    set_labels: Optional[Tuple[str, ...]] = None,
    title: str = 'Venn Diagram',
    figsize: Tuple[int, int] = (8, 8),
    set_colors: Tuple[str, ...] = ('red', 'green', 'blue'),
    alpha: float = 0.5, 
    savefig: Optional[str] = None
) -> Axes:
    """
    Create and optionally save a Venn diagram for two or three sets.

    A Venn diagram is a visual representation of the mathematical or logical
    relationship between sets of items. It depicts these sets as circles,
    with the overlap representing items common to the sets. It's a useful
    tool for comparing and contrasting groups of elements, especially in
    the field of data analysis and machine learning.
    
    This function supports up to 3 sets. Venn diagrams for more than 3 sets
    are not currently supported due to visualization complexity.

    Parameters
    ----------
    sets : Union[Dict[str, set], List[np.ndarray]]
        Either a dictionary with two or three items, each being a set of elements,
        or a list of two or three arrays. The arrays should be of consistent length,
        and their elements will be treated as unique identifiers.
        
        - `sets` specifies the input sets to be visualized.

    set_labels : Optional[Tuple[str, ...]], optional
        A tuple containing labels for the sets in the diagram. If None, default
        labels will be used. Default is None.
        
        - `set_labels` provides custom labels for the sets.

    title : str, optional
        Title of the Venn diagram. Default is 'Venn Diagram'.
        
        - `title` specifies the title of the plot.

    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (8, 8).
        
        - `figsize` determines the size of the plot.

    set_colors : Tuple[str, ...], optional
        Colors for the sets. Default is ('red', 'green', 'blue').
        
        - `set_colors` defines the colors for each set.

    alpha : float, optional
        Transparency level of the set colors. Default is 0.5.
        
        - `alpha` sets the transparency level for the set colors.

    savefig : Optional[str], optional
        Path and filename to save the figure. If None, the figure is not saved.
        Example: 'path/to/figure.png'. Default is None.
        
        - `savefig` specifies the file path to save the plot.

    Returns
    -------
    ax : Axes
        The axes object of the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xgboost import XGBClassifier
    >>> from gofast.plot.inspection import plot_venn_diagram
    
    >>> set1 = {1, 2, 3}
    >>> set2 = {2, 3, 4}
    >>> ax = plot_venn_diagram({'Set1': set1, 'Set2': set2}, 
                               set_labels=('Set1', 'Set2'))

    >>> arr1 = np.array([1, 2, 3])
    >>> arr2 = np.array([2, 3, 4])
    >>> ax = plot_venn_diagram([arr1, arr2], ('Array1', 'Array2'))
    
    >>> # Example of comparing feature contributions of RandomForest and XGBoost
    >>> X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=2, n_redundant=0, 
                               random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42)

    >>> # Train classifiers
    >>> rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    >>> xgb = XGBClassifier(random_state=42).fit(X_train, y_train)

    >>> # Get feature importances
    >>> rf_features = set(np.argsort(rf.feature_importances_)[-5:])  # Top 5 features
    >>> xgb_features = set(np.argsort(xgb.feature_importances_)[-5:])

    >>> # Plot Venn diagram
    >>> ax = plot_venn_diagram(
        sets={'RandomForest': rf_features, 'XGBoost': xgb_features},
        set_labels=('RandomForest Features', 'XGBoost Features'),
        title='Feature Contribution Comparison'
    )
    >>> plt.show()
    
    >>> # Using dictionaries
    >>> set1 = {1, 2, 3}
    >>> set2 = {2, 3, 4}
    >>> set3 = {4, 5, 6}
    >>> ax = plot_venn_diagram({'Set1': set1, 'Set2': set2, 'Set3': set3}, 
                           ('Set1', 'Set2', 'Set3'))

    >>> # Using arrays
    >>> arr1 = np.array([1, 2, 3])
    >>> arr2 = np.array([2, 3, 4])
    >>> ax = plot_venn_diagram([arr1, arr2], ('Array1', 'Array2'))

    Notes
    -----
    This function can be particularly useful in scenarios like feature analysis
    in machine learning models, where understanding the overlap and uniqueness
    of feature contributions from different models can provide insights into
    model behavior and performance.

    The Venn diagram visually represents the following sets and their intersections:

    .. math::

        A \cup B, \quad A \cap B, \quad A - B, \quad B - A

    for two sets :math:`A` and :math:`B`.

    For three sets, it represents:

    .. math::

        A \cup B \cup C, \quad A \cap B, \quad A \cap C, \quad B \cap C, \quad A - B - C, \quad B - A - C, \quad C - A - B

    See Also
    --------
    matplotlib_venn : Python library for plotting Venn diagrams.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.

    .. [2] matplotlib-venn documentation. Retrieved from
       https://github.com/konstantint/matplotlib-venn
    """
    
    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
    
    sets = validate_sets( sets, mode='deep', allow_empty= False )
    num_sets = len(sets)

    # Validate input
    if num_sets not in [2, 3]:
        raise ValueError("Only 2 or 3 sets are supported.")

    if set_labels is not None:
        if len(set_labels) != num_sets:
            raise ValueError("Length of 'set_labels' must match the number of sets.")
    else:
        # Default labels if not provided
        set_labels = tuple(f'Set {i+1}' for i in range(num_sets))

    # Prepare set values for Venn diagram
    if isinstance(sets, dict):
        set_values = tuple(sets.values())
    elif isinstance(sets, list):
        set_values = tuple(set(arr) for arr in sets)

    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)

    # Choose the appropriate Venn function
    if num_sets == 2:
        venn2(subsets=set_values, set_labels=set_labels, 
                     set_colors=set_colors[:2], alpha=alpha)
        venn2_circles(subsets=set_values)
    elif num_sets == 3:
        venn3(subsets=set_values, set_labels=set_labels,
                     set_colors=set_colors[:3], alpha=alpha)
        venn3_circles(subsets=set_values)

    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return ax

@ensure_pkg("rpy2", extra="plot_euler_diagram requires the 'rpy2' package.", 
            auto_install=PlotConfig.install_dependencies ,
            use_conda=PlotConfig.use_conda 
            )
def plot_euler_diagram(sets: dict[str, set[int]]) -> None:
    """
    Creates an Area-Proportional Euler Diagram using R's venneuler package.

    Euler diagrams are a generalization of Venn diagrams that represent 
    relationships between different sets or groupings. Unlike Venn diagrams, 
    they do not require all possible logical relationships to be shown. This 
    function interfaces with R to create and display an Euler diagram for a 
    given set of data.

    Parameters
    ----------
    sets : dict
        A dictionary where keys are set names (`str`) and values are the 
        sets (`set[int]`).

        - `Set1` (`set[int]`): Elements in the first set.
        - `Set2` (`set[int]`): Elements in the second set.
        - `Set3` (`set[int]`): Elements in the third set.

    Returns
    -------
    None
        The function displays the Euler diagram.

    Examples
    --------
    >>> from gofast.plot.inspection import plot_euler_diagram
    >>> sets = {
    ...     "Set1": {1, 2, 3},
    ...     "Set2": {3, 4, 5},
    ...     "Set3": {5, 6, 7}
    ... }
    >>> plot_euler_diagram(sets)

    This will create an Euler diagram for three sets, showing their overlaps.
    `Set1` contains elements 1, 2, and 3; `Set2` contains 3, 4, and 5; and 
    `Set3` contains 5, 6, and 7. The diagram will visually represent the 
    intersections between these sets.

    Notes
    -----
    This function requires a working R environment with the `venneuler` package 
    installed, as well as `rpy2` installed in the Python environment. It creates 
    a temporary file to save the plot generated by R, reads this file into Python 
    for display, and then removes the file.

    Mathematically, the function aims to create an Euler diagram where the area 
    of each region is proportional to the number of elements in the corresponding 
    set and intersections. Given sets `A`, `B`, and `C`, the Euler diagram 
    represents their sizes and intersections:
    
    .. math::

        |A \cap B|, |A \cap C|, |B \cap C|, |A \cap B \cap C|

    The areas in the diagram correspond to these intersection sizes.

    See Also
    --------
    venneuler : R package for creating Euler and Venn diagrams.
    rpy2 : Python interface to R.

    References
    ----------
    .. [1] Wilkinson, L. (2012). Venneuler: Venn and Euler diagrams. R package 
       version 1.1-0. https://cran.r-project.org/web/packages/venneuler/

    """ 

    import rpy2.robjects as robjects
    # from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    import matplotlib.image as mpimg
    import tempfile
    
    sets = validate_sets (sets, mode='deep', allow_empty= False )
    # Activate automatic conversion between pandas dataframes and R dataframes
    pandas2ri.activate()

    # Define an R script
    r_script = """
    library(venneuler)
    plot_venneuler <- function(sets) {
        v <- venneuler(sets)
        plot(v)
    }
    """

    # Convert the Python dictionary to an R-readable string
    sets_r = 'list(' + ', '.join(f"{k}=c({', '.join(map(str, v))})" 
                                 for k, v in sets.items()) + ')'

    # Run the R script
    robjects.r(r_script)
    plot_venneuler = robjects.globalenv['plot_venneuler']

    # Temporary file to save the plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name

    # Execute the R function and save the plot
    robjects.r(f"png('{tmpfile_name}')")
    plot_venneuler(robjects.r(sets_r))
    robjects.r("dev.off()")

    # Display the saved plot in Python
    img = mpimg.imread(tmpfile_name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # Remove the temporary file
    os.remove(tmpfile_name)
  
@ensure_pkg("upsetplot",
            extra="create_upset_plot requires the 'upsetplot' package.", 
            auto_install=PlotConfig.install_dependencies ,
            use_conda=PlotConfig.use_conda 
            )
def create_upset_plot(
    data: Dict[str, set], 
    sort_by: str = 'degree', 
    show_counts: bool = False
 ) -> None:
    """
    Creates an UpSet plot, which is an alternative to Venn diagrams for more 
    than three sets.

    The UpSet plot focuses on the intersections of the sets rather than on 
    the sets themselves.

    Parameters
    ----------
    data : dict
        A dictionary where each key is the name of a set (`str`) and each 
        value is the set itself (`set[int]`).

        - `Set1` (`set[int]`): Elements in the first set.
        - `Set2` (`set[int]`): Elements in the second set.
        - `Set3` (`set[int]`): Elements in the third set.
        - `Set4` (`set[int]`): Elements in the fourth set.

    sort_by : str, optional
        The criteria to sort the bars. Options are 'degree' and 'cardinality'.
        'degree' will sort by the number of sets in the intersection, 
        'cardinality' by the size of the intersection. Default is 'degree'.

    show_counts : bool, optional
        Whether to show the counts at the top of the bars in the UpSet plot. 
        Default is False.

    Returns
    -------
    None
        The function displays the UpSet plot.

    Examples
    --------
    >>> from gofast.plot.inspection import create_upset_plot
    >>> sets = {
    ...     "Set1": {1, 2, 3},
    ...     "Set2": {3, 4, 5},
    ...     "Set3": {5, 6, 7},
    ...     "Set4": {7, 8, 9}
    ... }
    >>> create_upset_plot(sets, sort_by='degree', show_counts=True)

    This will create an UpSet plot for four sets, showing their intersections 
    and the counts of elements in each intersection.

    Notes
    -----
    The UpSet plot provides a visual representation of the intersections of 
    multiple sets. The horizontal bars represent the sizes of the individual 
    sets, while the vertical bars represent the sizes of their intersections.

    Mathematically, if we have sets :math:`A_1, A_2, \ldots, A_n`, the UpSet 
    plot represents the size of intersections:

    .. math::

        |A_{i_1} \cap A_{i_2} \cap \ldots \cap A_{i_k}|

    where :math:`i_1, i_2, \ldots, i_k` are indices of the sets involved in 
    the intersection.

    See Also
    --------
    upsetplot.plot : Function to create an UpSet plot.
    upsetplot.from_contents : Function to convert data into UpSet format.

    References
    ----------
    .. [1] Lex, A., Gehlenborg, N., Strobelt, H., Vuillemot, R., & Pfister, H.
       (2014). UpSet: Visualization of Intersecting Sets. IEEE Transactions on 
       Visualization and Computer Graphics, 20(12), 1983-1992. 
       https://doi.org/10.1109/TVCG.2014.2346248

    """
    
    from upsetplot import plot, from_contents 
    
    data = validate_sets (data, mode='deep', allow_empty= False,
                          element_type=int )
    # Convert the data into the format required by UpSetPlot
    upset_data = from_contents(data)

    # Create and display the UpSet plot
    plot(upset_data, sort_by=sort_by, show_counts=show_counts)
    
@ensure_pkg("plotly","plot_sunburst requires the 'plotly' package.", 
            auto_install=PlotConfig.install_dependencies ,
            use_conda=PlotConfig.use_conda 
    )
def plot_sankey(
    data: DataFrame, 
    source_col: str, 
    target_col: str, 
    value_col: str, 
    label_col: Optional[str] = None,
    figsize: Tuple[int, int] = (800, 600), 
    title: Optional[str] = None
):
    """
    Creates a Sankey diagram from a pandas DataFrame using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the source, target, and value columns 
        for the Sankey diagram.
        
        - `data` should have at least three columns: source, target, and value.
        
    source_col : str
        The name of the column in `data` that contains the source nodes.
        
        - `source_col` specifies the origin nodes in the flow.
        
    target_col : str
        The name of the column in `data` that contains the target nodes.
        
        - `target_col` specifies the destination nodes in the flow.
        
    value_col : str
        The name of the column in `data` that contains the flow values 
        between the nodes.
        
        - `value_col` indicates the quantity of flow from source to target.
        
    label_col : Optional[str], optional
        The name of the column in `data` that contains the labels of the nodes.
        If None, the nodes will be labeled with unique identifiers. Default is None.
        
        - `label_col` allows for custom labels on the nodes. If not provided, 
          the function will use unique identifiers derived from source and 
          target columns.
          
    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in pixels. Default is (800, 600).
        
        - `figsize` determines the size of the generated plot in pixels.
        
    title : Optional[str], optional
        The title of the plot. If None, no title is set. Default is None.
        
        - `title` specifies the title of the Sankey diagram.

    Returns
    -------
    go.Figure
        The Plotly Figure object with the Sankey diagram for further 
        tweaking and rendering.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.inspection import plot_sankey 
    >>> df = pd.DataFrame({
    ...     'source': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'target': ['C', 'D', 'C', 'D', 'E', 'F'],
    ...     'value': [8, 2, 4, 8, 4, 2],
    ...     'label': ['Node A', 'Node B', 'Node C', 'Node D', 'Node E', 'Node F']
    ... })
    >>> fig = plot_sankey(df, 'source', 'target', 'value', 'label', 
                          title='My Sankey Diagram')
    >>> fig.show()  

    Notes
    -----
    Sankey diagrams are helpful for visualizing flow data between different 
    nodes or stages. The width of the arrows or links is proportional to the 
    flow quantity, allowing for easy comparison of volume or value transfers.
    
    Mathematically, a Sankey diagram visualizes the flow distribution as:

    .. math::

        F(i, j) \quad \text{for all} \quad (i, j) \in S \times T

    where :math:`F(i, j)` represents the flow from source node :math:`i` to 
    target node :math:`j`. The width of the flow lines is proportional to 
    the values of :math:`F(i, j)`.

    See Also 
    --------
    gofast.tools.mathex.infer_sankey_columns : 
        Infers source, target, and value columns for a Sankey diagram 
        from a DataFrame.

    References
    ----------
    .. [1] Plotly Technologies Inc. (2021). Plotly Python Graphing Library. 
       Retrieved from https://plotly.com/python/
    
    """
    import plotly.graph_objects as go

    # Prepare the data for the Sankey diagram
    label_list =  data[label_col].unique().tolist() if label_col else\
        pd.concat([data[source_col], data[target_col]]).unique().tolist()
                  
    source_indices = data[source_col].apply(label_list.index).tolist()
    target_indices = data[target_col].apply(label_list.index).tolist()

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=label_list,
        ),
        link=dict(
            source=source_indices,  # indices correspond to labels
            target=target_indices,
            value=data[value_col].tolist()
        ))])
    
    # Update layout
    fig.update_layout(title_text=title, width=figsize[0], height=figsize[1])
    
    return fig

@ensure_pkg("plotly", extra="plot_sunburst requires the 'plotly' package.", 
            auto_install=PlotConfig.install_dependencies ,
            use_conda=PlotConfig.use_conda 
    )
def plot_sunburst(
    d: List[Dict[str, str]], /, 
    figsize: Tuple[int, int] = (10, 8), 
    color: Optional[List[str]] = None,
    title: Optional[str] = None, 
    savefig: Optional[str] = None
) :
    """
    Plots a sunburst chart with the given data and parameters using Plotly.

    Parameters
    ----------
    d : List[Dict[str, str]]
        The input data for the sunburst chart, where each dict contains 'name',
        'value', and 'parent' keys.
        
        - `name` : The name of the node.
        - `value` : The value associated with the node.
        - `parent` : The parent node of the current node. Root nodes have an 
          empty string as the parent.

    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in inches. Default is (10, 8).
        
        - `figsize` determines the size of the generated plot in inches.
        
    color : Optional[List[str]], optional
        A list of color codes for the segments of the sunburst chart. If None,
        default colors are used.
        
        - `color` specifies the color for each segment of the sunburst chart. 
          If not provided, Plotly's default colors will be used.

    title : Optional[str], optional
        The title of the plot. If None, no title is set. Default is None.
        
        - `title` specifies the title of the sunburst chart.

    savefig : Optional[str], optional
        Path and filename where the figure should be saved. If ends with 'html',
        saves an interactive HTML file. Otherwise, saves a static image of the
        plot. Default is None.
        
        - `savefig` determines the file path and format for saving the plot. 
          If the filename ends with '.html', an interactive HTML file is saved. 
          Otherwise, a static image is saved in the specified format.

    Returns
    -------
    go.Figure
        The Plotly Figure object with the plot for further tweaking and rendering.

    Examples
    --------
    >>> from gofast.plot.inspection import plot_sunburst
    >>> d = [
    ...     {"name": "Category A", "value": 10, "parent": ""},
    ...     {"name": "Category B", "value": 20, "parent": ""},
    ...     {"name": "Subcategory A1", "value": 5, "parent": "Category A"},
    ...     {"name": "Subcategory A2", "value": 5, "parent": "Category A"}
    ... ]
    >>> plot_sunburst(d, figsize=(8, 8), title='Sunburst Chart Example')

    Notes
    -----
    Sunburst charts are used to visualize hierarchical data spanning outwards
    radially from root to leaves. The parent-child relation is represented by 
    the enclosure in this chart type.

    Mathematically, a sunburst chart visualizes hierarchical relationships 
    using nested rings, where each ring represents a level in the hierarchy:

    .. math::

        R_{n} \quad \text{for each level} \quad n

    The angle of each segment is proportional to its value, allowing for 
    comparison across different segments and levels.

    See Also
    --------
    gofast.tools.mathex.compute_sunburst_data : 
        Computes the data structure required for generating a sunburst chart 
        from a DataFrame.

    References
    ----------
    .. [1] Plotly Technologies Inc. (2021). Plotly Python Graphing Library. 
       Retrieved from https://plotly.com/python/
    
    """
    import plotly.graph_objects as go
    # Convert figsize from inches to pixels
    width_px, height_px = figsize[0] * 100, figsize[1] * 100

    fig = go.Figure(go.Sunburst(
        labels=[item['name'] for item in d],
        parents=[item['parent'] for item in d],
        values=[item['value'] for item in d],
        marker=dict(colors=color) if color else None
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),
                      width=width_px, height=height_px)

    if title:
        fig.update_layout(title=title)

    if savefig:
        if savefig.endswith('.html'):
            fig.write_html(savefig, auto_open=True)
        else:
            fig.write_image(savefig)

    fig.show()

    return fig

def plot_learning_inspections (
    models:List[object] , 
    X:NDArray, y:ArrayLike,  
    fig_size:Tuple[int] = ( 22, 18 ) , 
    cv: int = None, 
    savefig:Optional[str] = None, 
    titles = None, 
    subplot_kws =None, 
    **kws 
  ): 
    """ Inspect multiple models from their learning curves. 
    
    Mutiples Inspection plots that generate the test and training learning 
    curve, the training  samples vs fit times curve, the fit times 
    vs score curve for each model.  
    
    Parameters
    ----------
    models : list of estimator instances
        Each estimator instance implements `fit` and `predict` methods which
        will be cloned for each validation.
    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer Sckikit-learn :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    titles: str, list 
       List of model names if changes are needed. If ``None``, model names 
       are used by default. 
    kws: dict, 
        Additional keywords argument passed to :func:`plot_learning_inspection`. 
        
    Returns
    ----------
    axes: Matplotlib axes
    
    See also 
    ---------
    plot_learning_inspection:  Inspect single model 
    
    Examples 
    ---------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models.premodels import p 
    >>> from gofast.plot.evaluate  import plot_learning_inspections 
    >>> # import sparse  matrix from Bagoue dataset 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the two pretrained models from SVM 
    >>> models = [p.SVM.rbf.best_estimator_ , p.SVM.poly.best_estimator_]
    >>> plot_learning_inspections (models , X, y, ylim=(0.7, 1.01) )
    
    """
    
    models = is_iterable(models, exclude_string= True, transform =True )
    titles = list(is_iterable( 
        titles , exclude_string= True, transform =True )) 

    if len(titles ) != len(models): 
        titles = titles + [None for i in range (len(models)- len(titles))]
    # set the cross-validation to 4 
    cv = cv or 4  
    #set figure and subplots 
    if len(models)==1:
        msg = ( f"{plot_learning_inspection.__module__}."
               f"{plot_learning_inspection.__qualname__}"
               ) 
        raise PlotError ("For a single model inspection, use the"
                         f" function {msg!r} instead."
                         )
        
    fig , axes = plt.subplots (3 , len(models), figsize = fig_size )
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1, hspace = .5 )
    
    fig.subplots_adjust(**subplot_kws)
    
    if not is_iterable( axes) :
        axes =[axes ] 
    for kk, model in enumerate ( models ) : 
        title = titles[kk] or  get_estimator_name (model )
        plot_learning_inspection(model, X=X , y=y, axes = axes [:, kk], 
                               title =title, 
                               **kws)
        
    if savefig : 
        fig.savefig (savefig , dpi = 300 )
    plt.show () if savefig is None else plt.close () 
   
def plot_learning_inspection(
    model: BaseEstimator,  
    X: Union[ArrayLike, Any],  
    y: Union[ArrayLike, Any], 
    axes: Optional[ArrayLike] = None, 
    ylim: Optional[Tuple[float, float]] = None, 
    cv: int = 5, 
    n_jobs: Optional[int] = None,
    train_sizes: Optional[np.ndarray] = None, 
    display_legend: bool = True, 
    title: Optional[str] = None
) -> np.ndarray:
    """
    Inspect model from its learning curve.
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    model : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str, optional
        Title for the chart. If None, uses the estimator's name.

    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to `X` for classification or regression; None for 
        unsupervised learning.

    axes : array-like of shape (3,), optional
        Axes to use for plotting the curves. Default is None, which creates
        new axes.

    ylim : tuple of shape (2,), optional
        Defines minimum and maximum y-values plotted, e.g., (ymin, ymax). 
        Default is None.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Default is 5-fold
        cross-validation.

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if `y` is binary or multiclass,
        :class:`StratifiedKFold` is used. If the estimator is not a classifier
        or if `y` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various 
        cross-validators that can be used here.

    n_jobs : int or None, optional
        Number of jobs to run in parallel. `None` means 1 unless in a 
        :obj:`joblib.parallel_backend` context. `-1` means using all 
        processors.

    train_sizes : array-like of shape (n_ticks,), optional
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the `dtype` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e., it has to be 
        within (0, 1]. Otherwise, it is interpreted as absolute sizes of the 
        training sets. Note that for classification the number of samples 
        usually have to be big enough to contain at least one sample from each 
        class. Default is np.linspace(0.1, 1.0, 5).

    display_legend : bool, optional
        Whether to display the legend. Default is True.

    Returns
    -------
    axes : np.ndarray
        Matplotlib axes array with the plotted curves.

    Examples
    --------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models import p 
    >>> from gofast.plot.inspection import plot_learning_inspection 
    >>> X, y = fetch_data('bagoue prepared') 
    >>> plot_learning_inspection(p.SVM.rbf.best_estimator_, X, y)

    Notes
    -----
    This function generates three plots to inspect the model's performance:
    the learning curve, the scalability curve, and the performance curve.

    The learning curve shows the training and cross-validation scores as 
    functions of the number of training samples.

    The scalability curve shows the fit times as functions of the number of 
    training samples.

    The performance curve shows the cross-validation scores as functions of 
    the fit times.

    Mathematically, the learning curve represents the relationship between 
    the training score :math:`S_{train}` and the cross-validation score 
    :math:`S_{cv}` against the training set size :math:`N`.

    .. math::

        S_{train}(N), \quad S_{cv}(N)

    The scalability curve represents the relationship between the fit time 
    :math:`T` and the training set size :math:`N`.

    .. math::

        T(N)

    The performance curve represents the relationship between the 
    cross-validation score :math:`S_{cv}` and the fit time :math:`T`.

    .. math::

        S_{cv}(T)

    See Also
    --------
    sklearn.model_selection.learning_curve : Computes learning curves for a model.

    References
    ----------
    .. [1] Scikit-learn: Machine Learning in Python - 
       https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html

    """
    train_sizes = train_sizes or np.linspace(0.1, 1.0, 5)
    
    X, y = check_X_y(
        X, 
        y, 
        accept_sparse= True,
        to_frame =True 
        )
    
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    axes[0].set_title(title or get_estimator_name(model))
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].hlines(
        np.mean([train_scores[-1], test_scores[-1]]), 
        train_sizes[0],
        train_sizes[-1], 
        color="gray", 
        linestyle ="--", 
        label="Convergence score"
        )
    axes[0].plot(
        train_sizes, 
        train_scores_mean, 
        "o-", 
        color="r", 
        label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, 
        "o-", 
        color="g", 
        label="Cross-validation score"
    )

    if display_legend:
        axes[0].legend(loc="best")

    # set title name
    title_name = ( 
        f"{'the model'if title else get_estimator_name(model)}"
        )
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title(f"Scalability of {title_name}")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, 
                 test_scores_mean_sorted, "o-"
                 )
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title(f"Performance of {title_name}")

    return axes

@Dataify(auto_columns= True)
@ensure_pkg("mlxtend", extra=(
    "Can't plot heatmap using 'mlxtend' package."),  
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
 )
def plot_heatmapx(
    data: DataFrame, 
    columns: Optional[List[str]] = None, 
    savefig: Optional[str] = None, 
    **kws: Any
) -> Any:
    """
    Plot correlation matrix array as a heat map.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing the data.
        
        - `data` should have numerical columns for correlation computation.

    columns : Optional[List[str]], optional
        List of features. If given, only the dataframe with those features 
        is considered. Default is None.
        
        - `columns` specifies the subset of DataFrame columns to consider 
          for the correlation matrix. If None, all columns are used.
          
    savefig : Optional[str], optional
        Path and filename where the figure should be saved. If None, the figure 
        is shown using plt.show(). Default is None.
        
        - `savefig` determines the file path and format for saving the plot. 
          If provided, the plot is saved instead of shown.
          
    kws : Any
        Additional keyword arguments passed to `mlxtend.plotting.heatmap`.
        
        - `kws` allows for customization of the heatmap plot via keyword 
          arguments accepted by `mlxtend.plotting.heatmap`.

    Returns
    -------
    Any
        The `mlxtend.plotting.heatmap` axes object.

    Examples
    --------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_heatmapx
    >>> h = load_hlogs()
    >>> features = ['gamma_gamma', 'sp', 'natural_gamma', 'resistivity']
    >>> plot_heatmapx(h.frame, columns=features, cmap='PuOr')

    Notes
    -----
    This function computes the correlation matrix of the specified columns 
    in the DataFrame and visualizes it as a heat map using `mlxtend.plotting.heatmap`.
    
    The correlation matrix :math:`C` is defined as:

    .. math::

        C_{ij} = \frac{\text{cov}(X_i, X_j)}{\sigma_{X_i} \sigma_{X_j}}

    where :math:`\text{cov}(X_i, X_j)` is the covariance of variables :math:`X_i` 
    and :math:`X_j`, and :math:`\sigma_{X_i}` and :math:`\sigma_{X_j}` are the 
    standard deviations of :math:`X_i` and :math:`X_j`, respectively.

    See Also
    --------
    mlxtend.plotting.heatmap : Function to plot a heatmap of a correlation matrix.

    References
    ----------
    .. [1] Raschka, S. (2018). mlxtend: Providing machine learning and data science 
       utilities and extensions to Python's scientific computing stack. 
       The Journal of Open Source Software, 3(24), 638, 
       https://doi.org/10.21105/joss.00638
    """
    from mlxtend.plotting import  heatmap 

    cm = np.corrcoef(data.values.T)
    ax = heatmap(cm, row_names=columns, column_names=columns, **kws)
    
    if savefig is not None:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    
    return ax

@Dataify(auto_columns= True)
@ensure_pkg("mlxtend", extra=(
    "Can't plot the scatter matrix using 'mlxtend' package."), 
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
    )
def plot_matrix(
    data: DataFrame, 
    columns: Optional[List[str]] = None, 
    fig_size: Tuple[int, int] = (10, 8), 
    alpha: float = 0.5, 
    savefig: Optional[str] = None
) -> Any:
    """
    Visualize the pairwise correlation between different features in the dataset 
    in one place.

    Parameters
    ----------
    data : pd.DataFrame
        The pandas DataFrame containing the data.
        
        - `data` should have numerical columns for plotting the scatterplot matrix.

    columns : Optional[List[str]], optional
        List of features. If given, only the DataFrame with those features 
        is considered. Default is None.
        
        - `columns` specifies the subset of DataFrame columns to consider 
          for the scatterplot matrix. If None, all columns are used.
          
    fig_size : Tuple[int, int], optional
        Size of the displayed figure (width, height). Default is (10, 8).
        
        - `fig_size` determines the size of the generated plot in inches.
        
    alpha : float, optional
        Figure transparency. Default is 0.5.
        
        - `alpha` sets the transparency level for the scatterplot points.

    savefig : Optional[str], optional
        Path and filename where the figure should be saved. If None, the figure 
        is shown using plt.show(). Default is None.
        
        - `savefig` determines the file path and format for saving the plot. 
          If provided, the plot is saved instead of shown.

    Returns
    -------
    Any
        The `mlxtend.plotting.scatterplotmatrix` axes object.

    Examples
    --------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_matrix
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> h = load_hlogs()
    >>> features = ['gamma_gamma', 'natural_gamma', 'resistivity']
    >>> data = pd.DataFrame(np.log10(h.frame[features]), columns=features)
    >>> plot_matrix(data, columns=features)

    Notes
    -----
    This function creates a scatterplot matrix to visualize pairwise 
    relationships between features in a dataset. The scatterplot matrix 
    displays each feature against every other feature in a grid of subplots, 
    helping to identify potential correlations or patterns.

    Mathematically, the scatterplot matrix visualizes the relationship 
    between each pair of features :math:`X_i` and :math:`X_j` by plotting:

    .. math::

        (X_i, X_j) \quad \forall i, j \in \{1, 2, \ldots, n\}

    where :math:`n` is the number of features.

    See Also
    --------
    mlxtend.plotting.scatterplotmatrix : Function to plot a scatterplot matrix.

    References
    ----------
    .. [1] Raschka, S. (2018). mlxtend: Providing machine learning and data science 
       utilities and extensions to Python's scientific computing stack. 
       The Journal of Open Source Software, 3(24), 638, 
       https://doi.org/10.21105/joss.00638
    """

    from mlxtend.plotting import scatterplotmatrix
                                       
    if isinstance (columns, str): 
        columns = [columns ] 
    
    columns =list(columns)
        
    ax = scatterplotmatrix (
        data[columns].values , figsize =fig_size,names =columns , alpha =alpha 
        )
    plt.tight_layout()

    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 


def plot_loc_projection(
    X: Union[pd.DataFrame, np.ndarray], 
    Xt: Optional[Union[pd.DataFrame, np.ndarray]] = None, *, 
    columns: Optional[List[str]] = None, 
    test_kws: Optional[Dict] = None,  
    **baseplot_kws: Dict
) -> None:
    """
    Visualize train and test dataset based on the geographical coordinates.

    Since there is geographical information (latitude/longitude or
    easting/northing), it is a good idea to create a scatterplot of 
    all instances to visualize data.

    Parameters 
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample.

    Xt : Optional[Union[pd.DataFrame, np.ndarray]], optional
        Test set; data that is observed at testing and prediction time, used 
        as independent variables in learning. Default is None.

    columns : Optional[List[str]], optional
        Useful when a DataFrame is given with a dimension size greater than 2. 
        If such data is passed to `X` or `Xt`, `columns` must hold the names 
        to be considered as 'easting', 'northing' when UTM coordinates are 
        given or 'latitude', 'longitude' when latlon are given. If dimension 
        size is greater than 2 and `columns` is None, an error will be raised 
        to prompt the user to provide the index for 'y' and 'x' coordinate 
        retrieval. Default is None.

    test_kws : Optional[Dict], optional
        Keyword arguments passed to `matplotlib.plot.scatter` as test 
        location font and colors properties. Default is None.

    baseplot_kws : Dict
        All the keyword arguments passed to the properties of 
        `gofast.property.BasePlot` class.

    Returns
    -------
    None

    Examples
    --------
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.inspection import plot_loc_projection 
    >>> from gofast.utils import to_numeric_dtypes, naive_imputer
    >>> X, Xt, *_ = fetch_data('bagoue', split_X_y=True, as_frame=True) 
    >>> X = to_numeric_dtypes(X, pop_cat_features=True)
    >>> X = naive_imputer(X)
    >>> Xt = to_numeric_dtypes(Xt, pop_cat_features=True)
    >>> Xt = naive_imputer(Xt)
    >>> plot_kws = dict(fig_size=(8, 12),
                        lc='k',
                        marker='o',
                        lw=3.,
                        font_size=15.,
                        xlabel='easting (m)',
                        ylabel='northing (m)',
                        markerfacecolor='k', 
                        markeredgecolor='r',
                        alpha=1., 
                        markeredgewidth=2., 
                        show_grid=True,
                        galpha=0.2, 
                        glw=0.5, 
                        rotate_xlabel=90.,
                        fs=3.,
                        s=None)
    >>> plot_loc_projection(X, Xt, columns=['east', 'north'], 
                            trainlabel='train location', 
                            testlabel='test location', **plot_kws)

    Notes
    -----
    This function creates a scatter plot of geographical coordinates to visualize 
    the locations of training and testing datasets. It uses the provided 
    geographical information to plot the data points.

    The function supports both latitude/longitude and easting/northing coordinates.

    Mathematically, the function plots points :math:`(x, y)` for training data 
    and :math:`(x_t, y_t)` for testing data.

    .. math::

        \{(x_i, y_i)\} \quad \text{for training data}

        \{(x_{t_i}, y_{t_i})\} \quad \text{for testing data}

    See Also
    --------
    matplotlib.pyplot.scatter : Scatter plot in matplotlib.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.

    """
    trainlabel =baseplot_kws.pop ('trainlabel', None )
    testlabel =baseplot_kws.pop ('testlabel', None  )
    
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
    
    #check array
    X=check_array (
        X, 
        input_name="X", 
        to_frame =True, 
        )
    Xt =check_array (
        Xt, 
        input_name="Xt", 
        to_frame =True, 
        )
    # validate the projections.
    xy , xynames = projection_validator(X, Xt, columns )
    x, y , xt, yt =xy 
    xname, yname, xtarget_name, yname=xynames 

    pobj.xlim =[np.ceil(min(x)), np.floor(max(x))]
    pobj.ylim =[np.ceil(min(y)), np.floor(max(y))]   
    
    xpad = abs((x -x.mean()).min())/5.
    ypad = abs((y -y.mean()).min())/5.
 
    if  Xt is not None: 

        min_x, max_x = xt.min(), xt.max()
        min_y, max_y = yt.min(), yt.max()
        
        
        pobj.xlim = [min([pobj.xlim[0], np.floor(min_x)]),
                     max([pobj.xlim[1], np.ceil(max_x)])]
        pobj.ylim = [min([pobj.ylim[0], np.floor(min_y)]),
                     max([pobj.ylim[1], np.ceil(max_y)])]
      
    pobj.xlim =[pobj.xlim[0] - xpad, pobj.xlim[1] +xpad]
    pobj.ylim =[pobj.ylim[0] - ypad, pobj.ylim[1] +ypad]
    
     # create figure obj 
    fig = plt.figure(figsize = pobj.fig_size)
    ax = fig.add_subplot(1,1,1)
    
    xname = pobj.xlabel or xname 
    yname = pobj.ylabel or yname 
    
    if pobj.s is None: 
        pobj.s = pobj.fs *40 
    ax.scatter(x, y, 
               color = pobj.lc,
                s = pobj.s if not pobj.s else pobj.fs * pobj.s, 
                alpha = pobj.alpha , 
                marker = pobj.marker,
                edgecolors = pobj.marker_edgecolor,
                linewidths = pobj.lw,
                linestyles = pobj.ls,
                facecolors = pobj.marker_facecolor,
                label = trainlabel 
            )
    
    if  Xt is not None:
        if pobj.s is not None: 
            pobj.s /=2 
        test_kws = test_kws or dict (
            color = 'r',s = pobj.s, alpha = pobj.alpha , 
            marker = pobj.marker, edgecolors = 'r',
            linewidths = pobj.lw, linestyles = pobj.ls,
            facecolors = 'k'
            )
        ax.scatter(xt, yt, 
                    label = testlabel, 
                    **test_kws
                    )

    ax.set_xlim (pobj.xlim)
    ax.set_ylim (pobj.ylim)
    ax.set_xlabel( xname,
                  fontsize= pobj.font_size )
    ax.set_ylabel (yname,
                   fontsize= pobj.font_size )
    ax.tick_params(axis='both', 
                   labelsize= pobj.font_size )
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)
    
    if pobj.show_grid is True : 
        ax.grid(pobj.show_grid,
                axis=pobj.gaxis,
                which = pobj.gwhich, 
                color = pobj.gc,
                linestyle=pobj.gls,
                linewidth=pobj.glw, 
                alpha = pobj.galpha
                )
        if pobj.gwhich =='minor': 
            ax.minorticks_on()
            
    if len(pobj.leg_kws) ==0 or 'loc' not in pobj.leg_kws.keys():
         pobj.leg_kws['loc']='upper left'
    ax.legend(**pobj.leg_kws)
    pobj.save(fig)

def plot_matshow(
    arr: np.ndarray, /, 
    labelx: Optional[List[str]] = None, 
    labely: Optional[List[str]] = None, 
    matshow_kws: Optional[Dict[str, Any]] = None, 
    **baseplot_kws: Any
) -> None:
    
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    arr= check_array(
        arr, 
        to_frame =True, 
        input_name="Array 'arr'"
        )
    matshow_kws= matshow_kws or dict()
    fig = plt.figure(figsize = pobj.fig_size)

    ax = fig.add_subplot(1,1,1)
    
    cax = ax.matshow(arr, **matshow_kws) 
    cbax= fig.colorbar(cax, **pobj.cb_props)
    
    if pobj.cb_label is None: 
        pobj.cb_label=''
    ax.set_xlabel( pobj.xlabel,
          fontsize= pobj.font_size )
    
    # for label in zip ([labelx, labely]): 
    #     if label is not None:
    #         if not is_iterable(label):
    #             label = [label]
    #         if len(label) !=arr.shape[1]: 
    #             warnings.warn(
    #                 "labels and arr dimensions must be consistent"
    #                 f" Expect {arr.shape[1]}, got {len(label)}. "
    #                 )
                #continue
    if labelx is not None: 
        ax = _check_labelxy (labelx , arr, ax )
    if labely is not None: 
        ax = _check_labelxy (labely, arr, ax , axis ='y')
    
    if pobj.ylabel is None:
        pobj.ylabel =''
    if pobj.xlabel is None:
        pobj.xlabel = ''
    
    ax.set_ylabel (pobj.ylabel,
                   fontsize= pobj.font_size )
    ax.tick_params(axis=pobj.tp_axis, 
                    labelsize= pobj.font_size, 
                    bottom=pobj.tp_bottom, 
                    top=pobj.tp_top, 
                    labelbottom=pobj.tp_labelbottom, 
                    labeltop=pobj.tp_labeltop
                    )
    if pobj.tp_labeltop: 
        ax.xaxis.set_label_position('top')
    
    cbax.ax.tick_params(labelsize=pobj.font_size ) 
    cbax.set_label(label=pobj.cb_label,
                   size=pobj.font_size,
                   weight=pobj.font_weight)
    
    plt.xticks(rotation = pobj.rotate_xlabel)
    plt.yticks(rotation = pobj.rotate_ylabel)

    pobj.save(fig)

plot_matshow.__doc__ = """\
Quick matrix visualization using `matplotlib.pyplot.matshow`.

Parameters
----------
arr : np.ndarray
    2D array of shape (n_rows, n_cols).
    The matrix to be visualized.

labelx : Optional[List[str]], optional
    List of labels for the x-axis categories. Must match the number of 
    columns in `arr`.

labely : Optional[List[str]], optional
    List of labels for the y-axis categories. Must match the number of 
    rows in `arr`.

matshow_kws : Optional[Dict[str, Any]], optional
    Additional keyword arguments for `matplotlib.axes.Axes.matshow`.

baseplot_kws : Dict[str, Any]
    Additional keyword arguments for base plot properties, such as:
    
    - `fig_size` : Tuple[int, int], size of the figure.
    - `font_size` : float, font size for labels and ticks.
    - `cb_label` : str, label for the colorbar.
    - `xlabel` : str, label for the x-axis.
    - `ylabel` : str, label for the y-axis.
    - `rotate_xlabel` : float, rotation angle for x-axis labels.
    - `rotate_ylabel` : float, rotation angle for y-axis labels.
    - `tp_axis` : str, axis for tick parameters ('x', 'y', or 'both').
    - `tp_labeltop` : bool, whether to display x-axis labels on top.
    - `tp_labelbottom` : bool, whether to display x-axis labels on bottom.
    - `tp_bottom` : bool, whether to display x-axis ticks on bottom.
    - `tp_top` : bool, whether to display x-axis ticks on top.
    - `cb_props` : Dict[str, Any], properties for colorbar.

Returns
-------
None

Examples
--------
>>> import numpy as np
>>> from gofast.plot.inspection import plot_matshow 
>>> matshow_kwargs = {
...     'aspect': 'auto',
...     'interpolation': None,
...     'cmap': 'copper_r',
... }
>>> baseplot_kws = {
...     'fig_size': (10, 8),
...     'font_size': 15.,
...     'xlabel': 'Predicted flow classes',
...     'ylabel': 'Geological rocks',
...     'rotate_xlabel': 45.,
...     'rotate_ylabel': 45.,
...     'tp_labelbottom': False,
...     'tp_labeltop': True,
... }
>>> labelx = ['FR0', 'FR1', 'FR2', 'FR3', 'Rates'] 
>>> labely = ['VOLCANO-SEDIM. SCHISTS', 'GEOSYN. GRANITES', 
...           'GRANITES', '1.0', 'Rates']
>>> array2d = np.array([
...     [1., .5, 1., 1., .9286], 
...     [.5, .8, 1., .667, .7692],
...     [.7, .81, .7, .5, .7442],
...     [.667, .75, 1., .75, .82],
...     [.9091, .8064, .7, .8667, .7931]
... ])
>>> plot_matshow(array2d, labelx, labely, matshow_kwargs, **baseplot_kws)

Notes
-----
This function provides a quick visualization of a matrix using a heatmap.

The function takes a 2D array `arr` and visualizes it as a heatmap. The 
labels for the x-axis and y-axis can be specified using `labelx` and `labely`.

Mathematically, the heatmap visualizes the matrix elements :math:`A_{ij}`:

.. math::

    A = \\begin{bmatrix}
    a_{11} & a_{12} & \\cdots & a_{1n} \\
    a_{21} & a_{22} & \\cdots & a_{2n} \\
    \\vdots & \\vdots & \\ddots & \\vdots \\
    a_{m1} & a_{m2} & \\cdots & a_{mn}
    \\end{bmatrix}

See Also
--------
matplotlib.pyplot.matshow : Visualize a matrix in a new figure window.

References
----------
.. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
   Computing in Science & Engineering, 9(3), 90-95.
"""

  
def _check_labelxy (lablist, ar, ax, axis = 'x' ): 
    """ Assert whether the x and y labels given for setting the ticklabels 
    are consistent. 
    
    If consistent, function set x or y labels along the x or y axis 
    of the given array. 
    
    :param lablist: list, list of the label to set along x/y axis 
    :param ar: arraylike 2d, array to set x/y axis labels 
    :param ax: matplotlib.pyplot.Axes, 
    :param axis: str, default="x", kind of axis to set the label. 
    
    """
    warn_msg = ("labels along axis {axis} and arr dimensions must be"
                " consistent. Expects {shape}, got {len_label}")
    ax_ticks, ax_labels  = (ax.set_xticks, ax.set_xticklabels 
                         ) if axis =='x' else (
                             ax.set_yticks, ax.set_yticklabels )
    if lablist is not None: 
        lablist = is_iterable(lablist, exclude_string=True, 
                              transform =True )
        if not _check_consistency_size (
                lablist , ar[0 if axis =='x' else 1], error ='ignore'): 
            warnings.warn(warn_msg.format(
                axis = axis , shape=ar.shape[0 if axis =='x' else 1], 
                len_label=len(lablist))
                )
        else:
            ax_ticks(np.arange(0, ar.shape[0 if axis =='x' else 1]))
            ax_labels(lablist)
        
    return ax         
   
def plot_set_matrix(
    data: Dict[str, set], 
    cmap: str = 'Blues', 
    savefig: Optional[str] = None
    ) -> plt.Axes:
    """
    Creates and optionally saves a matrix-based representation to visualize 
    intersections among multiple sets.

    Rows represent elements and columns represent sets. Cells in the matrix 
    are filled to indicate membership.

    Parameters
    ----------
    data : Dict[str, set]
        A dictionary where each key is the name of a set and each value is 
        the set itself.
        
        - `data` contains the sets to be visualized in the matrix.
        
    cmap : str, optional
        The colormap used for the matrix plot. Defaults to 'Blues'.
        
        - `cmap` sets the color scheme for the matrix visualization.
        
    savefig : Optional[str], optional
        The file path and name to save the figure. If None, the figure is 
        not saved.
        
        - `savefig` specifies the file path to save the plot image. If not 
          provided, the plot is displayed without saving.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot.

    Example
    -------
    >>> from gofast.plot.inspection import plot_set_matrix
    >>> sets = {
    ...     "Set1": {1, 2, 3},
    ...     "Set2": {2, 3, 4},
    ...     "Set3": {3, 4, 5}
    ... }
    >>> ax = plot_set_matrix(sets, cmap='Greens', savefig='set_matrix.png')

    This will create a matrix plot for three sets, showing which elements 
    belong to which sets and optionally save it to 'set_matrix.png'.

    Notes
    -----
    This function generates a visual representation of set intersections using 
    a matrix format. The matrix allows easy identification of elements belonging 
    to multiple sets.

    The matrix is constructed as follows:

    .. math::

        M_{ij} = \\begin{cases}
        1 & \\text{if element } i \\text{ is in set } j \\\\
        0 & \\text{otherwise}
        \\end{cases}

    where :math:`M_{ij}` represents the membership of element :math:`i` in set 
    :math:`j`.

    See Also
    --------
    matplotlib.pyplot.imshow : Display data as an image, i.e., on a 2D regular raster.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.
    """
    data = validate_sets(data, mode='deep', allow_empty= False )
    # Create a DataFrame to represent the sets
    all_elements = sorted(set.union(*data.values()))
    matrix_data = pd.DataFrame(index=all_elements, columns=data.keys(), data=0)

    # Fill the DataFrame
    for set_name, elements in data.items():
        matrix_data.loc[list(elements), set_name] = 1  # Convert set to list

    # Create and display the matrix plot
    fig, ax = plt.subplots(figsize=(len(data), len(all_elements)/2))
    cax = ax.imshow(matrix_data, aspect='auto', cmap=cmap, interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Set Membership')
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.keys())
    ax.set_yticks(np.arange(len(all_elements)))
    ax.set_yticklabels(all_elements)
    ax.set_title('Matrix-Based Representation of Set Intersections')
    ax.set_xlabel('Sets')
    ax.set_ylabel('Elements')
    ax.grid(False)

    # Save the figure if a path is provided
    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return ax
     



