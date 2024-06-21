# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import os 
import warnings 
import numpy as np 
import pandas as pd 
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import learning_curve 

from ..api.types import NDArray, ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Dict, Union 
from ..api.util import to_snake_case 
from ..decorators import Dataify
from ..exceptions import PlotError 
from ..tools.coreutils import is_iterable, to_numeric_dtypes 
from ..tools.coreutils import _assert_all_types, projection_validator
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import check_X_y, check_array, validate_square_matrix 
from ..tools.validator import get_estimator_name, _check_consistency_size 
from ..tools.validator import parameter_validator
from ._config import PlotConfig 
from .utils import _manage_plot_kws, pobj, savefigure

__all__=[
    'plot_learning_inspection',
    'plot_learning_inspections',
    'plot_loc_projection',
    'plot_matshow',
    'plot_mlxtend_heatmap',
    'plot_mlxtend_matrix',
    'plot_sunburst',
    'plot_sankey',
    'plot_euler_diagram',
    'create_upset_plot',
    'plot_venn_diagram',
    'create_matrix_representation',
    'woodland_plot', 
    'plot_l_curve', 
    ]


@Dataify(auto_columns=True, prefix="feature_")  
def woodland_plot(
    data: DataFrame,*,
    quadrant: str="upper_left",
    compute_corr: bool=False,
    method: str='pearson',
    annot: bool=True,
    fig_size: Tuple [int, int]=(11, 9),
    fmt: str=".2f",
    linewidths: float=.5,
    xrot: int=45,
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
        Rotation angle in degrees for x-axis labels. Default is 45.
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
    >>> from gofast.plot.inspection import woodland_plot
    >>> df = pd.DataFrame(np.random.rand(10, 10), columns=list('ABCDEFGHIJ'))
    >>> woodland_plot(df, compute_corr=True, quadrant='upper_left', 
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
    rms, 
    roughness, 
    tau=None, 
    hansen_point=None, 
    rms_target=None,
    view_tline=False,
    hpoint_kws=dict(), 
    fig_size = (10, 4),
    ax=None,
    fig=None, 
    style = 'classic', 
    savefig=None, 
    **plot_kws
    ):
    """
    Plot the Hansen L-curve.
    
    The L-curve criteria is used to determine the suitable model 
    after runing multiple inversions with different :math:`\tau` values. 
    The function plots RMS vs. Roughness with an option to highlight a 
    specific point named Hansen point [1]_.
    
    The :math:`\tau` represents the measure of compromise between data fit and 
    model smoothness. To find out an appropriates-value, the inversion was 
    carried out with differents-values. The RMS error obtained from each 
    inversion is plotted against model roughnes [2]_. 
    
    Plots RMS vs. Roughness with an option to highlight the Hansen point.
    
    Parameters 
    ------------
    
    rms: ArrayLike, list, 
       Corresponding list pr Arraylike of RMS values.
       
    roughness: Arraylike, list, 
       List or ArrayLike of roughness values. 
       
    tau: Arraylike or list, optional 
       List of tau values to visualize as text mark in the plot. 

    hansen_point: A tuple (roughness_value, RMS_value) , optional 
       The Hansen point to visualize in the plot. It can be determine 
       automatically if ``highlight_point='auto'``.
       
    rms_target: float, optional 
      The root-mean-squared target. If set, and `view_tline` is ``False``, 
      the target value should be axis limit. 
      
     view_tline: bool, default=False 
       Display the target line should be  displayed.
       
    hpoint_kws: dict, optional 
      Keyword argument to highlight the hansen point in the figure. 
     
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure. Could be used to support other axes. 
       
    fig: Matplotlib.pyplot.figure, optional 
        Supply fig to save automatically the plot, otherwise, keep it 
        to ``None``.

    savefig: str, optional 
        Save figure name. The default resolution dot-per-inch is ``300``. 
         
    Return 
    ------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
        
    [2] Kouadio, K.L, Jianxin Liu, lui R., Liu W. Boukhalfa Z. (2024). An integrated
        Approach for sewage diversion: Case of Huayuan Mine. Geophysics, 89(4). 
        https://doi.org/10.1190/geo2023-0332.1
         
    Examples
    ---------
    >>> from gofast.tools.utils import plot_l_curve
    >>> # Test the function with the provided data points and 
    >>> # highlighting point (50, 3.12)
    >>> roughness_data = [0, 50, 100, 150, 200, 250, 300, 350]
    >>> RMS_data = [3.16, 3.12, 3.1, 3.08, 3.06, 3.04, 3.02, 3]
    >>> highlight_data = (50, 3.12)
    >>> plot_l_curve(roughness_data, RMS_data, highlight_data)
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
    sets: Union[Dict[str, set], List[ArrayLike]],
    set_labels: Optional[Tuple[str, ...]] = None,
    title: str = 'Venn Diagram',
    figsize: Tuple[int, int] = (8, 8),
    set_colors: Tuple[str, ...] = ('red', 'green', 'blue'),
    alpha: float = 0.5, 
    savefig: Optional[str]=None, 
) -> Axes:
    """
    Create and optionally save a Venn diagram for two sets.

    A Venn diagram is a visual representation of the mathematical or logical
    relationship between two sets of items. It depicts these sets as circles,
    with the overlap representing items common to both sets. It's a useful
    tool for comparing and contrasting groups of elements, especially in
    the field of data analysis and machine learning.
    
    This function supports up to 3 sets. Venn diagrams for more than 3 sets
    are not currently supported due to visualization complexity.

    Parameters
    ----------
    sets : Union[Dict[str, set], List[np.ndarray]]
        Either a dictionary with two items, each being a set of elements, or a 
        list of two arrays. The arrays should be of consistent length, 
        and their elements will be treated as unique identifiers.
    set_labels : Optional[Tuple[str, str]], optional
        A tuple containing two strings that label the sets in the diagram. If None, 
        default labels will be used.
    title : str, optional
        Title of the Venn diagram. Default is 'Venn Diagram'.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (8, 8).
    set_colors : Tuple[str, str], optional
        Colors for the two sets. Default is ('red', 'green').
    alpha : float, optional
        Transparency level of the set colors. Default is 0.5.
    savefig : Optional[str], optional
        Path and filename to save the figure. If None, the figure is not saved.
        Example: 'path/to/figure.png'

    Returns
    -------
    ax : Axes
        The axes object of the plot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xgboost import XGBClassifier
    >>> from gofast.plot.inspection import plot_venn_diagram
    
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
    """

    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
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
def plot_euler_diagram(sets: dict):
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
        A dictionary where keys are set names and values are the sets.

    Returns
    -------
    None. The function displays the Euler diagram.

    Example
    -------
    >>> from gofast.plot.inspection import plot_euler_diagram
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {3, 4, 5},
        "Set3": {5, 6, 7}
    }
    >>> plot_euler_diagram(sets)

    This will create an Euler diagram for three sets, showing their overlaps.
    Set1 contains elements 1, 2, and 3; Set2 contains 3, 4, and 5; and Set3 
    contains 5, 6, and 7. The diagram will visually represent the intersections 
    between these sets.

    Notes
    -----
    This function requires a working R environment with the 'venneuler' package 
    installed, as well as 'rpy2' installed in the Python environment. It creates 
    a temporary file to save the plot generated by R, reads this file into Python 
    for display, and then removes the file.
    """

    import rpy2.robjects as robjects
    # from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    import matplotlib.image as mpimg
    import tempfile
    
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
        data: Dict[str, set], sort_by: str = 'degree', 
        show_counts: bool = False):
    """
    Creates an UpSet plot, which is an alternative to Venn diagrams for more 
    than three sets.
    
    The UpSet plot focuses on the intersections of the sets rather than on the 
    sets themselves.

    Parameters
    ----------
    data : Dict[str, set]
        A dictionary where each key is the name of a set and each value is 
        the set itself.
    sort_by : str, optional
        The criteria to sort the bars. Options are 'degree' and 'cardinality'.
        'degree' will sort by the number of sets in the intersection, 
        'cardinality' by the size of the intersection.
    show_counts : bool, optional
        Whether to show the counts at the top of the bars in the UpSet plot.

    Returns
    -------
    None. The function displays the UpSet plot.

    Example
    -------
    >>> from gofast.plot.inspection import create_upset_plot
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {3, 4, 5},
        "Set3": {5, 6, 7},
        "Set4": {7, 8, 9}
    }
    >>> create_upset_plot(sets, sort_by='degree', show_counts=True)

    This will create an UpSet plot for four sets, showing their intersections 
    and the counts of elements in each intersection.
    """
    from upsetplot import plot, from_contents 
    # Convert the data into the format required by UpSetPlot
    upset_data = from_contents(data)

    # Create and display the UpSet plot
    plot(upset_data, sort_by=sort_by, show_counts=show_counts)
    
@ensure_pkg("plotly","plot_sunburst requires the 'plotly' package.", 
            auto_install=PlotConfig.install_dependencies ,
            use_conda=PlotConfig.use_conda 
    )
def plot_sankey(
    data: DataFrame , 
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
    source_col : str
        The name of the column in 'data' that contains the source nodes.
    target_col : str
        The name of the column in 'data' that contains the target nodes.
    value_col : str
        The name of the column in 'data' that contains the flow values
        between the nodes.
    label_col : Optional[str], optional
        The name of the column in 'data' that contains the labels of the nodes.
        If None, the nodes will be labeled with unique identifiers.
    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in pixels. Default is (800, 600).
    title : Optional[str], optional
        The title of the plot. If None, no title is set.

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
    
    See Also 
    ---------
    gofast.tools.mathex.infer_sankey_columns: 
        Infers source, target, and value columns for a Sankey diagram 
        from a DataFrame.
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
    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in pixels. Default is (1000, 800).
    color : List[str], optional
        A list of color codes for the segments of the sunburst chart. If None,
        default colors are used.
    title : str, optional
        The title of the plot. If None, no title is set.
    savefig : str, optional
        Path and filename where the figure should be saved. If ends with 'html',
        saves an interactive HTML file. Otherwise, saves a static image of the
        plot.

    Returns
    -------
    go.Figure
        The Plotly Figure object with the plot for further tweaking and rendering.

    See Also
    ---------
    gofast.tools.mathex.compute_sunburst_data: 
        Computes the data structure required for generating a sunburst chart 
        from a DataFrame.
        
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
    model,  
    X,  
    y, 
    axes=None, 
    ylim=None, 
    cv=5, 
    n_jobs=None,
    train_sizes=None, 
    display_legend = True, 
    title=None,
):
    """Inspect model from its learning curve. 
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    model : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

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

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    display_legend: bool, default ='True' 
        display the legend
        
    Returns
    ----------
    axes: Matplotlib axes 
    
    Examples 
    ----------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models import p 
    >>> from gofast.plot.evaluate  import plot_learning_inspection 
    >>> # import sparse  matrix from Bagoue datasets 
    >>> X, y = fetch_data ('bagoue prepared') 
    >>> # import the  pretrained Radial Basis Function (RBF) from SVM 
    >>> plot_learning_inspection (p.SVM.rbf.best_estimator_  , X, y )
    
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

@ensure_pkg("mlxtend", extra=(
    "Can't plot heatmap using 'mlxtend' package."),  
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
 )
def plot_mlxtend_heatmap (df, columns =None, savefig=None,  **kws): 
    """ Plot correlation matrix array  as a heat map 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param kws: additional keyword arguments passed to 
        :func:`mlxtend.plotting.heatmap`
    :return: :func:`mlxtend.plotting.heatmap` axes object 
    
    :example: 
        
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_mlxtend_heatmap
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'sp',
                'natural_gamma', 'resistivity']
    >>> plot_mlxtend_heatmap (h.frame , columns =features, cmap ='PuOr')
    """

    from mlxtend.plotting import  heatmap 
    cm = np.corrcoef(df[columns]. values.T)
    ax= heatmap(cm, row_names = columns , column_names = columns, **kws )
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

@ensure_pkg("mlxtend", extra=(
    "Can't plot the scatter matrix using 'mlxtend' package."), 
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
    )
def plot_mlxtend_matrix(df, columns =None, fig_size = (10 , 8 ),
                        alpha =.5, savefig=None  ):
    """ Visualize the pair wise correlation between the different features in  
    the dataset in one place. 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param fig_size: tuple of int (width, heigh) 
        Size of the displayed figure 
    :param alpha: figure transparency, default is ``.5``. 
    
    :return: :func:`mlxtend.plotting.scatterplotmatrix` axes object 
    :example: 
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_mlxtend_matrix
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'natural_gamma', 'resistivity']
    >>> data = pd.DataFrame ( np.log10 (h.frame[features]), columns =features )
    >>> plot_mlxtend_matrix (data, columns =features)

    """
    from mlxtend.plotting import scatterplotmatrix
                                       
    if isinstance (columns, str): 
        columns = [columns ] 
    try: 
        iter (columns)
    except : 
        raise TypeError(" Columns should be an iterable object, not"
                        f" {type (columns).__name__!r}")
    columns =list(columns)
    
    
    if columns is not None: 
        df =df[columns ] 
        
    ax = scatterplotmatrix (
        df[columns].values , figsize =fig_size,names =columns , alpha =alpha 
        )
    plt.tight_layout()

    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 
def plot_loc_projection(
    X: DataFrame | NDArray, 
    Xt: DataFrame | NDArray =None, *, 
    columns: List[str] =None, 
    test_kws: dict =None,  
    **baseplot_kws 
    ): 
    """ Visualize train and test dataset based on 
    the geographical coordinates.
    
    Since there is geographical information(latitude/longitude or
    easting/northing), it is a good idea to create a scatterplot of 
    all instances to visualize data.
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to considered as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
        
    test_kws: dict, 
        keywords arguments passed to :func:`matplotlib.plot.scatter` as test
        location font and colors properties. 
        
    baseplot_kws: dict, 
        All all  the keywords arguments passed to the peroperty  
        :class:`gofast.property.BasePlot` class. 
        
    Examples
    --------
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.evaluate  import plot_loc_projection 
    >>> # Discard all the non-numeric data 
    >>> # then inut numerical data 
    >>> from gofast.utils import to_numeric_dtypes, naive_imputer
    >>> X, Xt, *_ = fetch_data ('bagoue', split_X_y =True, as_frame =True) 
    >>> X =to_numeric_dtypes(X, pop_cat_features=True )
    >>> X= naive_imputer(X)
    >>> Xt = to_numeric_dtypes(Xt, pop_cat_features=True )
    >>> Xt= naive_imputer(Xt)
    >>> plot_kws = dict (fig_size=(8, 12),
                     lc='k',
                     marker='o',
                     lw =3.,
                     font_size=15.,
                     xlabel= 'easting (m) ',
                     ylabel='northing (m)' , 
                     markerfacecolor ='k', 
                     markeredgecolor='r',
                     alpha =1., 
                     markeredgewidth=2., 
                     show_grid =True,
                     galpha =0.2, 
                     glw=.5, 
                     rotate_xlabel =90.,
                     fs =3.,
                     s =None )
    >>> plot_loc_projection( X, Xt , columns= ['east', 'north'], 
                        trainlabel='train location', 
                        testlabel='test location', **plot_kws
                       )
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
    arr, / , labelx:List[str] =None, labely:List[str]=None, 
    matshow_kws=None, **baseplot_kws
    ): 
    
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

plot_matshow.__doc__ ="""\
Quick matrix visualization using matplotlib.pyplot.matshow.

Parameters
----------
arr: 2D ndarray, 
    matrix of n rowns and m-columns items 
matshow_kws: dict
    Additional keywords arguments for :func:`matplotlib.axes.matshow`
    
labelx: list of str, optional 
        list of labels names that express the name of each category on 
        x-axis. It might be consistent with the matrix number of 
        columns of `arr`. 
        
label: list of str, optional 
        list of labels names that express the name of each category on 
        y-axis. It might be consistent with the matrix number of 
        row of `arr`.
    
Examples
---------
>>> import numpy as np
>>> from gofast.plot.evaluate  import plot_matshow 
>>> matshow_kwargs ={
    'aspect': 'auto',
    'interpolation': None,
   'cmap':'copper_r', 
        }
>>> baseplot_kws ={'lw':3, 
           'lc':(.9, 0, .8), 
           'font_size':15., 
            'cb_format':None,
            #'cb_label':'Rate of prediction',
            'xlabel': 'Predicted flow classes',
            'ylabel': 'Geological rocks',
            'font_weight':None,
            'tp_labelbottom':False,
            'tp_labeltop':True,
            'tp_bottom': False
            }
>>> labelx =['FR0', 'FR1', 'FR2', 'FR3', 'Rates'] 
>>> labely =['VOLCANO-SEDIM. SCHISTS', 'GEOSYN. GRANITES', 
             'GRANITES', '1.0', 'Rates']
>>> array2d = np.array([(1. , .5, 1. ,1., .9286), 
                    (.5,  .8, 1., .667, .7692),
                    (.7, .81, .7, .5, .7442),
                    (.667, .75, 1., .75, .82),
                    (.9091, 0.8064, .7, .8667, .7931)])
>>> plot_matshow(array2d, labelx, labely, matshow_kwargs,**baseplot_kws )  

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
        
def create_matrix_representation(
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
    cmap : str, optional
        The colormap used for the matrix plot. Defaults to 'Blues'.
    savefig : Optional[str], optional
        The file path and name to save the figure. If None, the figure is 
        not saved.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot.

    Example
    -------
    >>> from gofast.plot.inspection import create_matrix_representation
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {2, 3, 4},
        "Set3": {3, 4, 5}
    }
    >>> ax = create_matrix_representation(sets, cmap='Greens', savefig='set_matrix.png')

    This will create a matrix plot for three sets, showing which elements 
    belong to which sets and optionally save it to 'set_matrix.png'.
    """
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


