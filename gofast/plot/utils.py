# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Additional plot utilities. 
"""
from __future__ import annotations 
import os
import re 
import copy 
import datetime 
import warnings
import itertools 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.axes
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 

from sklearn.metrics import r2_score 
from sklearn.utils import resample

from ..api.types import Optional, Tuple,  Union 
from ..api.types import Dict, ArrayLike, DataFrame
from ..api.property import BasePlot
from ..tools.coreutils import _assert_all_types, is_iterable, str2columns 
from ..tools.coreutils import is_in_if
from ..tools.validator import  assert_xy_in
from ._d_cms import D_COLORS, D_MARKERS, D_STYLES

__all__=["boxplot", "plot_r_squared", "plot_text"]

class PlotUtils(BasePlot):
    def __init__(self, **kwargs):
        """
        Initialize the plotting utility class which extends the BasePlot class,
        allowing for custom plot configurations.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments that are passed to the
            BasePlot's constructor.
        """
        super().__init__(**kwargs)
    
    def save(self, fig):
        """
        Save the figure with the specified attributes if `savefig`
        is set; otherwise, display the figure on screen.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to be saved or displayed.

        Notes
        -----
        - If `savefig` is not None, the figure is saved using the
          path specified in `savefig`.
        - The figure is saved with the resolution specified by `fig_dpi`,
          and the orientation can be set with `fig_orientation`.
        - If `savefig` is None, the figure will be displayed using
          `plt.show()` and not saved.

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1], [0, 1])
        >>> plot_utils = PlotUtils(figsize=(10, 5), savefig='plot.png',
        ...                        fig_dpi=300, fig_orientation='landscape')
        >>> plot_utils.save(fig)  # This will save the figure to 'plot.png'
        """
        if self.savefig is not None:
            fig.savefig(self.savefig, dpi=self.fig_dpi,
                        bbox_inches='tight', 
                        orientation=self.fig_orientation)
            plt.close()
        else:
            plt.show()
    
def create_custom_plotter(base_plotter):
    """
    Dynamically creates a custom plotter class that includes additional 
    methods and attributes from an existing plotter object.

    Parameters:
        base_plotter: An instance of a plotter class from which to inherit 
        properties.

    Returns:
        A new plotting class with combined features from `PlotUtils and `base_plotter`.
    """
    # Creating a dynamic type combining BasePlot and properties from the given plotter
    plot_class = type('CustomPlotter', (PlotUtils,), {**base_plotter.__dict__})
    
    # Update the docstring for the new class
    plot_class.__doc__ = """\
    Custom plotting class that extends :class:`~gofast.api.properties.BasePlot` 
    with dynamic properties.

    Inherits all matplotlib figure properties, allowing modification via 
    object attributes. For example:
        >>> plot_obj = CustomPlotter()
        >>> plot_obj.ls = '-.'  # Set line style
        >>> plot_obj.fig_size = (7, 5)  # Set figure size
        >>> plot_obj.lw = 7  # Set linewidth

    See also:
        Refer to :class:`~gofast.api.property.BasePlot` for details 
        on adjustable parameters.
    """
    
    return plot_class

# #################################################
# Creating a dynamic type combining BasePlot and 
# properties from the given plotter
pobj = type('DynamicPlotter', (PlotUtils,), {})

pobj.__doc__ = """\
Dynamic plotting class that extends :class:`~gofast.api.properties.BasePlot` 
with dynamic properties.

Inherits all matplotlib figure properties, allowing modification via 
object attributes. For example:
    >>> plot_obj = DynamicPlotter()
    >>> plot_obj.ls = '-.'  # Set line style
    >>> plot_obj.fig_size = (7, 5)  # Set figure size
    >>> plot_obj.lw = 7  # Set linewidth

See also:
    Refer to :class:`~gofast.api.property.BasePlot` for details 
    on adjustable parameters.
"""
# ##################################################

def boxplot(
    data: ArrayLike | DataFrame, /, 
    labels: list[str],
    title: str, 
    y_label: str, 
    figsize: tuple[int, int]=(8, 8), 
    color: str="lightgreen", 
    showfliers: bool=True, 
    whis: float=1.5, 
    width: float=0.5, 
    linewidth: float=2, 
    flierprops: dict=None, 
    sns_style="whitegrid", 
   ) -> plt.Axes:
    """
    Plots a custom boxplot with the given data and parameters using 
    Seaborn and Matplotlib.

    Parameters
    ----------
    data : np.ndarray
        The input data for each category to plot in the boxplot, 
        organized as a list of arrays.
    labels : list[str]
        The labels for the boxplot categories.
    title : str
        The title of the plot.
    y_label : str
        The label for the Y-axis.
    figsize : tuple[int, int], optional
        Figure dimension (width, height) in inches. Default is (10, 8).
    color : str, optional
        Color for all of the elements, or seed for a gradient palette.
        Default is "lightgreen".
    showfliers : bool, optional
        If True, show the outliers beyond the caps. Default is True.
    whis : float, optional
        Proportion of the IQR past the low and high quartiles to 
        extend the plot whiskers. Default is 1.5.
    width : float, optional
        Width of the full boxplot elements. Default is 0.5.
    linewidth : float, optional
        Width of the lines of the boxplot elements. Default is 2.
    flierprops : dict, optional
        The style of the fliers; if None, then the default is used.
    sns_style: str, defualt='whitegrid'
        The style of seaborn boxplot. 
    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot for further tweaking.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.utils import boxplot
    >>> np.random.seed(10)
    >>> d = [np.random.normal(0, std, 100) for std in range(1, 5)]
    >>> labels = ['s1', 's2', 's3', 's4']
    >>> plot_custom_boxplot(d, labels, 
    ...                     title='Class assignment (roc-auc): PsA activity',
    ...                     y_label='roc-auc', 
    ...                     figsize=(12, 7),
    ...                     color="green",
    ...                     showfliers=False, 
    ...                     whis=2,
    ...                     width=0.3, 
    ...                     linewidth=1.5,
    ...                     flierprops=dict(marker='x', color='black', markersize=5))
    Notes
    -----
    Boxplots are a standardized way of displaying the distribution of data 
    based on a five-number summary: minimum, first quartile (Q1), median, 
    third quartile (Q3), and maximum. It can reveal outliers, 
    data symmetry, grouping, and skewness.
    """
    if flierprops is None:
        flierprops = dict(marker='o', color='red', alpha=0.5)
    
    # Create a figure and a set of subplots
    plt.figure(figsize=figsize)
    
    # Create the boxplot
    bplot = sns.boxplot(data=data, width=width, color=color, 
                        showfliers=showfliers, whis=whis, 
                        flierprops=flierprops,
                        linewidth=linewidth
                        )
    
    # Set labels and title
    bplot.set_title(title)
    bplot.set_ylabel(y_label)
    bplot.set_xticklabels(labels)
    
    # Set the style of the plot
    sns.set_style(sns_style)
    
    # Show the plot
    plt.show()
    return bplot

def plot_r_squared(
    y_true, y_pred, 
    model_name="Regression Model",
    figsize=(10, 6), 
    r_color='red', 
    pred_color='blue',
    sns_plot=False, 
    show_grid=True, 
    **scatter_kws
):
    """
    Plot the R-squared value for a regression model's predictions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values by the regression model.
    model_name : str, optional
        The name of the regression model for display in the plot title. 
        Default is "Regression Model".
    figsize : tuple, optional
        The size of the figure to plot (width, height in inches). 
        Default is (10, 6).
    r_color : str, optional
        The color of the line that represents the actual values.
        Default is 'red'.
    pred_color : str, optional
        The color of the scatter plot points for the predictions.
        Default is 'blue'.
    sns_plot : bool, optional
        If True, use seaborn for plotting. Otherwise, use matplotlib.
        Default is False.
    show_grid : bool, optional
        If True, display the grid on the plot. Default is True.
    scatter_kws : dict
        Additional keyword arguments to be passed to 
        the `scatter` function.

    Returns
    -------
    ax : Axes
        The matplotlib Axes object with the R-squared plot.
         
    Example 
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_r_squared 
    >>> # Generate some sample data
    >>> np.random.seed(0)
    >>> y_true_sample = np.random.rand(100) * 100
    >>> # simulated prediction with small random noise
    >>> y_pred_sample = y_true_sample * (np.random.rand(100) * 0.1 + 0.95)  
    # Use the sample data to plot R-squared
    >>> plot_r_squared(y_true_sample, y_pred_sample, "Sample Regression Model")

    """
    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot using seaborn or matplotlib
    if sns_plot:
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, color=pred_color,
                        **scatter_kws)
    else:
        ax.scatter(y_true, y_pred, color=pred_color, **scatter_kws)
    
    # Plot the line of perfect predictions
    ax.plot(y_true, y_true, color=r_color, label='Actual values')
    # Annotate the R-squared value
    ax.legend(labels=[f'Predictions (R² = {r_squared:.2f})', 'Actual values'])
    # Set the title and labels
    ax.set_title(f'{model_name}: R-squared')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    
    # Display the grid if requested
    if show_grid:
        ax.grid(show_grid)
    
    # Show the plot
    plt.show()

    return ax

def make_plot_colors(
    d , / , 
    colors:str | list[str]=None , axis:int = 0, 
    seed:int  =None, chunk:bool =... 
    ): 
    """ Select colors according to the data size along axis 
    
    Parameters 
    ----------
    d: Arraylike 
       Array data to select colors according to the axis 
    colors: str, list of Matplotlib.colors map, optional 
        The colors for plotting each columns of `X` except the depth. If not
        given, default colors are auto-generated.
        If `colors` is string and 'cs4'or 'xkcd' is included. 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS 
        should be used instead. In addition if the `'cs4'` or `'xkcd'` is  
        suffixed by colons and integer value like ``cs4:4`` or ``xkcd:4``, the 
        CS4 or XKCD colors should be used from index equals to ``4``. 
        
 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS can 
        be used by setting `colors` to ``'cs4'`` or ``'xkcd'``. To reproduce 
        the same CS4 or XKCD colors, set the `seed` parameter to a 
        specific value. 
           
    axis: int, default=0 
       Axis along with the colors must be generated. By default colors is 
       generated along the row axis 
       
    seed: int, optional 
       Allow to reproduce the Matplotlib.colors.CS4_COLORS if `colors` is 
       set to ``cs4``. 
       
    chunk: bool, default=True 
       Chunk generated colors to fit the exact length of the `d` size 
       
    Returns 
    -------
    colors: list 
       List of new generated colors 
       
    Examples 
    --------
    >>> import numpy as np 
    >>> from gofast.tools.utils import make_plot_colors
    >>> ar = np.random.randn (7, 2) 
    >>> make_plot_colors (ar )
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime']
    >>> make_plot_colors (ar , axis =1 ) 
    Out[6]: ['g', 'gray']
    >>> make_plot_colors (ar , axis =1 , colors ='cs4')
    ['#F0F8FF', '#FAEBD7']
    >>> len(make_plot_colors (ar , axis =1 , colors ='cs4', chunk=False))
    150
    >>> make_plot_colors (ar , axis =1 , colors ='cs4:4')
    ['#F0FFFF', '#F5F5DC']
    """
    
    # get the data size where colors must be fitted. 
    # note colors should match either the row axis or colurms axis 
    axis = str(axis).lower() 
    if 'columns1'.find (axis)>=0: 
        axis =1 
    else: axis =0
    
    # manage the array 
    d= is_iterable( d, exclude_string=True, transform=True)
    if not hasattr (d, '__array__'): 
        d = np.array(d, dtype =object ) 
    
    axis_length = len(d) if len(d.shape )==1 else d.shape [axis]
    m_cs = make_mpl_properties(axis_length )
    
     #manage colors 
    # we assume the first columns is dedicated for 
    if colors ==...: colors =None 
    if ( 
            isinstance (colors, str) and 
            ( 
                "cs4" in str(colors).lower() 
                 or 'xkcd' in str(colors).lower() 
                 )
            ): 
        #initilize colors infos
        c = copy.deepcopy(colors)
        if 'cs4' in str(colors).lower() : 
            DCOLORS = mcolors.CSS4_COLORS
        else: 
            # remake the dcolors my removing the xkcd: in the keys: 
            DCOLORS = dict(( (k.replace ('xkcd:', ''), c) 
                            for k, c in mcolors.XKCD_COLORS.items()))  
        
        key_colors = list(DCOLORS.keys ())
        colors = list(DCOLORS.values() )
        
        shuffle_cs4=True 
        
        cs4_start= None
        #------
        if ':' in str(c).lower():
            cs4_start = str(c).lower().split(':')[-1]
        #try to converert into integer 
        try: 
            cs4_start= int (cs4_start)
        except : 
            if str(cs4_start).lower() in key_colors: 
                cs4_start= key_colors.index (cs4_start)
                shuffle_cs4=False
            else: 
                pass 
        
        else: shuffle_cs4=False # keep CS4 and dont shuffle 
        
        cs4_start= cs4_start or 0
        
        if shuffle_cs4: 
            np.random.seed (seed )
            colors = list(np.random.choice(colors  , len(m_cs)))
        else: 
            if cs4_start > len(colors)-1: 
                cs4_start = 0 
    
            colors = colors[ cs4_start:]
    
    if colors is not None: 
        if not is_iterable(colors): 
            colors =[colors]
        colors += m_cs 
    else :
        colors = m_cs 
        
    # shrunk data to map the exact colors 
    chunk =True if chunk is ... else False 
    
    return colors[:axis_length] if chunk else colors 

def savefigure(
    fig: object, 
    figname: str = None, 
    ext: str = '.png', 
    **skws
    ):
    """
    Save a matplotlib figure to a file with optional name and extension. 
    
    Parameters
    ----------
    fig : object
        Matplotlib figure object to be saved.
    figname : str, optional
        Name of the output file for the figure. If not provided, the file will
        be named with a timestamp and the extension provided. If a directory
        path is included, ensure the directory exists.
    ext : str, optional
        Extension type for the figure file. Defaults to '.png'. Other common
        types include '.jpg', '.jpeg', '.pdf', and '.svg'.
    **skws : dict
        Additional keyword arguments to pass to `matplotlib.pyplot.savefig`.

    Returns
    -------
    None
        The function saves the figure directly to a file and does not return
        any value.

    Warns
    -----
    UserWarning
        Warns the user if no file name is provided, and a default file name is
        used instead.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> savefigure(fig, 'my_plot', ext='pdf')
    
    >>> savefigure(fig, 'plot_without_ext')
    
    >>> savefigure(fig)
    """
    ext = '.' + str(ext).lower().strip().replace('.', '')
    
    if figname is None:
        figname = '_' + os.path.splitext(os.path.basename(__file__))[0] +\
                  datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + '.' + ext
        warnings.warn("No name of figure is given. Figure should be renamed as "
                      f"{figname!r}")
        
    file, ex = os.path.splitext(figname)
    if ex in ('', None):
        ex = ext
        figname = file + '.' + ex

    return fig.savefig(figname, **skws)

def resetting_ticks(get_xyticks, number_of_ticks=None):
    """
    Reset the positions of ticks on the x or y axis modulo 100, returning a
    new array of tick positions.

    Parameters
    ----------
    get_xyticks : list or np.ndarray
        List or ndarray of tick positions obtained via ax.get_x|yticks().
    number_of_ticks : int, optional
        Specifies the number of ticks to set on the x or y axis. If not provided,
        it calculates based on the existing number of ticks or defaults to the
        length of `get_xyticks`.

    Returns
    -------
    list or np.ndarray
        A new list or ndarray of modified tick positions.

    Raises
    ------
    TypeError
        If `get_xyticks` is not a list or ndarray.
    ValueError
        If `number_of_ticks` is not an integer or cannot be converted to an
        integer.

    Examples
    --------
    >>> import numpy as np
    >>> ticks = [10, 20, 30, 40, 50]
    >>> resetting_ticks(ticks)
    >>> resetting_ticks(ticks, 3)
    """
    if not isinstance(get_xyticks, (list, np.ndarray)):
        warnings.warn(
            'Arguments get_xyticks must be a list or ndarray, not <{0}>.'.format(
                type(get_xyticks))
        )
        raise TypeError(
            '<{0}> found. "get_xyticks" must be a list or ndarray.'.format(
                type(get_xyticks))
        )
    
    if number_of_ticks is None:
        if len(get_xyticks) > 2:
            number_of_ticks = int((len(get_xyticks) - 1) / 2)
        else:
            number_of_ticks = len(get_xyticks)
    
    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn('"number_of_ticks" must be an integer, not <{0}>.'.format(
                type(number_of_ticks)))
            raise ValueError(f'<{type(number_of_ticks).__name__}> detected. Must be integer.')
        
    number_of_ticks = int(number_of_ticks)
    
    if len(get_xyticks) > 2:
        if get_xyticks[1] % 10 != 0:
            get_xyticks[1] = get_xyticks[1] + (10 - get_xyticks[1] % 10)
        if get_xyticks[-2] % 10 != 0:
            get_xyticks[-2] = get_xyticks[-2] - get_xyticks[-2] % 10
    
        new_array = np.linspace(get_xyticks[1], get_xyticks[-2], number_of_ticks)
    elif len(get_xyticks) < 2:
        new_array = np.array(get_xyticks)

    return new_array

def make_mpl_properties(n: int, prop: str = 'color') -> list:
    """
    Generate a list of matplotlib properties such as colors, markers, or line 
    styles to match the specified number of samples.

    Parameters
    ----------
    n : int
        Number of property items needed. It generates a group of property items.
    prop : str, default='color'
        Name of the property to retrieve. Accepts 'color', 'marker', or 'line'.

    Returns
    -------
    list
        A list of property items with size equal to `n`.

    Raises
    ------
    ValueError
        If the `prop` argument is not one of 'color', 'marker', or 'line'.

    Examples
    --------
    >>> from gofast.tools.utils import make_mpl_properties
    >>> make_mpl_properties(10)
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', (0.6, 0.6, 0.6)]
    >>> make_mpl_properties(100, prop='marker')
    ['o', '^', 'x', 'D', ..., 11, 'None', None, ' ', '']
    >>> make_mpl_properties(50, prop='line')
    ['-', '-', '--', '-.', ..., 'solid', 'dashed', 'dashdot', 'dotted']
    """
    n=int(_assert_all_types(n, int, float, objname ="'n'"))
    prop = str(prop).lower().strip().replace('s', '')
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property {prop!r} is not available."
                         " Expect 'color', 'marker', or 'line'.")
    # Generate property lists
    props =[]
    if prop=='color': 
        d_colors =  D_COLORS 
        d_colors = mpl.colors.ListedColormap(d_colors[:n]).colors
        if len(d_colors) == n: 
            props= d_colors 
        else:
            rcolors = list(itertools.repeat(
                d_colors , (n + len(d_colors))//len(d_colors))) 
    
            props  = list(itertools.chain(*rcolors))
        
    if prop=='marker': 
        
        d_markers =  D_MARKERS + list(mpl.lines.Line2D.markers.keys()) 
        rmarkers = list(itertools.repeat(
            d_markers , (n + len(d_markers))//len(d_markers))) 
        
        props  = list(itertools.chain(*rmarkers))
    # repeat the lines to meet the number of cv_size 
    if prop=='line': 
        d_lines =  D_STYLES
        rlines = list(itertools.repeat(
            d_lines , (n + len(d_lines))//len(d_lines))) 
        # combine all repeatlines 
        props  = list(itertools.chain(*rlines))
    
    return props [: n ]
  
def resetting_colorbar_bound(
    cbmax: float,
    cbmin: float,
    number_of_ticks: int = 5,
    logscale: bool = False) -> np.ndarray:
    """
    Adjusts the bounds and tick spacing of a colorbar to make the ticks easier
    to read, optionally using a logarithmic scale.

    Parameters
    ----------
    cbmax : float
        Maximum value of the colorbar.
    cbmin : float
        Minimum value of the colorbar.
    number_of_ticks : int, optional
        Number of ticks to be placed on the colorbar. Defaults to 5.
    logscale : bool, optional
        Set to True if the colorbar should use a logarithmic scale. Defaults to False.

    Returns
    -------
    np.ndarray
        Array of tick values for the colorbar.

    Raises
    ------
    ValueError
        If `number_of_ticks` is not an integer or convertible to an integer.

    Examples
    --------
    >>> resetting_colorbar_bound(100, 0)
    array([ 0., 25., 50., 75., 100.])

    >>> resetting_colorbar_bound(100, 0, logscale=True)
    array([  1.,  5.62341,  31.62277,  177.82794,  1000.])
    """
    def round_modulo10(value, mod10):
        """
        Rounds value to nearest multiple of mod10, or to the nearest half multiple.
        """
        if value % mod10 == 0:
            return value
        elif value % (mod10 / 2) == 0:
            return value
        else:
            return value - (value % mod10)

    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn('"number_of_ticks" must be an integer, not'
                          f'<{type(number_of_ticks).__name__}>.')
            raise ValueError(f'<{type(number_of_ticks).__name__}> detected.'
                             'Must be an integer.')

    mod10 = np.log10(10) if logscale else 10
    
    if cbmax % cbmin == 0:
        return np.linspace(cbmin, cbmax, number_of_ticks)
    else:
        startpoint = cbmin + (mod10 - cbmin % mod10)
        endpoint = cbmax - (cbmax % mod10)
        return np.array(
            [round_modulo10(ii, mod10) for ii in np.linspace(
                startpoint, endpoint, number_of_ticks)]
        )
    
def fmt_text(
    data_text: str, 
    fmt: str = '~', 
    leftspace: int = 3, 
    return_to_line: int = 77
    ) -> str:
    """
    Formats a given text with specified left padding and underlines to make 
    a formatted report.

    Parameters
    ----------
    data_text : str
        The long text to be formatted.
    fmt : str, optional
        The character used for underlining and formatting. Default is '~'.
    leftspace : int, optional
        The number of spaces to indent the text from the left margin. 
        Default is 3.
    return_to_line : int, optional
        The maximum number of characters in a line before wrapping to the next 
        line. Default is 77.

    Returns
    -------
    str
        The formatted text as a string with underlines and line breaks.

    Examples
    --------
    >>> text = "This is a sample text that will be formatted with left spaces, 
    ...  underlines, and auto-wrapping."
    >>> print(fmt_text(text))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       This is a sample text that will be formatted with left -
       spaces, underlines, and auto-wrapping. ~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    return_to_line = int(return_to_line)
    begin_text = ' ' * leftspace
    formatted_text = begin_text + fmt * (return_to_line + 7) + '\n' + begin_text

    ss = 0
    for ii, char in enumerate(data_text):  # loop through the text
        if ii == len(data_text) - 1:  # if it is the last character of the text
            formatted_text += char + f' {fmt}\n' + begin_text + fmt * (
                return_to_line + 7) + '\n'
            break
        if ss == return_to_line:
            if data_text[ii + 1] != ' ':
                formatted_text += f' {fmt}-\n' + begin_text + fmt
            else:
                formatted_text += f' {fmt}\n' + begin_text + fmt
            ss = 0
        formatted_text += char  # add character
        ss += 1

    return formatted_text
 
def plotvec1(u: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
    """
    Plot three vectors as arrows on a 2D graph to visualize their 
    directions and magnitudes. The vectors should be 2D for proper visualization.

    Parameters
    ----------
    u : np.ndarray
        First vector, displayed in red. Expected to be a 1D array of length 2.
    z : np.ndarray
        Second vector. Expected to be a 1D array of length 2.
    v : np.ndarray
        Third vector, displayed in blue. Expected to be a 1D array of length 2.

    Returns
    -------
    None
        Displays a plot with vectors u, z, and v.

    Raises
    ------
    ValueError
        If any of the vectors are not 1D arrays of length 2.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([1, 2])
    >>> z = np.array([2, -1])
    >>> v = np.array([-1, 2])
    >>> plotvec1(u, z, v)
    """
    # Check if all vectors are numpy arrays of length 2
    for vec, name in zip([u, z, v], ['u', 'z', 'v']):
        if not isinstance(vec, np.ndarray) or vec.shape != (2,):
            raise ValueError(f"Vector {name} must be a 1D numpy array of length 2.")

    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')

    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()

def plotvec2(a: np.ndarray, b: np.ndarray) -> None:
    """
    Plot two vectors as arrows on a 2D graph to visualize their directions 
    and orthogonality. The vectors should be 2D for proper visualization.

    Parameters
    ----------
    a : np.ndarray
        First vector, displayed in red. Expected to be a 1D array of length 2.
    b : np.ndarray
        Second vector, displayed in blue. Expected to be a 1D array of length 2.

    Returns
    -------
    None
        Displays a plot with vectors a and b.

    Raises
    ------
    ValueError
        If any of the vectors are not 1D arrays of length 2.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 0])
    >>> b = np.array([0, 1])
    >>> plotvec2(a, b)
    """
    # Validate the input vectors
    for vec, name in zip([a, b], ['a', 'b']):
        if not isinstance(vec, np.ndarray) or vec.shape != (2,):
            raise ValueError(f"Vector {name} must be a 1D numpy array of length 2.")

    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    
    ax.arrow(0, 0, *b, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()

def plot_errorbar(
    ax,
    x_ar,
    y_ar,
    y_err=None,
    x_err=None,
    color='k',
    marker='x',
    ms=2, 
    ls=':', 
    lw=1, 
    e_capsize=2,
    e_capthick=.5,
    picker=None,
    **kws
 ):
    """
    convinience function to make an error bar instance
    
    Parameters
    ------------
    
    ax: matplotlib.axes 
        instance axes to put error bar plot on

    x_array: np.ndarray(nx)
        array of x values to plot
                  
    y_array: np.ndarray(nx)
        array of y values to plot
                  
    y_error: np.ndarray(nx)
        array of errors in y-direction to plot
    
    x_error: np.ndarray(ns)
        array of error in x-direction to plot
                  
    color: string or (r, g, b)
        color of marker, line and error bar
                
    marker: string
        marker type to plot data as
                 
    ms: float
        size of marker
             
    ls: string
        line style between markers
             
    lw: float
        width of line between markers
    
    e_capsize: float
        size of error bar cap
    
    e_capthick: float
        thickness of error bar cap
    
    picker: float
          radius in points to be able to pick a point. 
        
        
    Returns:
    ---------
    errorbar_object: matplotlib.Axes.errorbar 
           error bar object containing line data, errorbars, etc.
    """
    # this is to make sure error bars 
    #plot in full and not just a dashed line
    eobj = ax.errorbar(
        x_ar,
        y_ar,
        marker=marker,
        ms=ms,
        mfc='None',
        mew=lw,
        mec=color,
        ls=ls,
        xerr=x_err,
        yerr=y_err,
        ecolor=color,
        color=color,
        picker=picker,
        lw=lw,
        elinewidth=lw,
        capsize=e_capsize,
        # capthick=e_capthick
        **kws
         )
    
    return eobj

def get_color_palette (RGB_color_palette): 
    """
    Convert RGB color into matplotlib color palette. In the RGB color 
    system two bits of data are used for each color, red, green, and blue. 
    That means that each color runson a scale from 0 to 255. Black  would be
    00,00,00, while white would be 255,255,255. Matplotlib has lots of
    pre-defined colormaps for us . They are all normalized to 255, so they run
    from 0 to 1. So you need only normalize data, then we can manually  select 
    colors from a color map  

    :param RGB_color_palette: str value of RGB value 
    :type RGB_color_palette: str 
        
    :returns: rgba, tuple of (R, G, B)
    :rtype: tuple
     
    :Example: 
        
        >>> from gofast.tools.utils import get_color_palette 
        >>> get_color_palette (RGB_color_palette ='R128B128')
    """  
    
    def ascertain_cp (cp): 
        if cp >255. : 
            warnings.warn(
                ' !RGB value is range 0 to 255 pixels , '
                'not beyond !. Your input values is = {0}.'.format(cp))
            raise ValueError('Error color RGBA value ! '
                             'RGB value  provided is = {0}.'
                            ' It is larger than 255 pixels.'.format(cp))
        return cp
    if isinstance(RGB_color_palette,(float, int, str)): 
        try : 
            float(RGB_color_palette)
        except : 
              RGB_color_palette= RGB_color_palette.lower()
             
        else : return ascertain_cp(float(RGB_color_palette))/255.
    
    rgba = np.zeros((3,))
    
    if 'r' in RGB_color_palette : 
        knae = RGB_color_palette .replace('r', '').replace(
            'g', '/').replace('b', '/').split('/')
        try :
            _knae = ascertain_cp(float(knae[0]))
        except : 
            rgba[0]=1.
        else : rgba [0] = _knae /255.
        
    if 'g' in RGB_color_palette : 
        knae = RGB_color_palette .replace('g', '/').replace(
            'b', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except : 
            rgba [1]=1.
            
        else :rgba[1]= _knae /255.
    if 'b' in RGB_color_palette : 
        knae = knae = RGB_color_palette .replace('g', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except :
            rgba[2]=1.
        else :rgba[2]= _knae /255.
        
    return tuple(rgba)       

def _get_xticks_formatage (
        ax,  xtick_range, space= 14 , step=7,
        fmt ='{}',auto = False, ticks ='x', **xlkws):
    """ Skip xticks label at every number of spaces 
    :param ax: matplotlib axes 
    :param xtick_range: list of the xticks values 
    :param space: interval that the label must be shown.
    :param step: the number of label to skip.
    :param fmt: str, formatage type. 
    :param ticks: str, default='x', the ticks axis to format the labels. 
      can be ``'y'``. 
    :param auto: bool , if ``True`` a dynamic tick formatage will start. 
    
    """
    def format_ticks (ind, x):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % step ==0: 
            return fmt.format (ind)
        else: None 
        
    # show label every 'space'samples 
    if auto: 
        space = 10.
        step = int (np.ceil ( len(xtick_range)/ space )) 
        
    rotation = xlkws.get('rotation', 90 ) if 'rotation' in xlkws.keys (
        ) else xlkws.get('rotate_xlabel', 90 )
    
    if len(xtick_range) >= space :
        if ticks=='y': 
            ax.yaxis.set_major_formatter (plt.FuncFormatter(format_ticks))
        else: 
            ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))

        plt.setp(ax.get_yticklabels() if ticks=='y' else ax.get_xticklabels(), 
                 rotation = rotation )
    else: 
        
        # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        # # ticks_loc = ax.get_xticks().tolist()
        # ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
        # ax.set_xticklabels([fmt.format(x) for x in ticks_loc])
        tlst = [fmt.format(item) for item in  xtick_range]
        ax.set_yticklabels(tlst, **xlkws) if ticks=='y' \
            else ax.set_xticklabels(tlst, **xlkws) 
  
def _set_sns_style (s, /): 
    """ Set sns style whether boolean or string is given""" 
    s = str(s).lower()
    s = re.sub(r'true|none', 'darkgrid', s)
    return sns.set_style(s) 

def _is_target_in (X, y=None, target_name=None): 
    """ Create new target name for target_name if given 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        target data for plotting. Note that multitarget outpout is not 
        allowed yet. Moroever, it `y` is given as a dataframe, 'target_name' must 
        be supplied to retrive y as a pandas series object, otherwise an 
        error will raise. 
    :param target_name: str,  
        target name. If given and `y` is ``None``, Will try to find `target_name`
        in the `X` columns. If 'target_name' does not exist, plot for target is 
        cancelled. 
        
    :return y: Series 
    """
    _assert_all_types(X, pd.DataFrame)
    
    if y is not None: 
        y = _assert_all_types(y , pd.Series, pd.DataFrame, np.ndarray)
        
        if hasattr (y, 'columns'): 
            if target_name not in (y.columns): target_name = None 
            if target_name is None: 
                raise TypeError (
                    "'target_name' must be supplied when y is a dataframe.")
            y = y [target_name ]
        elif hasattr (y, 'name'): 
            target_name = target_name or y.name 
            # reformat inplace the name of series 
            y.name = target_name 
            
        elif hasattr(y, '__array__'): 
            y = pd.Series (y, name = target_name or 'target')
            
    elif y is None: 
        if target_name in X.columns :
            y = X.pop(target_name)

    return X, y 

def _toggle_target_in  (X , y , pos=None): 
    """ Toggle the target in the convenient position. By default the target 
    plot is the last subplots 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        the target for  plotting. 
    :param pos: int, the position to insert y in the dataframe X 
        By default , `y` is located at the last position 
        
    :return: Dataframe 
        Dataframe containing the target 'y'
        
    """
    
    pos =  0 if pos ==0  else ( pos or X.shape [1])

    pos= int ( _assert_all_types(pos, int, float ) ) 
    ms= ("The positionning of the target is out of the bound."
         "{} position is used instead.")
    
    if pos > X.shape[1] : 
        warnings.warn(ms.format('The last'))
        pos=X.shape[1]
    elif pos < 0: 
        warnings.warn(ms.format(
            " Negative index is not allowed. The first")
                      )
        pos=0 
 
    X.insert (pos, y.name, y )
    
    return X
    
def _skip_log10_columns ( X, column2skip, pattern =None , inplace =True): 
    """ Skip the columns that dont need to put value in logarithms.
    
    :param X: dataframe 
        pandas dataframe with valid columns 
    :param column2skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise. 
    :param pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        
    :return X: Dataframe
        Dataframe modified inplace with values computed in log10 
        except the skipped columns. 
        
    :example: 
       >>> from gofast.datasets import load_hlogs 
       >>> from gofast.tools.utils import _skip_log10_columns 
       >>> X0, _= load_hlogs (as_frame =True ) 
       >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
       >>> X0['sp'][:3] , X0['resistivity'][:3]  
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    15.919130
            1    16.000000
            2    24.422316
            Name: resistivity, dtype: float64)
       >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                         'strata_name', 'rock_name', 'well_diameter', 'sp']
       >>> _skip_log10_columns (X0, column2skip)
       >>> # now let visualize the same keys values 
       >>> X0['sp'][:3] , X0['resistivity'][:3]
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    1.201919
            1    1.204120
            2    1.387787
            Name: resistivity, dtype: float64)
      >>> # it is obvious the `resistiviy` values is log10 
      >>> # while `sp` still remains the same 
      
    """
    X0 = X.copy () 
    if not is_iterable( column2skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(column2skip).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    
    if isinstance(column2skip, str):
        column2skip = str2columns (column2skip, pattern=pattern  )
    #assert whether column to skip is in 
    if column2skip:
        cskip = copy.deepcopy (column2skip) 
        column2skip = is_in_if(X.columns, column2skip, return_diff= True)
        if len(column2skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
        if inplace : 
            X[column2skip] = np.log10 ( X[column2skip] ) 
            X.drop (columns =cskip , inplace =True )
            return  
        else : 
            X0[column2skip] = np.log10 ( X0[column2skip] ) 
            
    return X0
    
def plot_bar(x, y, wh= .8,  kind ='v', fig_size =(8, 6), savefig=None,
             xlabel =None, ylabel=None, fig_title=None, **bar_kws): 
    """
    Make a vertical or horizontal bar plot.

    The bars are positioned at x or y with the given alignment. Their dimensions 
    are given by width and height. The horizontal baseline is left (default 0)
    while the vertical baseline is bottom (default=0)
    
    Many parameters can take either a single value applying to all bars or a 
    sequence of values, one for each bar.
    
    Parameters 
    -----------
    x: float or array-like
        The x coordinates of the bars. is 'x' for vertical bar plot as `kind` 
        is set to ``v``(default) or `y` for horizontal bar plot as `kind` is 
        set to``h``. 
        See also align for the alignment of the bars to the coordinates.
    y: float or array-like
        The height(s) for vertical and width(s) for horizonatal of the bars.
    
    wh: float or array-like, default: 0.8
        The width(s) for vertical or height(s) for horizaontal of the bars.
        
    kind: str, ['vertical', 'horizontal'], default='vertical'
        The kind of bar plot. Can be the horizontal or vertical bar plots. 
    bar_kws: dict, 
        Additional keywords arguments passed to : 
            :func:`~matplotlib.pyplot.bar` or :func:`~matplotlib.pyplot.barh`. 
    """
    
    assert str(kind).lower().strip() in ("vertical", 'v',"horizontal", "h"), (
        "Support only the horizontal 'h' and vertical 'v' bar plots."
        " Got {kind!r}")
    kind =str(kind).lower().strip()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize =fig_size)
    if kind in ("vertical", "v"): 
        ax.bar (x, height= y, width =  wh , **bar_kws)
    elif kind in ("horizontal", "h"): 
        ax.barh (x , width =y , height =wh, **bar_kws)
        
    ax.set_xlabel (xlabel )
    ax.set_ylabel(ylabel) 
    ax.set_title (fig_title)
    if savefig is not  None: 
        savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    

def _format_ticks (value, tick_number, fmt ='S{:02}', nskip =7 ):
    """ Format thick parameter with 'FuncFormatter(func)'
    rather than using `axi.xaxis.set_major_locator (plt.MaxNLocator(3))`
    ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
    
    :param value: tick range values for formatting 
    :param tick_number: number of ticks to format 
    :param fmt: str, default='S{:02}', kind of tick formatage 
    :param nskip: int, default =7, number of tick to skip 
    
    """
    if value % nskip==0: 
        return fmt.format(int(value)+ 1)
    else: None 
    

def plot_confidence(
    y: Optional[Union[str, ArrayLike]] = None, 
    x: Optional[Union[str, ArrayLike]] = None,  
    data: Optional[DataFrame] = None,  
    ci: float = .95,  
    kind: str = 'line', 
    b_samples: int = 1000, 
    **sns_kws: Dict
) -> plt.Axes:
    """
    Plot confidence interval data using a line plot, regression plot, 
    or the bootstrap method.
    
    A Confidence Interval (CI) is an estimate derived from observed data statistics, 
    indicating a range where a population parameter is likely to be found at a 
    specified confidence level. Introduced by Jerzy Neyman in 1937, CI is a crucial 
    concept in statistical inference. Common types include CI for mean, median, 
    the difference between means, a proportion, and the difference in proportions.

    Parameters 
    ----------
    y : Union[np.ndarray, str], optional
        Dependent variable values. If a string, `y` should be a column name in 
        `data`. `data` cannot be None in this case.
    x : Union[np.ndarray, str], optional
        Independent variable values. If a string, `x` should be a column name in 
        `data`. `data` cannot be None in this case.
    data : pd.DataFrame, optional
        Input data structure. Can be a long-form collection of vectors that can be 
        assigned to named variables or a wide-form dataset that will be reshaped.
    ci : float, default=0.95
        The confidence level for the interval.
    kind : str, default='line'
        The type of plot. Options include 'line', 'reg', or 'bootstrap'.
    b_samples : int, default=1000
        The number of bootstrap samples to use for the 'bootstrap' method.
    sns_kws : Dict
        Additional keyword arguments passed to the seaborn plot function.

    Returns 
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.utils import plot_confidence 
    >>> df = pd.DataFrame({'x': range(10), 'y': np.random.rand(10)})
    >>> ax = plot_confidence(x='x', y='y', data=df, kind='line', ci=0.95)
    >>> plt.show()
    
    >>> ax = plot_confidence(y='y', data=df, kind='bootstrap', ci=0.95, b_samples=500)
    >>> plt.show()
    """

    plot_functions = {
        'line': lambda: sns.lineplot(data=data, x=x, y=y, errorbar=('ci', ci), **sns_kws),
        'reg': lambda: sns.regplot(data=data, x=x, y=y, ci=ci, **sns_kws)
    }
    if isinstance ( data, dict): 
        data = pd.DataFrame ( data )
        
    x, y = assert_xy_in(x, y, data=data, ignore ="x")
    if kind in plot_functions:
        ax = plot_functions[kind]()
    elif kind.lower().startswith('boot'):
        if y is None:
            raise ValueError("y must be provided for bootstrap method.")
        if not isinstance(b_samples, int):
            raise ValueError("`b_samples` must be an integer.")

        medians = [np.median(resample(y, n_samples=len(y))) for _ in range(b_samples)]
        plt.hist(medians)
        plt.show()
        
        p = ((1.0 - ci) / 2.0) * 100
        lower, upper = np.percentile(medians, [p, 100 - p])
        print(f"{ci*100}% confidence interval between {lower} and {upper}")

        ax = plt.gca()
    else:
        raise ValueError(f"Unrecognized plot kind: {kind}")

    return ax
   
def plot_confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    data: Optional[DataFrame] = None,
    figsize: Tuple[int, int] = (6, 6),
    scatter_s: int = 0.5,
    line_colors: Tuple[str, str, str] = ('firebrick', 'fuchsia', 'blue'),
    line_styles: Tuple[str, str, str] = ('-', '--', ':'),
    title: str = 'Different Standard Deviations',
    show_legend: bool = True
) -> plt.Axes:
    """
    Plots the confidence ellipse of a two-dimensional dataset.

    This function visualizes the confidence ellipse representing the covariance 
    of the provided 'x' and 'y' variables. The ellipses plotted represent 1, 2, 
    and 3 standard deviations from the mean.

    Parameters
    ----------
    x : Union[str, np.ndarray, pd.Series]
        The x-coordinates of the data points or column name in DataFrame.
    y : Union[str, np.ndarray, pd.Series]
        The y-coordinates of the data points or column name in DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (6, 6).
    scatter_s : int, optional
        The size of the scatter plot markers. Default is 0.5.
    line_colors : Tuple[str, str, str], optional
        The colors of the lines for the 1, 2, and 3 std deviation ellipses.
    line_styles : Tuple[str, str, str], optional
        The line styles for the 1, 2, and 3 std deviation ellipses.
    title : str, optional
        The title of the plot. Default is 'Different Standard Deviations'.
    show_legend : bool, optional
        If True, shows the legend. Default is True.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    Note 
    -----
    The approach that is used to obtain the correct geometry 
    is explained and proved here:
      https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
      
    The method avoids the use of an iterative eigen decomposition 
    algorithm and makes use of the fact that a normalized covariance 
    matrix (composed of pearson correlation coefficients and ones) is 
    particularly easy to handle.
    
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> ax = plot_confidence_ellipse(x, y)
    >>> plt.show()
    """
    x, y= assert_xy_in(x, y, data = data )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.scatter(x, y, s=scatter_s)

    for n_std, color, style in zip([1, 2, 3], line_colors, line_styles):
        confidence_ellipse(x, y, ax, n_std=n_std, label=f'${n_std}\sigma$',
                           edgecolor=color, linestyle=style)

    ax.set_title(title)
    if show_legend:
        ax.legend()
    return ax

def confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    ax: plt.Axes, 
    n_std: float = 3.0, 
    facecolor: str = 'none', 
    data: Optional[DataFrame] = None,
    **kwargs
) -> Ellipse:
    """
    Creates a covariance confidence ellipse of x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays with the same size.
    ax : plt.Axes
        The axes object where the ellipse will be plotted.
    n_std : float, optional
        The number of standard deviations to determine the ellipse's radius. 
        Default is 3.
    facecolor : str, optional
        The color of the ellipse's face. Default is 'none' (no fill).
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.

    **kwargs
        Additional arguments passed to the Ellipse patch.

    Returns
    -------
    ellipse : Ellipse
        The Ellipse object added to the axes.

    Raises
    ------
    ValueError
        If 'x' and 'y' are not of the same size.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> fig, ax = plt.subplots()
    >>> confidence_ellipse(x, y, ax, n_std=2, edgecolor='red')
    >>> ax.scatter(x, y, s=3)
    >>> plt.show()
    """
    x, y = assert_xy_in(x, y, data = data )

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(
        scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_text (
    x, y, 
    text=None , 
    data =None, 
    coerce =False, 
    basename ='S', 
    fig_size =( 7, 7 ), 
    show_line =False, 
    step = None , 
    xlabel ='', 
    ylabel ='', 
    color= 'k', 
    mcolor='k', 
    lcolor=None, 
    show_leg =False,
    linelabel='', 
    markerlabel='', 
    ax=None, 
    **text_kws
    ): 
    """ Plot text(s) indicating each position in the line. 
    
    Parameters 
    -----------
    x, y: str, float, Array-like 
        The position to place the text. By default, this is in data 
        coordinates. The coordinate system can be changed using the 
        transform parameter.
        
    text: str, 
        The text
        
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    coerce:bool, default=False 
       Force the plot despite the given textes do not match the number of  
       positions `x` and `y`. If ``False``, number of positions must be 
       consistent with x and y, otherwise error raises. 
       
    basename: str, default='S' 
       the text to prefix the position when the text is not given. 
       
    fig_size: tuple, default=(7, 7) 
       Matplotlib figure size.
       
    show_line: bool, default=False 
       Display the line from x, y. 
       
    step: int,Optional 
       The number of intermediate positions to skip in the plotting text. 
       
    xlabel, ylabel: str, Optional, 
       The labels of x and y. 
       
    color: str, default='k', 
       Text color.
       
    mcolor: str, default='k', 
       Marker color. 
       
    lcolor: str, Optional 
       Line color if `show_line` is set to ``True``. 
       
    show_leg: bool, default=False 
       Display the legend of line and marker labels. 
       
    linelabel, markerlabel: str, Optional 
        The labels of the line and marker. 
       
    ax: Matplotlib.Axes, optional 
       Support plot to another axes 
       
       .. versionadded:: 0.2.5 
       
    text_kws: dict, 
       Keyword arguments passed to :meth:`matplotlib.axes.Axes.text`. 

    Return 
    -------
    ax: Matplotlib axes 
    
    Examples 
    --------
    >>> import gofast as gf 
    >>> data =gf.make_erp (as_frame =True, n_stations= 7 )
    >>> x , y =[ 0, 1, 3 ], [2, 3, 6] 
    >>> texto = ['AMT-E1147', 'AMT-E1148',  'AMT-E180']
    >>> plot_text (x, y , text = texto)# no need to set  coerce, same length 
    >>> data =gf.make_erp (as_frame =True, n_stations= 20 )
    >>> x , y = data.easting, data.northing
    >>> text1 = ['AMT-E1147', 'AMT-E1148',  'AMT-E180'] 
    >>> plot_text (x, y , coerce =True , text = text1 , show_leg= True, 
                   show_line=True, linelabel='E1-line', markerlabel= 'Site', 
               basename ='AMT-E0' 
               )
    """
    # assume x, y  series are passed 
    if isinstance(x, str) or hasattr ( x, 'name'): 
        xlabel = x  if isinstance(x, str) else x.name 
        
    if isinstance(y, str) or hasattr ( y, 'name'): 
        ylabel = y  if isinstance(y, str) else y.name 
        
    if x is None and  y is None:
        raise TypeError("x and y are needed for text plot. NoneType"
                        " cannot be plotted.")    
        
    x, y = assert_xy_in(x, y, data = data ) 

    if text is None and not coerce: 
       raise TypeError ("Text cannot be plotted. To force plotting text with"
                        " the basename, set ``coerce=True``.")

    text = is_iterable(text , exclude_string= True , transform =True )
    
    if ( len(text) != len(y) 
        and not coerce) : 
        raise ValueError("In principle text array and x/y must be consistent."
                         f" Got {len(text)} and {len(y)}. To plot anyway,"
                         " set ``coerce=True``.")
    if coerce : 
        basename =str(basename)
        text += [f'{basename}{i+len(text):02}' for i in range (len(y) )]

    if step is not None: 
        step = _assert_all_types(step , float, int , objname ='Step') 
        for ii in range(len(text)): 
            if not ii% step ==0: 
                text[ii]=''

    if ax is None: 
        
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # plot = ax.scatter if show_line else ax.plot 
    ax_m = None 
    if show_line: 
        ax.plot (x, y , label = linelabel, color =lcolor 
                 ) 
        
    for ix, iy , name in zip (x, y, text ): 
        ax.text ( ix , iy , name , color = color,  **text_kws)
        if name !='':
           ax_m  = ax.scatter ( [ix], [iy] , marker ='o', color =mcolor, 
                       )
  
    ax.set_xlabel (xlabel)
    ax.set_ylabel (ylabel) 
    
    ax_m.set_label ( markerlabel) if ax_m is not None else None 
    
    if show_leg : 
        ax.legend () 
        
    return ax 

def _make_axe_multiple ( n, ncols = 3 , fig_size =None, fig =None, ax= ... ): 
    """ Make multiple subplot axes from number of objects. """
    if is_iterable (n): 
       n = len(n) 
     
    nrows = n // ncols + ( n % ncols ) 
    if nrows ==0: 
       nrows =1 
       
    if ax in ( ... , None) : 
        fig, ax = plt.subplots (nrows, ncols, figsize = fig_size )  
    
    return fig , ax 
    

def _manage_plot_kws ( kws, dkws = dict () ): 
    """ Check whether the default values are in plot_kws then pop it"""
    
    kws = dkws or kws 
    for key in dkws.keys(): 
        # if key not in then add it. 
        if key not in kws.keys(): 
            kws[key] = dkws.get(key)
            
    return kws 

def is_colormap(color_name):
    """
    Checks if the given color name is a valid colormap in Matplotlib.

    Parameters:
    - color_name: str, the name of the color or colormap to check.

    Returns:
    - bool, True if the color_name is a colormap, False otherwise.
    """
    # Get the list of all colormaps in Matplotlib
    colormaps = plt.colormaps()
    # Check if the given color_name is in the list of colormaps
    return color_name in colormaps


  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
    
    
    
