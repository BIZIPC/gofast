# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
The `dimensionality` module offers visualization tools for dimensionality
reduction analysis. It includes functions for plotting unified 
PCA results, individual PCA components, and cumulative variance explained by 
PCA components.
"""

from __future__ import annotations 
import warnings 
import scipy.sparse as sp 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA

from ..api.types import NDArray, ArrayLike
from ..api.types import List, Tuple, Optional
from ..tools.coreutils import is_iterable 
from ..tools.validator import check_array, _check_array_in 
from .utils import pobj, make_mpl_properties 

__all__=['plot_unified_pca', 'plot_pca_components', 'plot_cumulative_variance']

def plot_pca_components (
    components: ArrayLike, *, 
    feature_names: Optional[List[str]]= None , 
    cmap: str= 'viridis', 
    savefig: Optional [str]=None, 
    **kws
    ): 

    """
    Visualize the coefficients of principal component analysis (PCA) as 
    a heatmap.

    Parameters
    ----------
    components : ndarray or PCA object
        The PCA components array with shape (n_components, n_features) or a PCA
        object from :class:`gofast.analysis.dimensionality.nPCA`. If a PCA object 
        is provided, `feature_names` is not needed as it will retrieve the feature
        names automatically from the PCA object.
    feature_names : list of str, optional
        List of feature names for the heatmap axes. This is necessary if `components` 
        is an array. If `components` is a PCA object, feature names are retrieved 
        automatically and this parameter can be omitted.
    cmap : str, default 'viridis'
        The colormap name for the heatmap. Uses Matplotlib colormaps.
    savefig : str, optional
        Path where the figure should be saved. If not provided, the figure is not 
        saved.
    kws : dict
        Additional keyword arguments passed to :meth:`matplotlib.pyplot.matshow`.

    Examples
    --------
    Using a PCA object:

    >>> from gofast.datasets import fetch_data
    >>> from gofast.analysis import nPCA
    >>> from gofast.plot.dimensionality import plot_pca_components
    >>> X, _ = fetch_data('bagoue pca', return_X_y=True)
    >>> pca = nPCA(X, n_components=2, return_X=False)
    >>> plot_pca_components(pca)

    Using components and feature names individually:

    >>> components = pca.components_
    >>> features = pca.feature_names_in_
    >>> plot_pca_components(components, feature_names=features, cmap='jet_r')

    """
    if sp.issparse (components): 
        raise TypeError ("Sparse array is not supported for PCA "
                         "components visualization."
                         )
    # if pca object is given , get the features names
    if hasattr(components, "feature_names_in_"): 
        feature_names = list (getattr (components , "feature_names_in_" ) ) 
        
    if not hasattr (components , "__array__"): 
        components = _check_array_in  (components, 'components_')
        
    plt.matshow(components, cmap =cmap , **kws)
    plt.yticks ([0 , 1], ['First component', 'Second component']) 
    cb=plt.colorbar() 
    cb.set_label('Coeff value')
    if not is_iterable(feature_names ): 
        feature_names = [feature_names ]
        
    if len(feature_names)!= components.shape [1] :
        warnings.warn("Number of features and eigenvectors might"
                      " be consistent, expect {0}, got {1}". format( 
                          components.shape[1], len(feature_names))
                      )
        feature_names=None 
    if feature_names is not None: 
        plt.xticks (range (len(feature_names)), 
                    feature_names , rotation = 60 , ha='left' 
                    )
    plt.xlabel ("Feature") 
    plt.ylabel ("Principal components") 
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_cumulative_variance(
    data: NDArray,
    n_components: Optional[int] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    threshold_color: str = 'red',
    line_color: str = 'teal',
    title: str = None, 
    xlabel: str = None, 
    ylabel: str = None, 
    threshold_label: Optional[str] =None, 
    grid_style: str = ':',
    grid_width: float = 0.5,
    axis_width: float = 2,
    axis_color: str = 'black',
    show_grid: bool = True
) -> plt.Axes:
    """
    Plots the cumulative explained variance ratio by principal components 
    using PCA.
    
    Optionally, a threshold line can be drawn to indicate the desired level 
    of explained variance.

    Parameters
    ----------
    data : np.ndarray
        The input dataset for PCA. Must be 2D (samples x features).
    n_components : int, optional
        Number of principal components to consider. Defaults to min(data.shape).
    threshold : float, optional
        A variance ratio threshold to draw a horizontal line. Defaults to None.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height) in inches. Defaults to (10, 6).
    threshold_color : str, optional
        Color of the threshold line. Defaults to 'red'.
    line_color : str, optional
        Color of the cumulative variance line. Defaults to 'teal'.
    title : str, optional
        Title of the plot. Defaults to 'Cumulative Explained Variance 
        Ratio by Principal Components'.
    xlabel : str, optional
        X-axis label. Defaults to 'Number of Components'.
    ylabel : str, optional
        Y-axis label. Defaults to 'Cumulative Explained Variance Ratio'.
    threshold_label : str, optional
        Label for the threshold line. Defaults to 'Variance Threshold'.
    grid_style : str, optional
        Style of the grid lines (lines, dashes, dots, etc.).
        Defaults to ':' (dotted line).
    grid_width : float, optional
        Width of the grid lines. Defaults to 0.5.
    axis_width : float, optional
        Width of the axes' spines. Defaults to 2.
    axis_color : str, optional
        Color of the axes' spines. Defaults to 'black'.
    show_grid : bool, optional
        If True, display grid lines on the plot. Defaults to True.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the plot for further customization.
    
    Raises
    ------
    ValueError
        If 'data' is not a 2D array.
        If 'n_components' is greater than the number of features in 'data'.
        If 'threshold' is not between 0 and 1 when provided.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> ax = plot_cumulative_variance(iris.data)
    >>> ax.set_title('Updated Plot Title')
    >>> plt.show()
    """
    title= title or  'Cumulative Explained Variance Ratio by Principal Components'
    xlabel = xlabel or 'Number of Components',
    ylabel= ylabel or 'Cumulative Explained Variance Ratio'
    threshold_label =threshold_label or 'Variance Threshold'
    
    if data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D array.")
    
    if n_components is None:
        n_components = min(data.shape)
    elif n_components > data.shape[1]:
        raise ValueError("'n_components' cannot be greater than "
                         f"the number of features ({data.shape[1]}).")
    
    if threshold is not None and (threshold < 0 or threshold > 1):
        raise ValueError("'threshold' must be between 0 and 1.")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    # Calculate the cumulative explained variance ratio
    cumulative_variance= np.cumsum(pca.explained_variance_ratio_)
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the cumulative explained variance ratio
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='-', color=line_color,
             label='Cumulative Variance')
    
    # Plot the threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color=threshold_color, linestyle='--',
                   label=f'{threshold_label} ({threshold:.2f})' 
                   if threshold_label else None)
    
    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(cumulative_variance) + 1))
    ax.legend(loc='best')
    
    # Customize the grid
    if show_grid:
        ax.grid(True, linestyle=grid_style, linewidth=grid_width)
    
    # Customize the axes' appearance
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
        spine.set_color(axis_color)
    
    plt.show()
    
    return ax    
def plot_unified_pca(
    components:NDArray,
    Xr: NDArray,
    y: ArrayLike,
    classes: ArrayLike=None,
    markers:List [str]=None, 
    colors: List [str ]=None, 
    **baseplot_kws, 
 ):
    """
    The biplot is the best way to visualize all-in-one following a PCA analysis.
    
    There is an implementation in R but there is no standard implementation
    in Python. 

    Parameters  
    -----------
    components: NDArray, shape (n_components, n_eigenvectors ), 
        the eigenvectors of the PCA. The shape in axis must much the number 
        of component computed using PCA. If the `Xr` shape 1 equals to the 
        shape 0 of the component matrix `components`, it will be transposed 
        to fit `Xr` shape 1.
        
    Xr: NDArray of transformed X. 
        the PCA projected data scores on n-given components.The reduced  
        dimension of train set 'X' with maximum ratio as sorted eigenvectors 
        from first to the last component. 
 
    y: Array-like, 
        the target composing the class labels.
    classes: list or int, 
        class categories or class labels 
    markers: str, 
        Matplotlib list of markers for plotting  classes.
    colors: str, 
        Matplotlib list of colors to customize plots 
        
    baseplot: dict, :class:`gofast.property.BasePlot`. 
        Matplotlib property from `BasePlot` instances. 

    Examples 
    ---------
    >>> from gofast.analysis import nPCA
    >>> from gofast.datasets import fetch_data
    >>> from gofast.plot import  plot_unified_pca, pobj_obj  #  is Baseplot instance 
    >>> X, y = fetch_data ('bagoue pca' )  # fetch pca data 
    >>> pca= nPCA (X, n_components= 2 , return_X= False ) # return PCA object 
    >>> components = pca.components_ [:2, :] # for two components 
    >>> plot_unified_pca ( components , pca.X, y ) # pca.X is the reduced dim X 
    >>> # to change for instance line width (lw) or style (ls) 
    >>> # just use the baseplotobject (pobj_obj)
    
    References 
    -----------
    Originally written by `Serafeim Loukas`_, serafeim.loukas@epfl.ch 
    and was edited to fit the :term:`gofast` package API. 
    
    .. _Serafeim Loukas: https://towardsdatascience.com/...-python-7c274582c37e
    
    """
    #xxxxxxxxx update base plot keyword arguments
    for k  in list(baseplot_kws.keys()): 
        setattr (pobj , k, baseplot_kws[k])
        
    Xr = check_array(
        Xr, 
        to_frame= False, 
        input_name="X reduced 'Xr'"
        )
    components = check_array(
        components, 
        to_frame =False ,
        input_name="PCA components"
        )
    Xr = np.array (Xr); components = np.array (components )
    xs = Xr[:,0] # projection on PC1
    ys = Xr[:,1] # projection on PC2
    
    if Xr.shape[1]==components.shape [0] :
        # i.e components is not transposed 
        # transposed then 
        components = components.T 
    n = components.shape[0] # number of variables
    
    fig = plt.figure(figsize=pobj.fig_size, #(10,8),
               dpi=pobj.fig_dpi #100
               )
    if classes is None: 
        classes = np.unique(y)
    if colors is None:
        # make color based on group
        # to fit length of classes
        colors = make_mpl_properties(
            len(classes))
        
    colors = [colors[c] for c in range(len(classes))]
    if markers is None:
        markers= make_mpl_properties(len(classes), prop='marker')
        
    markers = [markers[m] for m in range(len(classes))]
    
    for s,l in enumerate(classes):
        plt.scatter(xs[y==l],ys[y==l], 
                    color = colors[s], 
                    marker=markers[s]
                    ) 
    for i in range(n):
        # plot as arrows the variable scores 
        # (each variable has a score for PC1 and one for PC2)
        plt.arrow(0, 0, components[i,0], components[i,1], 
                  color = pobj.lc, #'k', 
                  alpha = pobj.alpha, #0.9,
                  linestyle = pobj.ls, # '-',
                  linewidth = pobj.lw, #1.5,
                  overhang=0.2)
        plt.text(components[i,0]* 1.15, components[i,1] * 1.15, 
                 "Var"+str(i+1),
                 color = 'k', 
                 ha = 'center',
                 va = 'center',
                 fontsize= pobj.font_size
                 )
    plt.tick_params(axis ='both', labelsize = pobj.font_size)
    
    plt.xlabel(pobj.xlabel or "PC1",size=pobj.font_size)
    plt.ylabel(pobj.ylabel or "PC2",size=pobj.font_size)
    limx= int(xs.max()) + 1
    limy= int(ys.max()) + 1
    plt.xlim([-limx,limx])
    plt.ylim([-limy,limy])
    plt.grid()
    plt.tick_params(axis='both',
                    which='both', 
                    labelsize=pobj.font_size
                    )
    
    pobj.save(fig)
    # if self.savefig is not None: 
    #     savefigure (plt, self.savefig, dpi = self.fig_dpi )