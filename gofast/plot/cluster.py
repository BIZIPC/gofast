# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
The `cluster` module provides visualization tools for cluster analysis. 
It includes functions for plotting silhouettes, dendrograms, cluster comparisons,
Voronoi diagrams, and more, aiding in the evaluation and interpretation of 
clustering results.
"""

from __future__ import annotations 
import itertools 
import warnings 

import numpy as np 
import pandas as pd 
import matplotlib as mpl
from matplotlib import cm 
from matplotlib.axes import Axes
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy.stats as spstats
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.spatial import Voronoi, voronoi_plot_2d

from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples
  
from ..api.types import NDArray, ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional
from ..tools.coreutils import is_iterable, to_numeric_dtypes 
from ..tools.mathext import linkage_matrix 
from ..tools.validator import check_X_y, check_array, check_y 
from ..tools.validator import validate_positive_integer, is_frame
from .utils import _get_xticks_formatage, make_mpl_properties, savefigure 

__all__=[
    'plot_silhouette',
    'plot_silhouette2',
    'plot_dendrogram',
    'plot_dendroheat',
    'plot_dendrogram2',
    'plot_clusters',
    'plot_elbow',
    'plot_cluster_comparison',
    'plot_voronoi',
]

def plot_dendroheat(
    df: DataFrame | NDArray, 
    columns: List[str] = None, 
    labels: Optional[List[str]] = None,
    metric: str = 'euclidean',  
    method: str = 'complete', 
    kind: str = 'design', 
    cmap: str = 'hot_r', 
    fig_size: Tuple[int] = (8, 8), 
    facecolor: str = 'white', 
    **kwd
):
    """
    Attaches dendrogram to a heat map.
    
    Hierarchical dendrograms are often used in combination with a heat map,
    allowing us to represent individual values in a data array or matrix
    containing our training examples with a color code.
    
    Parameters 
    ----------
    df : DataFrame or NDArray of shape (n_samples, n_features)
        DataFrame or ndarray. If array is given, must specify the column names
        to match the array shape.
        
    columns : list of str, optional
        List of labels to name each column of arrays of shape 
        (n_samples, n_features). If DataFrame is given, don't need to 
        specify the columns.
        
    labels : list of str, optional
        By default, `labels` is None so the index of the original observation
        is used to label the leaf nodes. Otherwise, this is an n-sized
        sequence, with `n == Z.shape[0] + 1`. The `labels[i]` value is the 
        text to put under the i-th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
        
    metric : str, default 'euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If `X` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, default 'complete'
        The linkage algorithm to use. See the `Linkage Methods` section below
        for full descriptions in :func:`gofast.tools.exmath.linkage_matrix`.
        
    kind : {'squareform', 'condense', 'design'}, default 'design'
        Approach to summing up the linkage matrix. A condensed distance matrix
        is a flat array containing the upper triangular of the distance matrix.
        This is the form that `pdist` returns. Alternatively, a collection of
        `m` observation vectors in `n` dimensions may be passed as an `m` by `n`
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs. The `design` approach uses the complete input 
        example matrix, also called 'design matrix', to yield a correct 
        linkage matrix similar to `squareform` and `condense`.
        
    cmap : str, default 'hot_r'
        Matplotlib colormap.
        
    fig_size : tuple of int, default (8, 8)
        The size of the figure.
        
    facecolor : str, default 'white'
        Matplotlib facecolor.
        
    **kwd : dict
        Additional keyword arguments passed to 
        :func:`scipy.cluster.hierarchy.dendrogram`.
        
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.cluster import plot_dendroheat
    >>> np.random.seed(123)
    >>> variables = ['X', 'Y', 'Z']
    >>> labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    >>> X = np.random.random_sample([5, 3]) * 10
    >>> df = pd.DataFrame(X, columns=variables, index=labels)
    >>> plot_dendroheat(df)
    
    >>> from gofast.datasets import load_bagoue
    >>> X, y = load_bagoue(as_frame=True)
    >>> X = X[['magnitude', 'power', 'sfi']].astype(float)
    >>> plot_dendroheat(X)
    
    Notes
    -----
    This function creates a combined dendrogram and heat map to visualize 
    hierarchical clustering of multivariate data. The data should be organized 
    as a DataFrame or ndarray. The linkage matrix is computed using the 
    specified metric and method.

    The condensed distance matrix can be computed as:
    
    .. math::
        D(i, j) = \\sqrt{\\sum_k (X[i, k] - X[j, k])^2}
    
    where :math:`D(i, j)` is the distance between observations :math:`i` and 
    :math:`j`, and :math:`X` is the input data.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram : Function to create a dendrogram plot.
    scipy.spatial.distance.pdist : Function to compute pairwise distances.

    References
    ----------
    .. [1] Murtagh, F. (1985). Multidimensional Clustering Algorithms. 
       Compstat Lectures 4, Wuerzburg: Physica-Verlag.
    """
    from ..tools.spatialutils import linkage_matrix 
    
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("X and columns must be consistent,"
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df, columns = columns )
        
    # create a new figure object  and define x axis position 
    # and y poaition , width, heigh of the dendrogram via the  
    # add_axes attributes. Furthermore, we rotate the dengrogram
    # to 90 degree counter-clockwise. 
    fig = plt.figure (figsize = fig_size , facecolor = facecolor )
    axd = fig.add_axes ([.09, .1, .2, .6 ])
    
    row_cluster = linkage_matrix(df = df, metric= metric, 
                                 method =method , kind = kind ,  
                                 )
    orient ='left' # use orientation 'right for matplotlib version < v1.5.1
    mpl_version = mpl.__version__.split('.')
    if mpl_version [0] =='1' : 
        if mpl_version [1] =='5' : 
            if float(mpl_version[2]) < 1. :
                orient = 'right'
                
    r = dendrogram(row_cluster , orientation= orient,  **kwd )
    # 2. reorder the data in our initial dataframe according 
    # to the clustering label that can be accessed by a dendrogram 
    # which is essentially a Python dictionnary via a key leaves 
    df_rowclust = df.iloc [r['leaves'][::-1]] if hasattr(
        df, 'columns') else df  [r['leaves'][::-1]]
    
    # 3. construct the heatmap from the reordered dataframe and position 
    # in the next ro the dendrogram 
    axm = fig.add_axes ([.23, .1, .63, .6]) #.6 # [.23, .1, .2, .6]
    cax = axm.matshow (df_rowclust , 
                       interpolation = 'nearest' , 
                       cmap=cmap, 
                       )
    #4.  modify the asteric  of the dendogram  by removing the axis 
    # ticks and hiding the axis spines. Also we add a color bar and 
    # assign the feature and data record names to names x and y axis  
    # tick lables, respectively 
    axd.set_xticks ([]) # set ticks invisible 
    axd.set_yticks ([])
    for i in axd.spines.values () : 
        i.set_visible (False) 
        
    fig.colorbar(cax )
    xticks_loc = list(axm.get_xticks())
    yticks_loc = list(axm.get_yticks())

    df_rowclust_cols = df_rowclust.columns if hasattr (
        df_rowclust , 'columns') else [f"{i+1}" for i in range (df.shape[1])]
    axm.xaxis.set_major_locator(mticker.FixedLocator(xticks_loc))
    axm.xaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_cols)))
    
    df_rowclust_index = df_rowclust.index if hasattr(
        df_rowclust , 'columns') else [f"{i}" for i in range (df.shape[0])]
    axm.yaxis.set_major_locator(mticker.FixedLocator(yticks_loc))
    axm.yaxis.set_major_formatter(mticker.FixedFormatter(
        [''] + list (df_rowclust_index)))
    
    plt.show () 

def plot_elbow(
    X, n_clusters: int = 8, n_init: int = 10, max_iter: int = 300, 
    random_state: int = 42, fig_size: Tuple[int, int] = (10, 4), 
    marker: str = 'o', savefig: Optional[str] = None, **kwd
): 
    """
    Plot elbow method to find the optimal number of clusters, `k`, for a 
    given data.
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. The data will be converted to C 
        ordering, causing a memory copy if the given data is not 
        C-contiguous. If a sparse matrix is passed, a copy will be made if 
        it's not in CSR format.
        
    n_clusters : int, default=8
        The maximum number of clusters to consider for the elbow method.
        
    n_init : int, default=10
        Number of times the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of 
        `n_init` consecutive runs in terms of inertia.
        
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
        
    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use 
        an int to make the randomness deterministic.
        
    fig_size : tuple of int, default=(10, 4)
        The size of the figure.
        
    marker : str, default='o'
        Cluster marker point.
        
    savefig : str, optional
        The path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure`.
        
    **kwd : dict
        Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`.
        
    Returns 
    -------
    ax : matplotlib.pyplot.Axes
        The matplotlib Axes object for the plot.
        
    Examples
    --------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.plot.cluster import plot_elbow 
    >>> res_gamma = load_hlogs().frame[['resistivity', 'gamma_gamma']]  
    >>> plot_elbow(res_gamma, n_clusters=11)
    
    Notes
    -----
    The elbow method helps determine the optimal number of clusters for 
    k-means clustering by plotting the sum of squared distances from each 
    point to its assigned cluster center (inertia) against the number of 
    clusters. The point where the inertia starts to decrease more slowly 
    (forming an "elbow") indicates the optimal number of clusters.

    The inertia for k-means clustering is defined as:

    .. math::
        \text{Inertia} = \sum_{i=1}^{n} \min_{\mu_j \in C} (||x_i - \mu_j||^2)
    
    where :math:`x_i` is a data point, :math:`\mu_j` is the centroid of 
    cluster :math:`C`, and :math:`||x_i - \mu_j||^2` is the Euclidean 
    distance between :math:`x_i` and :math:`\mu_j`.

    See Also
    --------
    sklearn.cluster.KMeans : K-means clustering algorithm.
    
    References
    ----------
    .. [1] Jain, A. K. (2010). Data clustering: 50 years beyond K-means. 
       Pattern recognition letters, 31(8), 651-666.
    """
    distorsions =[] 
    n_clusters = validate_positive_integer(n_clusters, "n_clusters")
    for i in range (1, n_clusters ): 
        km =KMeans (n_clusters =i , init= 'k-means++', 
                    n_init=n_init , max_iter=max_iter, 
                    random_state =random_state 
                    )
        km.fit(X) 
        distorsions.append(km.inertia_) 
            
    ax = _plot_elbow (distorsions, n_clusters =n_clusters,fig_size = fig_size ,
                      marker =marker , savefig =savefig, **kwd) 

    return ax 
    
def _plot_elbow (distorsions: list  , n_clusters:int ,fig_size = (10 , 4 ),  
               marker='o', savefig =None, **kwd): 
    """ Plot the optimal number of cluster, k', for a given class 
    
    :param distorsions: list - list of values withing the sum-squared-error 
        (SSE) also called  `inertia_` in sckit-learn. 
    
    :param n_clusters: number of clusters. where k starts and end. 
    
    :returns: ax: Matplotlib.pyplot axes objects 
    
    :Example: 
    >>> import numpy as np 
    >>> from sklearn.cluster import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.tools.utils import plot_elbow
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> # compute distorsiosn for KMeans range 
    >>> distorsions =[] ; n_clusters = 11
    >>> for i in range (1, n_clusters ): 
            km =KMeans (n_clusters =i , 
                        init= 'k-means++', 
                        n_init=10 , 
                        max_iter=300, 
                        random_state =0 
                        )
            km.fit(X) 
            distorsions.append(km.inertia_) 
    >>> plot_elbow (distorsions, n_clusters =n_clusters)
        
    """
    fig, ax = plt.subplots ( nrows=1 , ncols =1 , figsize = fig_size ) 
    
    ax.plot (range (1, n_clusters), distorsions , marker = marker, 
              **kwd )
    plt.xlabel ("Number of clusters") 
    plt.ylabel ("Distorsion")
    plt.tight_layout()
    
    if savefig is not None: 
        savefigure(fig, savefig )
    plt.show() if savefig is None else plt.close () 
    
    return ax 

def plot_clusters(
    n_clusters, 
    X, 
    y_pred, 
    cluster_centers=None, 
    savefig=None
):
    """
    Visualize the clusters identified by a clustering algorithm such as k-means
    in the dataset.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to visualize.
        
    X : ndarray of shape (n_samples, n_features)
        Data containing the features, expected to be two-dimensional.
        
    y_pred : array-like of shape (n_samples,)
        Array containing the predicted class labels.
        
    cluster_centers : ndarray, optional
        Array containing the coordinates of the centroids or similar points
        with continuous features. If not provided, centroids will not be plotted.
        
    savefig : str, optional
        The path where the figure should be saved. If not specified, the figure
        is displayed using `plt.show()`.

    Examples
    --------
    Example of plotting clusters with centroids:

    >>> from sklearn.cluster import KMeans
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from gofast.plot.cluster import plot_clusters
    >>> from gofast.datasets import fetch_data
    >>> data = fetch_data('hlogs').frame
    >>> data = data[['resistivity', 'gamma_gamma']]
    >>> scaler = MinMaxScaler()
    >>> X_scaled = scaler.fit_transform(data)
    >>> kmeans = KMeans(n_clusters=3, init='random')
    >>> y_pred = kmeans.fit_predict(X_scaled)
    >>> plot_clusters(3, X_scaled, y_pred, kmeans.cluster_centers_)

    Notes
    -----
    The function uses Matplotlib to create scatter plots of the clusters. 
    Each cluster is plotted with a different color and marker. Centroids, if
    provided, are plotted as red stars. This function is typically used after
    performing clustering to visualize the distribution and separation of clusters.

    The mathematical formulation for the cluster centroid is:

    .. math::
        \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
    
    where :math:`\mu_j` is the centroid of cluster :math:`C_j` and 
    :math:`x_i` are the data points in cluster :math:`C_j`.

    See Also
    --------
    sklearn.cluster.KMeans : K-means clustering algorithm.
    matplotlib.pyplot.scatter : Scatter plot in Matplotlib.

    References
    ----------
    .. [1] MacQueen, J. (1967). Some methods for classification and analysis 
       of multivariate observations. In Proceedings of the fifth Berkeley 
       symposium on mathematical statistics and probability (Vol. 1, No. 14, 
       pp. 281-297).
    """
    n_clusters = validate_positive_integer(n_clusters, "n_clusters")

    X, y_pred = check_X_y(
        X, 
        y_pred, 
        )

    if len(X.shape )!=2 or X.shape[1]==1: 
        ndim = 1 if X.shape[1] ==1 else np.ndim (X )
        raise ValueError(
            f"X is expected to be a two dimensional data. Got {ndim}!")
    # for consistency , convert y to array    
    y_pred = np.array(y_pred)
    
    colors = make_mpl_properties(n_clusters)
    markers = make_mpl_properties(n_clusters, 'markers')
    for n in range (n_clusters):
        plt.scatter (X[y_pred ==n, 0], 
                     X[y_pred ==n , 1],  
                     s= 50 , c= colors [n ], 
                     marker=markers [n], 
                     edgecolors=None if markers [n] =='x' else 'black', 
                     label = f'Cluster {n +1}'
                     ) 
    if cluster_centers is not None: 
        cluster_centers = np.array (cluster_centers)
        plt.scatter (cluster_centers[:, 0 ], 
                     cluster_centers [:, 1], 
                     s= 250. , marker ='*', 
                     c='red', edgecolors='black', 
                     label='centroids' 
                     ) 
    plt.legend (scatterpoints =1 ) 
    plt.grid() 
    plt.tight_layout() 
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 

def plot_dendrogram2(
    X: ArrayLike, 
    *ybounds, 
    fig_size=(12, 5), 
    savefig=None,  
    **kws
    ):
    """
    Quick plot dendrogram using the ward clustering function from Scipy.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Array of features to be used for generating the dendrogram.
        
    ybounds : int, optional
        Integer values to draw horizontal cluster lines at specified y-coordinates,
        indicating the number of clusters. If no values are provided, no horizontal
        lines are drawn.
        
    fig_size : tuple of int, default (12, 5)
        The size of the matplotlib figure given as (width, height).
        
    savefig : str, optional
        The file path or a file-like object where the figure will be saved. If not
        specified, the figure is displayed using `plt.show()`.
        
    kws : dict
        Additional keyword arguments passed to `scipy.cluster.hierarchy.dendrogram`.

    Examples
    --------
    Using a dataset:

    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.cluster import plot_dendrogram2
    >>> X, _ = fetch_data('Bagoue analysed')  # data is already scaled 
    >>> data = X[['power', 'magnitude']]
    >>> plot_dendrogram2(data)

    Adding horizontal cluster lines:

    >>> plot_dendrogram2(data, 20, 30)

    Notes
    -----
    The function uses `scipy.cluster.hierarchy.ward` to perform the clustering
    and `scipy.cluster.hierarchy.dendrogram` to generate the dendrogram. If
    `ybounds` is provided, it should be an even number of integer values 
    indicating pairs of y-coordinates to draw horizontal lines for cluster cuts.

    The ward clustering minimizes the variance of the clusters being merged:

    .. math::
        d(u, v) = \\sqrt{\\frac{2|u||v|}{|u| + |v|}} \\|c_u - c_v\\|

    where :math:`d(u, v)` is the distance between clusters :math:`u` and :math:`v`,
    :math:`|u|` and :math:`|v|` are the sizes of the clusters, and :math:`c_u` and
    :math:`c_v` are the centroids of the clusters.

    See Also
    --------
    scipy.cluster.hierarchy.ward : Perform ward clustering.
    scipy.cluster.hierarchy.dendrogram : Generate a dendrogram plot.

    References
    ----------
    .. [1] Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical
       clustering: an overview. Wiley Interdisciplinary Reviews: Data Mining
       and Knowledge Discovery, 2(1), 86-97.
    """
    
    # assert ybounds agument if given
    msg =(". Note that the bounds in y-axis are the y-coordinates for"
          " horizontal lines regarding to the number of clusters that"
          " might be cutted.")
    try : 
        ybounds = [ int (a) for a in ybounds ] 
    except Exception as typerror: 
        raise TypeError  (str(typerror) + msg)
    else : 
        if len(ybounds)==0 : ybounds = None 
    # the scipy ward function returns 
    # an array that specifies the 
    # distance bridged when performed 
    # agglomerate clustering
    linkage_array = ward(X) 
    
    # plot the dendrogram for the linkage array 
    # containing the distances between clusters 
    dendrogram( linkage_array , **kws )
    
    # mark the cuts on the tree that signify two or three clusters
    # change the gca figsize 
    plt.rcParams["figure.figsize"] = fig_size
    ax= plt.gca () 
  
    if ybounds is not None: 
        if not is_iterable(ybounds): 
            ybounds =[ybounds] 
        if len(ybounds) <=1 : 
            warnings.warn(f"axis y bound might be greater than {len(ybounds)}")
        else : 
            # split ybound into sublist of pair (x, y) coordinates
            nsplits = len(ybounds)//2 
            len_splits = [ 2 for i in range (nsplits)]
            # compose the pir list (x,y )
            itb = iter (ybounds)
            ybounds = [list(itertools.islice (itb, it)) for it in len_splits]
            bounds = ax.get_xbound () 
            for i , ( x, y)  in enumerate (ybounds)  : 
                ax.plot(bounds, [x, y], '--', c='k') 
                ax.text ( bounds [1], y , f"cluster {i +1:02}",
                         va='center', 
                         fontdict ={'size': 15}
                         )
    # get xticks and format labels
    xticks_loc = list(ax.get_xticks())
    _get_xticks_formatage(ax, xticks_loc, space =14 )
    
    plt.xlabel ("Sample index ")
    plt.ylabel ("Cluster distance")
            
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 

def plot_silhouette (
    X:NDArray |DataFrame, 
    labels:ArrayLike=None, 
    prefit:bool=True, 
    n_clusters:int =3,  
    n_init: int=10 , 
    max_iter:int=300 , 
    random_state:int=None , 
    tol:float=1e4 , 
    metric:str='euclidean', 
    **kwd 
 ): 
    r"""
    quantifies the quality  of clustering samples. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
        
    labels : array-like 1d of shape (n_samples,)
        Label values for each sample.
         
    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.
        
    prefit : bool, default=False
        Whether a prefit `labels` is expected to be passed into the function
        directly or not.
        If `True`, `labels` must be a fit predicted values target.
        If `False`, `labels` is fitted and updated from `X` by calling
        `fit_predict` methods. Any other values passed to `labels` is 
        discarded.
         
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
    
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Note
    -----
    The sihouette coefficient is bound between -1 and 1 

    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
    
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
                
    Examples 
    --------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.plot.cluster  import plot_silhouette
    >>> # use resistivity and gamma for this demo
    >>> X_res_gamma = load_hlogs().frame[['resistivity', 'gamma_gamma']]  
    
    (1) Plot silhouette with 'prefit' set to 'False' 
    >>> plot_silhouette (X_res_gamma, prefit =False)
    
    """
    if  ( 
        not prefit 
        and labels is not None
        ): 
        warnings.warn("'labels' is given while 'prefix' is 'False'"
                      "'prefit' will set to 'True'")
        prefit=True 
        
    if labels is not None: 
        if not hasattr (labels, '__array__'): 
            raise TypeError( "Labels (target 'y') expects an array-like: "
                            f"{type(labels).__name__!r}")
        labels=check_y (
            labels, 
            to_frame =True, 
            )
        if len(labels)!=len(X): 
            raise TypeError("X and labels must have a consistency size."
                            f"{len(X)} and {len(labels)} respectively.")
            
    if prefit and labels is None: 
        raise TypeError ("Labels can not be None, while 'prefit' is 'True'"
                         " Turn 'prefit' to 'False' or provide the labels "
                         "instead.")
    if not prefit : 
        km= KMeans (n_clusters =n_clusters , 
                    init='k-means++', 
                    n_init =n_init , 
                    max_iter = max_iter , 
                    tol=tol, 
                    random_state =random_state
                        ) 
        labels = km.fit_predict(X ) 
        
    return _plot_silhouette(X, labels, metric = metric , **kwd)
    
    
def _plot_silhouette (X, labels, metric ='euclidean', **kwds ):
    r"""Plot quantifying the quality  of clustering silhouette 
    
    Parameters 
    ---------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Examples
    ---------
    >>> import numpy as np 
    >>> from gofast.exlib.sklearn import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.plot.cluster  import plot_silhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plot_silhouette (X, y_km)
  
    See also 
    ---------
    gofast.tools.plotutils.plot_silhouette: Plot naive silhouette 
    
    Notes
    ------ 
    
    Silhouette is used as graphical tools,  to plot a measure how tighly is  
    grouped the examples of the clusters are.  To calculate the silhouette 
    coefficient, three steps is allows: 
        
    * calculate the **cluster cohesion**, :math:`a(i)`, as the average 
        distance between examples, :math:`x^{(i)}`, and all the others 
        points
    * calculate the **cluster separation**, :math:`b^{(i)}` from the next 
        average distance between the example , :math:`x^{(i)}` amd all 
        the example of nearest cluster 
    * calculate the silhouette, :math:`s^{(i)}`, as the difference between 
        the cluster cohesion and separation divided by the greater of the 
        two, as shown here: 
            
        .. math:: 
            
            s^{(i)}=\frac{b^{(i)} - a^{(i)}}{max {{b^{(i)},a^{(i)} }}}
    
    Note that the sihouette coefficient is bound between -1 and 1 
    
    """
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(X, labels= labels, metric = metric ,
                                         **kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =cm.jet (float(i)/n_clusters )
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                 height =1.0 , 
                 edgecolor ='none', 
                 color =color, 
                 )
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline (silhouette_avg, 
                 color='red', 
                 linestyle ='--'
                 )
    plt.yticks(yticks, cluster_labels +1 ) 
    plt.ylabel ("Cluster") 
    plt.xlabel ("Silhouette coefficient")
    plt.tight_layout()

    plt.show() 

def plot_silhouette2(
    X, labels, 
    metric='euclidean',
    savefig=None, 
    **kwds 
):
    r"""
    Plot quantifying the quality of clustering silhouette.

    Parameters 
    ----------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    savefig : str, optional
        The path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure`.
        
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
    Examples
    --------
    >>> import numpy as np 
    >>> from sklearn.clusters import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.plot.cluster import plot_silhouette2
    >>> d = load_iris()
    >>> X = d.data[:, 0][:, np.newaxis]  # take the first axis 
    >>> km = KMeans(n_clusters=3, init='k-means++', n_init=10, 
                    max_iter=300, tol=1e-4, random_state=0)
    >>> y_km = km.fit_predict(X) 
    >>> plot_silhouette2(X, y_km)

    Notes
    -----
    The silhouette score measures how similar a point is to its own cluster 
    compared to other clusters. It ranges from -1 to 1, where higher values 
    indicate that the point is well matched to its own cluster and poorly 
    matched to neighboring clusters.

    The silhouette coefficient for a sample is defined as:

    .. math::
        s = \frac{b - a}{\max(a, b)}

    where :math:`a` is the mean intra-cluster distance and :math:`b` is the 
    mean nearest-cluster distance for each sample.

    See Also
    --------
    sklearn.metrics.silhouette_samples : Compute the silhouette scores for 
        each sample.
    sklearn.metrics.silhouette_score : Compute the mean silhouette score for 
        all samples.

    References
    ----------
    .. [1] Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the 
       interpretation and validation of cluster analysis. Journal of 
       computational and applied mathematics, 20, 53-65.
    """
    
    X, labels = check_X_y(
        X, 
        labels, 
        to_frame= True, 
        )
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(
        X, labels= labels, metric = metric ,**kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =mpl.cm.jet (float(i)/n_clusters )
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                 height =1.0 , 
                 edgecolor ='none', 
                 color =color, 
                 )
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline (silhouette_avg, 
                 color='red', 
                 linestyle ='--'
                 )
    plt.yticks(yticks, cluster_labels +1 ) 
    plt.ylabel ("Cluster") 
    plt.xlabel ("Silhouette coefficient")
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 

def plot_dendrogram(
    df: DataFrame, 
    columns: List[str] = None, 
    labels: ArrayLike = None,
    metric: str = 'euclidean',  
    method: str = 'complete', 
    kind: str = 'design',
    return_r: bool = False, 
    verbose: bool = False, 
    **kwd
):
    r"""
    Visualizes the linkage matrix in the results of dendrogram.
    
    Note that the categorical features, if exist in the dataframe, should 
    automatically be discarded.
    
    Parameters 
    ----------
    df : DataFrame or NDArray of shape (n_samples, n_features)
        DataFrame or ndarray. If array is given, must specify the column names
        to match the array shape.
        
    columns : list of str, optional
        List of labels to name each column of arrays of shape 
        (n_samples, n_features). If DataFrame is given, columns don't need 
        to be specified.
        
    kind : {'squareform', 'condense', 'design'}, default='design'
        Kind of approach to summing up the linkage matrix. A condensed 
        distance matrix is a flat array containing the upper triangular 
        part of the distance matrix. This is the form that ``pdist`` returns. 
        Alternatively, a collection of :math:`m` observation vectors in 
        :math:`n` dimensions may be passed as an :math:`m` by :math:`n` array. 
        All elements of the condensed distance matrix must be finite, i.e., 
        no NaNs or infs. The `design` approach uses the complete input example 
        matrix (also called 'design matrix') to yield a correct linkage matrix 
        similar to `squareform` and `condense`.
        
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a 
        feature array. If metric is a string, it must be one of the options 
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`. If 
        ``X`` is the distance array itself, use "precomputed" as the metric. 
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, default='complete'
        The linkage algorithm to use. See the `Linkage Methods` section below 
        for full descriptions in :func:`gofast.tools.mathext.linkage_matrix`.
        
    labels : ndarray, optional
        By default, `labels` is None so the index of the original observation 
        is used to label the leaf nodes. Otherwise, this is an :math:`n`-sized 
        sequence, with `n == Z.shape[0] + 1`. The `labels[i]` value is the 
        text to put under the :math:`i`-th leaf node only if it corresponds to 
        an original observation and not a non-singleton cluster.
        
    return_r : bool, default=False
        Return the r-dictionary if set to `True`, otherwise returns nothing.
    
    verbose : bool, default=False
        If `True`, outputs messages of the names of categorical features 
        dropped.
    
    **kwd : dict
        Additional keyword arguments passed to 
        :func:`scipy.cluster.hierarchy.dendrogram`.
        
    Returns
    -------
    r : dict
        A dictionary of data structures computed to render the dendrogram. 
        It has the following keys:

        ``'color_list'``:
            A list of color names. The k'th element represents the color of 
            the k'th link.

        ``'icoord'`` and ``'dcoord'``:
            Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]`` 
            where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]`` 
            where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is 
            ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

        ``'ivl'``:
            A list of labels corresponding to the leaf nodes.

        ``'leaves'``:
            For each i, ``H[i] == j``, cluster node ``j`` appears in position 
            ``i`` in the left-to-right traversal of the leaves, where 
            :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, 
            the ``i``-th leaf node corresponds to an original observation. 
            Otherwise, it corresponds to a non-singleton cluster.

        ``'leaves_color_list'``:
            A list of color names. The k'th element represents the color of 
            the k'th leaf.
          
    Examples 
    --------
    >>> from gofast.datasets import load_iris 
    >>> from gofast.plot.cluster import plot_dendrogram
    >>> data = load_iris() 
    >>> X = data.data[:, :2] 
    >>> plot_dendrogram(X, columns=['X1', 'X2'])

    Notes
    -----
    This function generates a dendrogram to visualize the hierarchical 
    clustering of multivariate data. The data should be a DataFrame or 
    ndarray, and the function will automatically discard any categorical 
    features.

    The linkage matrix is computed using the specified metric and method, 
    which defines the distances between clusters.

    The distance between two clusters :math:`u` and :math:`v` can be 
    computed as:

    .. math::
        d(u, v) = \min \{ d(x, y) : x \in u, y \in v \}

    where :math:`d(x, y)` is the Euclidean distance between points :math:`x` 
    and :math:`y`.

    See Also
    --------
    scipy.cluster.hierarchy.dendrogram : Generate a dendrogram plot.
    scipy.spatial.distance.pdist : Compute pairwise distances.

    References
    ----------
    .. [1] Murtagh, F. (1985). Multidimensional Clustering Algorithms. 
       Compstat Lectures 4, Wuerzburg: Physica-Verlag.
    """
    if hasattr (df, 'columns') and columns is not None: 
        df = df [columns ]
        
    df = to_numeric_dtypes(df, pop_cat_features= True, verbose =verbose )
    
    df=check_array (
        df, 
        input_name="Data 'df' ", 
        to_frame =True, 
        )

    kind:str = kind or 'design'
    row_cluster = linkage_matrix(df = df, columns = columns, metric= metric, 
                                 method =method , kind = kind ,
                                 )
    #make dendogram black (1/2)
    # set_link_color_palette(['black']) 
    r= dendrogram(row_cluster, labels= labels  , 
                           # make dendogram colors (2/2)
                           # color_threshold= np.inf,  
                           **kwd)
    plt.tight_layout()
    plt.ylabel ('Euclidian distance')
    plt.show ()
    
    return r if return_r else None 

def plot_cluster_comparison(
    data,  
    cluster_col, 
    class_col, 
    figsize=(10, 6), 
    palette="RdYlGn", 
    title="Clusters versus Prior Classes"
):
    """
    Plots a comparison of clusters versus prior class distributions 
    using a heatmap.

    Parameters
    ----------
    data : DataFrame
        A pandas DataFrame containing at least two columns for clustering 
        and class comparison.
        
    cluster_col : str
        The name of the column in `data` that contains cluster labels.
        
    class_col : str
        The name of the column in `data` that contains prior class labels.
        
    figsize : tuple, optional
        The size of the figure to be created (width, height in inches). 
        Default is (10, 6).
        
    palette : str, optional
        The color palette to use for differentiating Pearson residuals in the 
        heatmap. Default is "RdYlGn".
        
    title : str, optional
        The title of the plot. Default is "Clusters versus Prior Classes".

    Returns
    -------
    ax : matplotlib.axes.Axes
        A matplotlib Axes object with the heatmap.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.cluster import plot_cluster_comparison
    >>> np.random.seed(0)
    >>> sample_data = pd.DataFrame({
    ...     'Cluster': np.random.choice(['1', '4', '5'], size=100),
    ...     'PriorClass': np.random.choice(['1', '2', '3'], size=100)
    ... })
    >>> plot_cluster_comparison(sample_data, 'Cluster', 'PriorClass')
    >>> plt.show()

    Notes
    -----
    This function creates a heatmap to visualize the relationship between 
    clusters identified by a clustering algorithm and prior classes in the 
    dataset. It uses a contingency table to compute Pearson residuals, which 
    are plotted in the heatmap to highlight the deviations from expected 
    counts under the assumption of independence.

    The Pearson residual for cell :math:`(i, j)` in the contingency table is 
    calculated as:

    .. math::
        R_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}

    where :math:`O_{ij}` is the observed count and :math:`E_{ij}` is the 
    expected count under the null hypothesis of independence.

    See Also
    --------
    pandas.crosstab : Compute a simple cross-tabulation of two (or more) factors.
    scipy.stats.chi2_contingency : Chi-square test of independence of variables 
        in a contingency table.

    References
    ----------
    .. [1] Agresti, A. (2002). Categorical Data Analysis. John Wiley & Sons.
    """
    is_frame(data, df_only= True, raise_exception= True, objname="Data") 

    # Create a contingency table
    contingency_table = pd.crosstab(data[cluster_col], data[class_col])

    # Calculate Pearson residuals
    chi2, _, _, expected = spstats.chi2_contingency(contingency_table)
    pearson_residuals = (contingency_table - expected) / np.sqrt(expected)

    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(pearson_residuals, annot=True, fmt=".2f", cmap=palette,
                     cbar_kws={'label': 'Pearson residuals'})
    ax.set_title(title)
    ax.set_xlabel(class_col)
    ax.set_ylabel(cluster_col)

    # Display p-value
    p_value = spstats.chi2.sf(chi2, (contingency_table.shape[0] - 1) 
                                  * (contingency_table.shape[1] - 1))
    plt.text(0.95, 0.05, f'p-value\n{p_value:.2e}', horizontalalignment='right', 
             verticalalignment='bottom', 
             transform=ax.transAxes, color='black', bbox=dict(
                 facecolor='white', alpha=0.5))

    return ax

def plot_voronoi(
    X: ArrayLike, 
    y: ArrayLike, *, 
    cluster_centers: ArrayLike,
    ax: Optional[Axes] = None,
    show_vertices: bool = False, 
    line_colors: str = 'k',
    line_width: float = 1.,
    line_alpha: float = 1.,   
    fig_size: Tuple[int, int] = (7, 7), 
    fig_title: str = ''
) -> Axes:
    """
    Plots the Voronoi diagram of k-means clusters overlaid with the data points,
    highlighting the regions closest to each cluster center.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n_samples, n_features) containing the dataset's features.
        
    y : np.ndarray
        1D array of shape (n_samples,) containing the target or cluster labels.
        
    cluster_centers : np.ndarray
        2D array of shape (n_clusters, n_features) containing the coordinates of
        the cluster centers, typically obtained from a clustering algorithm like KMeans.
        
    ax : matplotlib.axes.Axes, optional
        A Matplotlib Axes instance on which the plot will be drawn. If None, a new 
        Axes instance is created with the specified `fig_size`.
        
    show_vertices : bool, default False
        Whether to show the Voronoi vertices on the plot.
        
    line_colors : str, default 'k'
        Color of the Voronoi boundary lines.
        
    line_width : float, default 1.0
        Width of the Voronoi boundary lines.
        
    line_alpha : float, default 1.0
        Transparency of the Voronoi boundary lines.
        
    fig_size : Tuple[int, int], default (7, 7)
        Size of the figure in which the axes are contained, used only if `ax` is None.
        
    fig_title : str, default ''
        Title of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot for further tweaking or displaying.

    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import KMeans
    >>> from gofast.plot.cluster import plot_voronoi
    >>> X, y = make_moons(n_samples=2000, noise=0.2)
    >>> km = KMeans(n_clusters=2, init='random').fit(X)
    >>> plot_voronoi(X, y, cluster_centers=km.cluster_centers_)

    Notes
    -----
    The function plots a Voronoi diagram to visualize the partitioning of 
    the feature space based on the provided cluster centers. The Voronoi 
    diagram divides the plane into regions closest to each cluster center.

    The Voronoi diagram is constructed using the following mathematical 
    formulation:

    .. math::
        V_i = \{ x \mid \|x - c_i\| \leq \|x - c_j\| \text{ for all } j \neq i \}

    where :math:`V_i` is the Voronoi cell for the cluster center :math:`c_i` 
    and :math:`\|x - c_i\|` represents the Euclidean distance between a point 
    :math:`x` and the cluster center :math:`c_i`.

    See Also
    --------
    scipy.spatial.Voronoi : Compute Voronoi diagram.
    scipy.spatial.voronoi_plot_2d : Plot Voronoi diagram.

    References
    ----------
    .. [1] Okabe, A., Boots, B., Sugihara, K., & Chiu, S. N. (2000). Spatial 
       tessellations: concepts and applications of Voronoi diagrams (Vol. 501). 
       John Wiley & Sons.
    """
    
    X, y = check_X_y(X, y, )
    cluster_centers = check_array(cluster_centers )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', alpha=0.2, 
               label = 'Voronoi plot')
    vor = Voronoi(cluster_centers)
    voronoi_plot_2d(vor, ax=ax, show_vertices=show_vertices, 
                    alpha=0.5, 
                    line_colors=line_colors,
                    line_width=line_width ,
                    line_alpha=line_alpha,  
                    )
    #ax.legend() 
    ax.set_title (fig_title , fontsize=20)
    #fig.suptitle(fig_title, fontsize=20) 
    return ax 
 
    