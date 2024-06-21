# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

from ..api.types import Optional, Union, Tuple
from ..api.types import DataFrame, ArrayLike, Array1D, Series
from ..tools.validator import assert_xy_in
from ..tools.validator import _is_arraylike_1d, is_frame 
from ..tools.coreutils import to_series_if 
from ..tools.coreutils import get_colors_and_alphas
from ..tools.funcutils import make_data_dynamic

__all__=[ 
    "correlation","perform_linear_regression","perform_kmeans_clustering",
    "perform_spectral_clustering","mds_similarity"
    ]

def correlation(
   x: Optional[Union[str, Array1D, Series]] = None,  
   y: Optional[Union[str, Array1D, Series]] = None, 
   data: Optional[DataFrame] = None,
   method: str = 'pearson', 
   view: bool = False, 
   plot_type: Optional[str] = None, 
   cmap: str = 'viridis', 
   fig_size: Optional[Tuple[int, int]] = None, 
   **kws):
    """
    Computes and optionally visualizes the correlation between two datasets 
    or within a DataFrame. If both `x` and `y` are provided, calculates the 
    pairwise correlation. If only `x` is provided as a DataFrame and `y` is None,
    computes the correlation matrix for `x`.

    Parameters
    ----------
    x : Optional[Union[str, Array1D, Series]], default None
        The first dataset for correlation analysis or the entire dataset 
        if `y` is None. Can be an array-like object or a column name in `data`.
    y : Optional[Union[str, Array1D, Series]], default None
        The second dataset for correlation analysis. Can be an array-like object
        or a column name in `data`. If omitted, and `x` is a DataFrame, calculates
        the correlation matrix of `x`.
    data : Optional[DataFrame], default None
        A DataFrame containing `x` and/or `y` columns if they are specified as 
        column names.
    method : str, default 'pearson'
        The method of correlation ('pearson', 'kendall', 'spearman') or a callable
        with the signature (np.ndarray, np.ndarray) -> float.
    view : bool, default False
        If True, visualizes the correlation using a scatter plot (for pairwise 
        correlation) or a heatmap (for correlation matrices).
    plot_type : Optional[str], default None
        Type of plot for visualization when `view` is True. Options are 'scatter'
        for pairwise correlation or None for no visualization.
    cmap : str, default 'viridis'
        Colormap for the visualization plot.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for visualization.
    **kws : dict
        Additional keyword arguments for the correlation computation method.

    Returns
    -------
    correlation_value : float or pd.DataFrame
        The correlation coefficient if `x` and `y` are provided, or the correlation
        matrix if `x` is a DataFrame and `y` is None.

    Examples
    --------
    >>> from gofast.stats.relationships import correlation
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> corr('A', 'B', data=data)
    1.0

    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> corr(x, y)
    1.0

    >>> correlation(data)
    DataFrame showing correlation matrix of 'data'
    
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [5, 4, 3, 2, 1]
    print("Correlation coefficient:", correlation(x, y, view=True))

    >>> # Correlation matrix of a DataFrame
    >>> data = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10), 
                             'C': np.random.rand(10)})
    print("Correlation matrix:\n", correlation(data))
    correlation(data, view=True)
    
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
    >>> correlation('A', 'B', data=data, view=True, plot_type='scatter')
    -1.0

    Correlation matrix of a DataFrame:
    >>> correlation(data=data, view=True)
    Outputs a heatmap of the correlation matrix.

    Pairwise correlation between two array-like datasets:
    >>> x = [1, 2, 3, 4]
    >>> y = [4, 3, 2, 1]
    >>> correlation(x, y, view=True, plot_type='scatter')
    -1.0

    Compute pairwise correlation:
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> correlation(x, y)
    1.0

    Compute and visualize the correlation matrix of a DataFrame:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
    >>> correlation(data=data, view=True)

    Compute pairwise correlation with column names from a DataFrame:
    >>> correlation('A', 'B', data=data, view=True, plot_type='scatter')
    -1.0
    Note
    ----
    The function is designed to provide a versatile interface for correlation 
    analysis, accommodating different types of input and supporting various 
    correlation methods through keyword arguments. It utilizes pandas' `corr()` 
    method for DataFrame inputs, enabling comprehensive correlation analysis within 
    tabular data.
    
    """
    if data is not None: 
        is_frame ( data, df_only=True, raise_exception=True, objname="Data")
    if x is None and y is None:
        if data is not None:
            cor_matrix= data.corr(method=method, **kws)
            if view:
                plt.figure(figsize=fig_size if fig_size else (8, 6))
                sns.heatmap(cor_matrix, annot=True, cmap=cmap)
                plt.title("Correlation Matrix")
                plt.show()
            return cor_matrix
        else:
            raise ValueError("At least one of 'x', 'y', or 'data' must be provided.")
            
    # Validate inputs
    if data is None and (isinstance(x, str) or isinstance(y, str)):
        raise ValueError("Data must be provided when 'x' or 'y' are column names.")

    # Extract series from data if x or y are column names
    x_series = data[x] if isinstance(x, str) and data is not None else x
    y_series = data[y] if isinstance(y, str) and data is not None else y

    # If x is a DataFrame and y is None, compute the correlation matrix
    if isinstance(x_series, pd.DataFrame) and y_series is None:
        correlation_matrix = x_series.corr(method=method, **kws)
    # If x and y are defined, compute pairwise correlation
    elif x_series is not None and y_series is not None:
        x_series = pd.Series(x_series) if not isinstance(x_series, pd.Series) else x_series
        y_series = pd.Series(y_series) if not isinstance(y_series, pd.Series) else y_series
        correlation_value = x_series.corr(y_series, method=method, **kws)
    else:
        raise ValueError("Invalid input: 'x' and/or 'y' must be provided.")

    # Visualization
    if view:
        plt.figure(figsize=fig_size if fig_size else (6, 6))
        if 'correlation_matrix' in locals():
            sns.heatmap(correlation_matrix, annot=True, cmap=cmap)
            plt.title("Correlation Matrix")
        elif plot_type == 'scatter':
            colors, _= get_colors_and_alphas(len(x_series), cmap=cmap  )
            plt.scatter(x_series, y_series, color=colors )
            plt.title(f"Scatter Plot: Correlation = {correlation_value:.2f}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
        plt.show()

    return correlation_matrix if 'correlation_matrix' in locals() else correlation_value

def perform_linear_regression(
    x: Union[ArrayLike, list, str] = None, 
    y: Union[ArrayLike, list, str] = None, 
    data: DataFrame = None,
    view: bool = False, 
    sample_weight=None, 
    as_frame=False, 
    plot_type: str = 'scatter_line', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
) -> Tuple[LinearRegression, ArrayLike, float]:
    """
    Performs linear regression analysis between an independent variable (x) and
    a dependent variable (y), returning the fitted model, its coefficients,
    and intercept.

    Linear regression is modeled as:

    .. math::
        y = X\beta + \epsilon
    
    where:
    
    - :math:`y` is the dependent variable,
    - :math:`X` is the independent variable(s),
    - :math:`\beta` is the coefficient(s) of the model,
    - :math:`\epsilon` is the error term.

    Parameters
    ----------
    x : str, list, or array-like, optional
        Independent variable(s). If a string is provided, `data` must
        also be supplied, and `x` should refer to a column name within `data`.
    y : str, list, or array-like, optional
        Dependent variable. Similar to `x`, if a string is provided,
        it should refer to a column name within a supplied `data` DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing the variables specified in `x` and `y`.
        Required if `x` or `y` are specified as strings.
    view : bool, optional
        If True, generates a plot to visualize the data points and the
        regression line. Default is False.
    sample_weight : array-like of shape (n_samples,), default=None
        Individual weights for each sample.
    as_frame : bool, optional
        If True, returns the output as a pandas Series. Default is False.
    plot_type : str, optional
        Type of plot for visualization. Currently supports 'scatter_line'.
        Default is 'scatter_line'.
    cmap : str, optional
        Color map for the plot. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments to pass to the `LinearRegression`
        model constructor.

    Returns
    -------
    model : LinearRegression
        The fitted linear regression model.
    coefficients : np.ndarray
        Coefficients of the independent variable(s) in the model.
    intercept : float
        Intercept of the linear regression model.

    Note
    ----
    This function streamlines the process of performing linear regression analysis,
    making it straightforward to model relationships between two variables and 
    extract useful statistics such as the regression coefficients and intercept.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from gofast.stats.relationships import perform_linear_regression
    >>> x = np.random.rand(100)
    >>> y = 2.5 * x + np.random.normal(0, 0.5, 100)
    >>> model, coefficients, intercept = perform_linear_regression(x, y)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")

    Using a DataFrame:
    
    >>> df = pd.DataFrame({'X': np.random.rand(100), 
    ...                    'Y': 2.5 * np.random.rand(100) + np.random.normal(0, 0.5, 100)})
    >>> model, coefficients, intercept = perform_linear_regression('X', 'Y', data=df)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")
    """
    x_values, y_values = assert_xy_in(
        x, y,
        data=data, 
        xy_numeric= True
    )
    
    if _is_arraylike_1d(x_values): 
        x_values = x_values.reshape(-1, 1)
        
    model = LinearRegression(**kwargs)
    model.fit(x_values, y_values, sample_weight=sample_weight) 
    coefficients = model.coef_
    intercept = model.intercept_

    if view:
        plt.figure(figsize=fig_size)
        plt.scatter(x_values, y_values, color='blue', label='Data Points')
        plt.plot(x_values, model.predict(x_values), color='red', label='Regression Line')
        plt.title('Linear Regression Analysis')
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.legend()
        plt.show()
        
    if as_frame:
        return to_series_if(
            model, coefficients, intercept, 
            value_names=['Linear-model', "Coefficients", "Intercept"], 
            name='linear_regression'
        )
    return model, coefficients, intercept


@make_data_dynamic(capture_columns=True, dynamize= False )
def perform_kmeans_clustering(
    data: ArrayLike,
    n_clusters: int = 3,
    n_init="auto", 
    columns: list = None,
    view: bool = True,
    cmap='viridis', 
    fig_size: Tuple[int, int] = (8, 8),
    **kwargs
) -> Tuple[KMeans, ArrayLike]:
    r"""
    Applies K-Means clustering to the dataset, returning the fitted model and 
    cluster labels for each data point.

    K-Means clustering aims to partition `n` observations into `k` clusters 
    in which each observation belongs to the cluster with the nearest mean,
    serving as a prototype of the cluster.

    .. math::
        \underset{S}{\mathrm{argmin}}\sum_{i=1}^{k}\sum_{x \in S_i}||x - \mu_i||^2

    Where:
    - :math:`S` is the set of clusters
    - :math:`k` is the number of clusters
    - :math:`x` is each data point
    - :math:`\mu_i` is the mean of points in :math:`S_i`.

    Parameters
    ----------
    data : array_like or pd.DataFrame
        Multidimensional dataset for clustering. Can be a pandas DataFrame 
        or a 2D numpy array. If a DataFrame and `columns` is specified, only
        the selected columns are used for clustering.
    n_clusters : int, optional
        Number of clusters to form. Default is 3.
    n_init : str or int, optional
        Number of time the k-means algorithm will be run with different 
        centroid seeds. The final results will be the best output of `n_init` 
        consecutive runs in terms of inertia. If "auto", it is set to 10 or
        max(1, 2 + log(n_clusters)), whichever is larger.
    columns : list, optional
        Specific columns to use for clustering if `data` is a DataFrame. 
        Ignored if `data` is an array_like. Default is None.
    view : bool, optional
        If True, generates a scatter plot of the clusters with centroids.
        Default is True.
    cmap : str, optional
        Colormap for the scatter plot, default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the scatter plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to the `KMeans` constructor.

    Returns
    -------
    model : KMeans
        The fitted KMeans model.
    labels : np.ndarray
        Cluster labels for each point in the dataset.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    >>> model, labels = perform_kmeans_clustering(X, n_clusters=3)
    >>> print(labels)

    Using a DataFrame and selecting specific columns:
    >>> df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    >>> model, labels = perform_kmeans_clustering(
    ...     df, columns=['feature1', 'feature2'], n_clusters=3)
    >>> print(labels)

    See Also
    --------
    sklearn.cluster.KMeans : 
        The KMeans clustering algorithm provided by scikit-learn.
        
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data_for_clustering = data[columns]
    else:
        data_for_clustering = data

    km = KMeans(n_clusters=n_clusters, n_init=n_init,  **kwargs)
    labels = km.fit_predict(data_for_clustering)

    if view:
        plt.figure(figsize=fig_size)
        # Scatter plot for clusters
        if isinstance(data_for_clustering, pd.DataFrame):
            plt.scatter(
                data_for_clustering.iloc[:, 0], data_for_clustering.iloc[:, 1],
                c=labels, cmap=cmap, marker='o', edgecolor='k', s=50, alpha=0.6)
        else:
            plt.scatter(data_for_clustering[:, 0], data_for_clustering[:, 1],
                        c=labels, cmap=cmap, marker='o', edgecolor='k',
                        s=50, alpha=0.6)
        
        # Plot centroids
        centers = km.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red',
                    s=200, alpha=0.5, marker='+')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    return km, labels

@make_data_dynamic(capture_columns=True, dynamize=False)
def mds_similarity(
    data,
    n_components: int = 2,
    columns: Optional[list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[tuple] = (10, 6),
    **kws
):
    """
    Perform Multidimensional Scaling (MDS) to project the dataset into a 
    lower-dimensional space while preserving the pairwise distances between 
    points as much as possible.

    MDS seeks a low-dimensional representation of the data in which the 
    distances respect well the distances in the original high-dimensional space.

    .. math::
        \min_{X} \sum_{i<j} (||x_i - x_j|| - d_{ij})^2

    where :math:`d_{ij}` are the distances in the original space, :math:`x_i` and 
    :math:`x_j` are the coordinates in the lower-dimensional space, and 
    :math:`||\cdot||` denotes the Euclidean norm.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        The dataset to perform MDS on. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    n_components : int, optional
        The number of dimensions in which to immerse the dissimilarities,
        by default 2.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame, by default None.
    as_frame : bool, optional
        If True, the function returns the result as a pandas DataFrame,
        by default False.
    view : bool, optional
        If True, displays a scatter plot of the MDS results, by default False.
    cmap : str, optional
        Colormap for the scatter plot, by default 'viridis'.
    fig_size : tuple, optional
        Size of the figure for the scatter plot, by default (10, 6).
    **kws : dict
        Additional keyword arguments passed to `sklearn.manifold.MDS`.

    Returns
    -------
    mds_result : ndarray or DataFrame
        The coordinates of the data in the MDS space as a NumPy array or
        pandas DataFrame, depending on the `as_frame` parameter.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from gofast.stats.relationships import mds_similarity
    >>> digits = load_digits()
    >>> mds_coordinates = mds_similarity(digits.data, n_components=2, view=True)
    >>> print(mds_coordinates.shape)
    (1797, 2)

    Using with a DataFrame and custom columns:

    >>> import pandas as pd
    >>> df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(
    ...    digits.data.shape[1])])
    >>> mds_df = mds_similarity(df, columns=['pixel_0', 'pixel_64'], as_frame=True, view=True)
    >>> print(mds_df.head())

    This function is particularly useful for visualizing high-dimensional 
    data in two or three dimensions, allowing insights into the structure and
    relationships within the data that are not readily apparent in 
    the high-dimensional space.
    """
    # Ensure the data is an array for MDS processing
    data_array = np.asarray(data)

    # Initialize and apply MDS
    mds = MDS(n_components=n_components, **kws)
    mds_result = mds.fit_transform(data_array)
    
    # Visualization
    if view:
        plt.figure(figsize=fig_size)
        scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], cmap=cmap)
        plt.title('Multidimensional Scaling (MDS) Results')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Data Value')
        plt.show()
    
    # Convert the result to a DataFrame if requested
    if as_frame:
        columns = [f'Component {i+1}' for i in range(n_components)]
        mds_result = pd.DataFrame(mds_result, columns=columns)
    
    return mds_result

@make_data_dynamic(capture_columns=True, dynamize=False)
def perform_spectral_clustering(
    data: Union[DataFrame, ArrayLike],
    n_clusters: int = 2,
    assign_labels: str = 'discretize',
    as_frame: bool=False, 
    random_state: int = None,
    columns: Optional[list] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = (10, 6),
    **kws
):
    """
    Perform Spectral Clustering on a dataset, with an option to visualize
    the clustering results.

    Spectral Clustering uses the eigenvalues of a similarity matrix to perform
    dimensionality reduction before clustering in fewer dimensions. This method
    is particularly effective for identifying clusters that are not necessarily
    globular.

    .. math::
        L = D^{-1/2} (D - W) D^{-1/2} = I - D^{-1/2} W D^{-1/2}

    Where :math:`W` is the affinity matrix, :math:`D` is the diagonal degree matrix,
    and :math:`L` is the normalized Laplacian.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Dataset for clustering. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    n_clusters : int, optional
        Number of clusters to form, default is 2.
    assign_labels : str, optional
        Strategy for assigning labels in the embedding space: 'kmeans',
        'discretize', or 'cluster_qr', default is 'discretize'.
    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy for assigning labels in the embedding space. There are two
        ways to assign labels after the Laplacian embedding. k-means is a
        popular choice, but it can be sensitive to initialization.
        Discretization is another approach which is less sensitive to random
        initialization .

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls.

    columns : list, optional
        Specific columns to use if `data` is a DataFrame.
    view : bool, optional
        If True, displays a scatter plot of the clustered data.
    cmap : str, optional
        Colormap for the scatter plot.
    fig_size : tuple, optional
        Size of the figure for the scatter plot.
    **kws : dict
        Additional keyword arguments passed to `SpectralClustering`.
    
    See Also
    --------
    sklearn.cluster.SpectralClustering: Spectral clustering
    sklearn.cluster.KMeans : K-Means clustering.
    sklearn.cluster.DBSCAN : Density-Based Spatial Clustering of
        Applications with Noise.
        
    Returns
    -------
    labels : ndarray or DataFrame
        Labels of each point. Returns a DataFrame if `as_frame=True`, 
        containing the original data and a 'cluster' column with labels.

    Examples
    --------
    >>> from sklearn.datasets import make_circles
    >>> from gofast.stats import perform_spectral_analysis 
    >>> X, _ = make_circles(n_samples=300, noise=0.1, factor=0.2, random_state=42)
    >>> labels = perform_spectral_clustering(X, n_clusters=2, view=True)

    Using a DataFrame and returning results as a DataFrame:
    >>> df = pd.DataFrame(X, columns=['x', 'y'])
    >>> results_df = perform_spectral_clustering(df, n_clusters=2, as_frame=True)
    >>> print(results_df.head())
    """
    data_for_clustering = np.asarray(data)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels=assign_labels,
        random_state=random_state,
        **kws
    )
    labels = clustering.fit_predict(data_for_clustering)

    if view:
        _plot_clustering_results(data_for_clustering, labels, cmap, fig_size)

    if as_frame:
        results_df = pd.DataFrame(
            data_for_clustering, columns=columns if columns else [
                'feature_{}'.format(i) for i in range(data_for_clustering.shape[1])])
        results_df['cluster'] = labels
        return results_df

    return labels

def _plot_clustering_results(data, labels, cmap, fig_size):
    """Helper function to plot clustering results."""
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=0.6)
    plt.title('Spectral Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.show()