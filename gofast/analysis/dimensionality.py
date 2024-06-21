# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Reduce dimension for data visualisation.

Reduce number of dimension down to two (or to three) for instance, make  
it possible to plot high-dimension training set on the graph and often
gain some important insights by visually detecting patterns, such as 
clusters.

"""
from __future__ import annotations 
import os
import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding

from .._gofastlog import gofastlog
from ..api.box import KeyBox
from ..api.extension import make_introspection 
from ..api.summary import ReportFactory  
from ..api.types import Any,Dict, Optional, ArrayLike
from ..api.types import NDArray, DataFrame
from ..api.util import parse_component_kind 
from ..tools.validator import check_array, validate_positive_integer

_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    'nPCA', 'kPCA', 'LLE', 'iPCA', 
    'get_most_variance_component',
    'project_ndim_vs_explained_variance', 
    "get_feature_importances",
]

def get_feature_importances(
    components: 'NDArray', 
    fnames: 'ArrayLike'=None, 
    n_axes: int = 2, 
    scale_by_variance: bool = False, 
    explained_variance: 'NDArray' = None, 
    view: bool = False, 
    kind: str = 'PC1'
    ) -> 'list':
    """
    Retrieves the feature importances from PCA components and optionally 
    scales them by the explained variance ratio.

    Parameters
    ----------
    components : np.ndarray
        The array containing PCA components where each row represents a component,
        and each column corresponds to a feature.
    fnames : ArrayLike, optional 
        An array-like structure containing the names of the features corresponding
        to the columns in the PCA components matrix.
    n_axes : int, optional
        The number of principal components to consider for feature importances.
        Defaults to 2.
    scale_by_variance : bool, optional
        If True, scales the feature importances by the explained variance ratio 
        of each component. 
        `explained_variance` must be provided if this is True.
    explained_variance : np.ndarray, optional
        The array containing the explained variance ratio for each principal 
        component.
        Required if `scale_by_variance` is True.
    view : bool, optional
        If True, plots the feature importances in descending order for each 
        component.
        Defaults to False.
    kind : str, optional 
       A string that identifies the principal component number to extract,
       e.g., 'pc1'. The string should contain a numeric part that corresponds
       to the component index in `pc_list`. 
       Default is ``pc1``. 

    Returns
    -------
    list
        A list of tuples, each tuple containing (
            'pc{i}', feature_names, sorted_component_values),
        where 'pc{i}' is the principal component label (starting from 1),
        `feature_names` is an array of feature names sorted by their importance,
        and `sorted_component_values` are the corresponding sorted component 
        values.

    Raises
    ------
    ValueError
        If `scale_by_variance` is True but `explained_variance` is not provided.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.analysis.dimensionality import get_feature_importances
    >>> features = ['feature1', 'feature2', 'feature3']
    >>> components = np.array([[0.5, -0.8, 0.3], [0.4, 0.9, -0.1]])
    >>> explained_variances = np.array([0.6, 0.4])
    >>> importances = get_feature_importances(
        components, features, 2, True, explained_variances, True)
    """
    if scale_by_variance and explained_variance is None:
        raise ValueError("Explained variance must be provided when"
                         " 'scale_by_variance' is True.")
        
    n_axes = validate_positive_integer(n_axes, "n_axes")
    
    components = np.asarray(components)
    if components.shape[0] < n_axes:
        warnings.warn(f"Requested number of axes ({n_axes}) exceeds"
                      f" the available components ({components.shape[0]}). "
                      f"Reducing to {components.shape[0]}.", UserWarning)
        n_axes = components.shape[0]
    
    if isinstance (fnames, str):
        fnames =[fnames]
        
    if fnames is None:
        fnames = [f'Feature {i+1}' for i in range(components.shape[1])]
    elif len(fnames) != components.shape[1]:
        raise ValueError("The length of 'fnames' must match the number"
                         " of features in 'components'.")
    pc = []
    for i in range(n_axes):
         # sort by absolute value, descending
        indices = np.argsort(-np.abs(components[i, :])) 
        sorted_components = components[i, indices]
        sorted_fnames = np.array(fnames)[indices]

        if scale_by_variance:
            sorted_components *= explained_variance[i]

        pc.append((f'pc{i+1}', sorted_fnames, sorted_components))

    if view:
        sorted_fnames, sorted_components = parse_component_kind(pc, kind)
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_fnames, sorted_components, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Scaled Feature Importance' if scale_by_variance 
                   else 'Feature Importance')
        plt.title(f'Feature Importances for {str(kind).title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return pc
 
def nPCA(
    X: NDArray | DataFrame, 
    n_components: Optional[int | float]=None, *, 
    view: bool=False, 
    return_X: bool=True, 
    plot_kws: Dict[str, Any]=None, 
    n_axes: int=None, 
    **pca_kws: Any
 )-> NDArray| 'nPCA':

    X = check_array(X)

    n_axes = validate_positive_integer(
        n_axes, "n_axes") if n_axes is not None else n_axes 
    
    pca = PCA(n_components=n_components or 0.95, **pca_kws)
    X_transformed = pca.fit_transform(X)
    
    obj=KeyBox() 
    obj.X=X_transformed  # Store the transformed data

    # Calculate cumulative sum of explained variance ratio if needed
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    # Plotting if view is True
    if view:
        plot_kws = plot_kws or {'label': 'Explained variance vs number of dimensions'}
        plt.plot(cumsum, **plot_kws)
        plt.xlabel('Number of Dimensions')
        plt.ylabel('Explained Variance')
        plt.title('Explained Variance vs. Number of Dimensions')
        plt.grid(True)  # Added grid for better readability
        plt.show()
    
    # Copy PCA attributes to obj
    for key, value in pca.__dict__.items():
        setattr(obj, key, value)
    
    # Set the number of axes
    obj.n_axes = n_axes if n_axes is not None else pca.n_components_
    
    # Handle feature importances if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        obj.feature_importances_ = find_f_importances(
            np.array(X.columns), pca.components_, obj.n_axes
        )
    return X_transformed if return_X else obj
 
nPCA.__doc__="""\
Normal Principal Components analysis (PCA)

Performs Principal Components Analysis (PCA), a popular linear dimensionality 
reduction technique. PCA identifies the hyperplane that lies closest to the 
data and projects the data onto it, reducing the number of dimensions while 
attempting to preserve as much variance as possible.

Parameters
----------
X : ndarray or DataFrame
    Training set represented as an M x N matrix where M is the number of 
    samples and N is the number of features. This dataset is used as 
    independent variables in learning. `X` must be two-dimensional and 
    numerical.
n_components : float or int, optional
    The number of dimensions to keep. If set between 0 and 1, it represents 
    the fraction of variance to preserve. If `None` (default), 95% of the 
    variance is preserved.
return_X : bool, default=True
    If True, returns the transformed dataset with the most representative 
    variance preserved. Otherwise, returns the PCA model object.
view : bool, default=False
    If True, plots the explained variance as a function of the number of 
    dimensions.
n_axes : int, optional
    The number of principal components for which to retrieve the variance 
    ratio. If `None`, the feature importance is computed using the 
    cumulative variance that represents 95% of the total.
plot_kws : dict, optional
    Additional keyword arguments passed to `matplotlib.pyplot.plot` for 
    customizing the variance plot.
pca_kws : dict
    Additional keyword arguments passed to `sklearn.decomposition.PCA`.

Returns
-------
ndarray or nPCA object
    The transformed dataset or the PCA model object, depending on the 
    value of `return_X`.

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from gofast.analysis.dimensionality import nPCA
>>> from gofast.datasets import fetch_data
>>> X= fetch_data('Bagoue analysed dataset')
>>> pca = nPCA(X, 0.95, n_axes =3, return_X=False)
>>> pca.components_
>>> pca.feature_importances_

>>> X = load_iris().data
>>> pca_result = nPCA(X, n_components=0.95, view=True, return_X=False)
>>> print(pca_result.components_)
>>> print(pca_result.feature_importances_)

See Also
--------
PCA : Principal component analysis class in scikit-learn.
matplotlib.pyplot.plot : Plotting function for visualizing variance.

Notes
-----
PCA is particularly useful in processing and reducing the dimensionality 
of high-dimensional datasets, making them more manageable for downstream 
analysis and visualization.
"""  
def _process_pca(
        X0, n_components, n_batches, ipca_kws, store_in_binary_file, 
        filename, return_X, view):
    """
    Processes PCA incrementally and handles data storage and transformation.
    """
    # Initialize IncrementalPCA
    inc_pcaObj = IncrementalPCA(n_components=n_components, **ipca_kws)

    # Fit PCA incrementally by batches
    for X_batch in np.array_split(X0, n_batches):
        inc_pcaObj.partial_fit(X_batch)

    # Transform the original data
    X = inc_pcaObj.transform(X0)

    # Optionally store the results in a binary file
    if store_in_binary_file:
        if not filename or not os.path.isfile(filename):
            warnings.warn('A valid binary filename must be provided for storage on disk.')
            _logger.error('A valid binary filename is required but was not provided.')
            raise FileNotFoundError('Valid binary filename not found.')
        
        # Setup memory-mapped file for large datasets
        X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=X0.shape)
        batch_size = X0.shape[0] // n_batches
        inc_pcaObj = IncrementalPCA(n_components=n_components,
                                    batch_size=batch_size, **ipca_kws)
        X = inc_pcaObj.fit(X_mm)

    # Create a clone object of KeyBox for storing results and further analysis
    obj = KeyBox()
    obj.X = X
    make_introspection(obj, inc_pcaObj)
    setattr(obj, 'n_axes', getattr(obj, 'n_components_', None))

    # Handle feature importances for DataFrame inputs
    if isinstance(X0, pd.DataFrame):
        pca_components_ = getattr(obj, 'components_', None)
        obj.feature_importances_ = find_f_importances(np.array(
            list(X0.columns)), pca_components_, obj.n_axes)

    # Optionally visualize PCA results
    if view:
        project_ndim_vs_explained_variance(obj, obj.n_components)

    # Return the transformed data or the 
    # KeyBox object based on user preference
    return X if return_X else obj

def iPCA(
    X: NDArray | DataFrame,
    n_components: Optional[float | int] =None,*, 
    view: bool =False, 
    n_batches: int =None,
    return_X:bool=True, 
    store_in_binary_file: bool =False,
    filename: Optional[str]=None,
    verbose: int=0,
    **ipca_kws
 )-> NDArray| 'iPCA': 

    X=check_array(X)
    # Check if the number of components is not specified
    if n_components is None:
        # Retrieve the most significant number of components that
        # accounts for 95% variance
        n_components = get_most_variance_component(X, verbose=verbose)
    
        # Validate if the number of batches is also not specified, 
        # raise an appropriate error
        if n_batches is None:
            raise ValueError("The number of batches cannot be None"
                             " when 'n_components' is not specified.")
        # Calculate the maximum number of components allowed based on the batch size
        max_components_per_batch = len(X) // n_batches + 1
    
        # Check if the computed number of components exceeds 
        # the allowed components per batch
        if n_components > max_components_per_batch:
            # Adjust n_components to fit within the allowed maximum and issue a warning
            adjusted_components = len(X) // n_batches
            warnings.warn(
                f"Specified 'n_components' ({n_components}) exceeds the maximum "
                f"number allowed by the batch size ({max_components_per_batch}). "
                f"'n_components' has been adjusted to {adjusted_components}.",
                UserWarning
            )
            n_components = adjusted_components
            _logger.debug(
                f"'n_components' is reset to {n_components} based on batch size.")

    return _process_pca(
            X, n_components, n_batches, ipca_kws, store_in_binary_file, 
            filename, return_X, view
     )

iPCA.__doc__="""\
Incremental PCA 

Incremental PCA allows for processing the dataset in mini-batches, which is 
beneficial for large datasets that do not fit into memory. It is also 
suitable for applications requiring online updates of the decomposition.
 
Once problem with the preceeding implementation of PCA is that 
requires the whole training set to fit in memory in order of the SVD
algorithm to run. This is usefull for large training sets, and also 
applying PCA online(i.e, on the fly as a new instance arrive)
 
Parameters 
-------------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.
n_components : int or None, optional
    The number of dimensions to keep. If between 0 and 1, it indicates the 
    fraction of variance to preserve. Note that `'n_components'` parameter of 
    `iPCA` must be an int in the range [1, inf) or None not float.
n_batches : int, optional
    The number of batches to split the training dataset into for incremental
    learning. Must be specified if `store_in_binary_file` is True.
store_in_binary_file : bool, default=False
    If True, uses numpy `memmap` to store data on disk, allowing manipulation
    as if it were entirely in memory. Only the necessary data is loaded 
    into memory as needed.
filename : str, optional
    The filename for storing the binary file on disk when using `memmap`.
    Required if `store_in_binary_file` is True.
return_X : bool, default=True
    If True, returns the transformed dataset. Otherwise, returns the PCA model
    object configured with the `IncrementalPCA` settings.
view : bool, default=False
    If True, plots the explained variance as a function of the number of
    dimensions.
ipca_kws : dict, optional
    Additional keyword arguments passed to `sklearn.decomposition.IncrementalPCA`.

Returns
-------
ndarray or iPCA object
    The transformed dataset or the iPCA model object, depending on `return_X`.

Examples
--------
>>> from gofast.analysis.dimensionality import iPCA
>>> from gofast.datasets import fetch_data
>>> X = fetch_data('Bagoue analysed data')
>>> X_transf = iPCA(X, n_components=2, n_batches=100, view=True)

Notes
-----
For large datasets or online learning scenarios, `iPCA` is preferable to standard
PCA since it does not require loading the entire dataset into memory.
"""
def plot_kernel_pca_results(
        X_transformed: 'NDArray', title: str = 'Kernel PCA Results'):
    if X_transformed.shape[1] < 2:
        raise ValueError("Kernel PCA results must have at least"
                         " two dimensions for this plot.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='blue',
                marker='o', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(True)
    plt.show() 

def kPCA(
    X: 'NDArray | DataFrame',
    n_components: float | int = None,
    *, 
    return_X: bool = True, 
    kernel: str = 'rbf',
    reconstruct_pre_image: bool = False,
    view: bool = False, 
    verbose: bool = False, 
    **kpca_kws
) -> 'NDArray | KeyBox': 
    
    obj = KeyBox() 
    if n_components is None: 
        n_components = get_most_variance_component(X, verbose= verbose)
    Xr = np.asarray(X.copy())  # Ensure array operations are compatible
    kpcaObj = KernelPCA(n_components=n_components, kernel=kernel,
                        fit_inverse_transform=reconstruct_pre_image,
                        **kpca_kws)

    obj.X = kpcaObj.fit_transform(Xr)
    
    if reconstruct_pre_image:
        from sklearn.metrics import mean_squared_error
        obj.X_preimage = kpcaObj.inverse_transform(obj.X)
        # Compute the reconstruction pre-image error
        obj.X_preimage_error = mean_squared_error(Xr, obj.X_preimage)
    
    # Populate attributes inherited from kpca object
    make_introspection(obj, kpcaObj)
    # Set axes and features importances
    set_axes_and_feature_importances(obj, Xr)

    # Optionally visualize Kernel PCA results
    if view:
        plot_kernel_pca_results(
            obj.X, title=f'Results of Kernel PCA with {kernel} kernel')
        
    return obj.X if return_X else obj
  
kPCA.__doc__="""\
Kernel PCA 

`kPCA` performs complex nonlinear projections for dimentionality
reduction.

Commonly the kernel tricks is a mathematically technique that implicitly
maps instances into a very high-dimensionality space(called the feature
space), enabling non linear classification or regression with SVMs. 
Recall that a linear decision boundary in the high dimensional 
feature space corresponds to a complex non-linear decison boundary
in the original space.

Parameters 
-------------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.

n_components: int, optional 
    Number of dimension to preserve. If`n_components` is ranged between 
    float 0. to 1., it indicated the number of variance ratio to preserve. 
    If ``None`` as default value the number of variance to preserve is 
    ``95%``.
    
return_X: bool, default =True , 
    return the train set transformed with most representative varaince 
    ratio. 
    
kernel: {'linear', 'poly', \
        'rbf', 'sigmoid', 'cosine', 'precomputed'}, default='rbf'
    Kernel used for PCA.
    
kpca_kws: dict, 
    Additional keyword arguments passed to 
    :class:`sklearn.decomposition.KernelPCA`

Returns 
----------
X (NDArray) or `kPCA` object, 
    The transformed training set or the kPCA container attributes for 
    plotting purposes. 
    
Examples
----------
>>> from gofast.analysis.dimensionality import kPCA
>>> from gofast.datasets import fetch_data 
>>> X=fetch_data('Bagoue analysis data')
>>> Xtransf=kPCA(X,n_components=None,kernel='rbf', gamma=0.04, view=True)
"""

def visualize_lle_embedding(X, title="LLE Embedding"):
    plt.figure(figsize=(8, 6))
    if X.shape[1] >= 2:
        plt.scatter(X[:, 0], X[:, 1], c='blue', marker='o',
                    edgecolor='k', alpha=0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.grid(True)
        plt.show()
    else:
        print("Visualization requires at least 2 dimensions.")

def LLE(
    X: 'NDArray | DataFrame',
    n_components: Optional[float | int] = None,
    *,
    return_X: bool = True, 
    n_neighbors: int = 5, 
    view: bool = False, 
    verbose: bool = False, 
    **lle_kws
) -> 'NDArray | LLE':
    # Create a new instance for LLE results
    obj = KeyBox()
    
    # Handle the n_components logic
    if n_components is None:
        n_components = get_most_variance_component(X, verbose=verbose)
    
    # Setup and apply LLE
    lleObj = LocallyLinearEmbedding(
        n_components=n_components, n_neighbors=n_neighbors, **lle_kws)
    X_transformed = lleObj.fit_transform(X)
    obj.X = X_transformed
    
    # Populate attributes from the LLE object
    make_introspection(obj, lleObj)
    
    # Optionally visualize LLE embedding
    if view:
        visualize_lle_embedding(X_transformed)
    
    return X_transformed if return_X else obj

 
LLE.__doc__="""\
Locally Linear Embedding(LLE) 

`LLE` is nonlinear dimensinality reduction based on closest neighbors 
(c.n).

LLE is another powerfull non linear dimensionality reduction(NLDR)
technique. It is Manifold Learning technique that does not rely
on projections like `PCA`. In a nutshell, works by first measurement
how each training instance library lineraly relates to its closest 
neighbors(c.n.), and then looking for a low-dimensional representation 
of the training set where these local relationships are best preserved
(more details shortly).Using LLE yields good resuls especially when 
makes it particularly good at unrolling twisted manifolds, especially
when there is too much noise.

Parameters
----------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.

n_components: int, optional 
    Number of dimension to preserve. If`n_components` is ranged between 
    float 0. to 1., it indicated the number of variance ratio to preserve. 
    If ``None`` as default value the number of variance to preserve is 
    ``95%``.

n_neighbors : int, default=5
    Number of neighbors to consider for each point.
        
return_X: bool, default =True , 
    return the train set transformed with most representative varaince 
    ratio. 
lle_kws: dict, 
    Additional keyword arguments passed to 
    :class:`sklearn.decomposition.LocallyLinearEmbedding`. 
    
Returns 
----------
X (NDArray) or `LLE` object, 
    The transformed training set or the LLE container attributes for 
    plotting purposes. 
     
References
-----------
Gokhan H. Bakir, Jason Wetson and Bernhard Scholkoft, 2004;
"Learning to Find Pre-images";Tubingen, Germany:Max Planck Institute
for Biological Cybernetics.

S. Roweis, L.Saul, 2000, Nonlinear Dimensionality Reduction by
Loccally Linear Embedding.

Notes
------
Scikit-Learn used the algorithm based on Kernel Ridge Regression
     
Example
-------
>>> from gofast.analysis.dimensionality import LLE
>>> from gofast.datasets import fetch_data 
>>> X=fetch_data('Bagoue analysed')
>>> lle_kws ={
...    'n_components': 4, 
...    "n_neighbors": 5}
>>> Xtransf=LLE(X, view =True, **lle_kws)

"""

def find_f_importances(
    fnames: 'ArrayLike', 
    components: 'np.ndarray', n_axes: int = 2, 
    view: bool=False, 
    kind: str='pc1', 
    ) -> 'list':
    """
    Retrieves the feature importances based on the principal component analysis.

    Parameters
    ----------
    fnames : ArrayLike
        An array-like structure containing the names of the features corresponding
        to the columns in the PCA components matrix.
    components : np.ndarray
        The array containing PCA components. Each row represents a component,
        and each column corresponds to a feature.
    n_axes : int, optional
        The number of principal components to consider for feature importances.
        Defaults to 2.
    view : bool, optional
            If True, plots the feature importances in descending order for 
            each component. Defaults to False.
    kind : str, optional 
       A string that identifies the principal component number to extract,
       e.g., 'pc1'. The string should contain a numeric part that corresponds
       to the component index in `pc_list`. 
       Default is ``pc1``. 
       
    Returns
    -------
    list
        A list of tuples, each tuple containing:
        ('pc{i}', feature_names, sorted_component_values)
        where 'pc{i}' is the principal component label (starting from 1),
        `feature_names` is an array of feature names sorted by their importance,
        and `sorted_component_values` are the corresponding sorted component values.

    Raises
    ------
    UserWarning
        If `n_axes` exceeds the number of available axes in the `components`.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.analysis.dimensionality import find_f_importances
    >>> features = ['feature1', 'feature2', 'feature3']
    >>> components = np.array([[0.5, -0.8, 0.3], [0.4, 0.9, -0.1]])
    >>> importances = find_f_importances(features, components, 2)
    >>> print(importances)
    [('pc1', array(['feature2', 'feature1', 'feature3']), array([-0.8,  0.5,  0.3])),
     ('pc2', array(['feature2', 'feature1', 'feature3']), array([ 0.9,  0.4, -0.1]))]
    
    >>> features = ['feature1', 'feature2', 'feature3']
    >>> components = np.array([[0.5, -0.8, 0.3], [0.4, 0.9, -0.1]])
    >>> importances = find_f_importances(features, components, 2, view=True)
    
    Notes
    -----
    The function warns and adjusts `n_axes` if it exceeds the number of rows in
    `components`. This is to ensure that the number of components requested does
    not surpass the available components derived from PCA.
    """
    n_axes = validate_positive_integer(n_axes, "n_axes")
    if components.shape[0] < n_axes:
        warnings.warn(f"Requested number of axes ({n_axes}) exceeds the"
                      f" available components ({components.shape[0]}). "
                      f"Reducing to {components.shape[0]}.", UserWarning)
        n_axes = components.shape[0]

    pc = []
    for i in range(n_axes):
        # Sort indices of the components based on their
        # absolute values in descending order
        indices = np.argsort(-np.abs(components[i, :]))
        sorted_components = components[i, indices]
        sorted_fnames = np.array(fnames)[indices]

        pc.append((f'pc{i+1}', sorted_fnames, sorted_components))
        
    if view:
        sorted_fnames, sorted_components = parse_component_kind(pc, kind)
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_fnames, sorted_components, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Feature Importance')
        plt.title(f'Feature Importances for {str(kind).title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
            
    return pc

def validate_explained_variance_ratio(obj, verbose=0):
    """
    Validates whether an object has the 'explained_variance_ratio_' attribute
    and returns the cumulative sum of this attribute if available.

    Parameters
    ----------
    obj : object
        An object that might have an 'explained_variance_ratio_' attribute,
        typically a fitted PCA model or similar.
    verbose : bool
        If True, provides detailed debugging information and recommendations.

    Returns
    -------
    np.ndarray or None
        The cumulative sum of the explained variance ratio if the attribute
        exists, otherwise None.

    Raises
    ------
    UserWarning
        Warns if the object does not have the 'explained_variance_ratio_'
        attribute.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.datasets import load_iris
    >>> from gofast.analysis.dimensionality import validate_explained_variance_ratio
    >>> X = load_iris().data
    >>> pca = PCA().fit(X)
    >>> cumsum = validate_explained_variance_ratio(pca, verbose=True)
    >>> print(cumsum)
    """
    try:
        # Attempt to retrieve and return the cumulative sum 
        # of the explained variance ratio
        cumsum = np.cumsum(getattr(obj, 'explained_variance_ratio_'))
        return cumsum
    except AttributeError:
        # Determine the type of the object for a specific warning message
        obj_name = None
        if hasattr(obj, 'kernel'):
            obj_name = 'KernelPCA'
        elif hasattr(obj, 'n_neighbors') and hasattr(obj, 'nbrs_'):
            obj_name = 'LocallyLinearEmbedding'

        # Construct a warning message based on the type
        if obj_name is not None:
            warning_msg = (
                f"{obj_name} object does not support 'explained_variance_ratio_'. "
                "Cannot validate or plot the explained variance ratio.")
            warnings.warn(warning_msg, UserWarning)
            _logger.debug(
                f"{obj.__class__.__name__} is identified as {obj_name} and"
                f" lacks 'explained_variance_ratio_'.")

        # Optionally print detailed information if verbose is True
        if verbose:
            detailed_msg = (
                f"{obj_name if obj_name else 'The object'} does not have the attribute "
                "'explained_variance_ratio_', which is required for plotting "
                "or validating the explained variance ratio."
                )
            summary =ReportFactory().add_recommendations( 
                detailed_msg, keys ="explained_variance_ratio_ exists?",
                max_char_text=90 )
            print(summary)

        # Set the attribute to None for safety and return None
        setattr(obj, 'explained_variance_ratio_', None)
        
        return None

def project_ndim_vs_explained_variance(
        obj, /, n_components: float | int = None,
        verbose: int=0,  **plot_kws) -> object | None:
    """
    Plots the number of dimensions against the explained variance ratio of a 
    PCA object or similar.

    This function provides a visual representation of the explained variance
    vs. the number of dimensions for PCA-related objects.

    Parameters
    ----------
    obj : object
        PCA-like object that should contain an attribute `explained_variance_ratio_`.
        Applicable to objects like PCA, KernelPCA, IncrementalPCA, etc.
    n_components : int or float, optional
        Number of principal components considered in the plot. If not provided,
        the function attempts to plot all components available in the object.
    plot_kws : dict
        Additional keyword arguments to pass to the matplotlib plot function.

    Returns
    -------
    object or None
        If plotting is successful, returns None. If the necessary attributes
        are missing and plotting cannot proceed, returns the original object.

    Raises
    ------
    UserWarning
        Warns if the `explained_variance_ratio_` attribute is missing or if
        `n_components` is not set when required.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.datasets import load_iris
    >>> from gofast.analysis.dimensionality import project_ndim_vs_explained_variance
    >>> X = load_iris().data
    >>> pca = PCA().fit(X)
    >>> project_ndim_vs_explained_variance(pca)

    Notes
    -----
    Ensure that the object passed to this function is fitted and contains the
    `explained_variance_ratio_` attribute necessary for plotting.
    """
    if n_components is None:
        warnings.warn('`n_components` is None, unable to plot projection'
                      ' without specified components.',UserWarning)
        return obj
    # Attempt to access the explained variance ratio attribute
    cumsum = validate_explained_variance_ratio (obj, verbose)
    if cumsum is None: 
        return 
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsum) + 1), cumsum, marker='o', **plot_kws)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.show()
    return None

def get_most_variance_component(
    X: NDArray | DataFrame, 
    n_components: int = None,
    verbose: int=0, 
    **pca_kws
    ) -> 'ArrayLike':
    """
    Determines the number of principal components that explain at least 95% of 
    the variance in the dataset.

    Parameters
    ----------
    X : NDArray | DataFrame
        Training set to perform PCA on.
    n_components : int, optional
        The number of principal components to retain. If None, the function 
        calculates the number of components that explain 95% of the variance.
    verbose: int, default=1 
       
    **pca_kws : dict
        Additional keyword arguments passed to the PCA constructor.
        
    Returns
    -------
    int
        Number of principal components that explain at least 95% of the variance.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.analysis.dimensionality import get_most_variance_component
    >>> X = load_iris().data
    >>> print(get_most_variance_component(X))
    2

    Notes
    -----
    This function is particularly useful in scenarios where you need to
    reduce dimensionality but are unsure about the number of dimensions
    to retain to preserve significant variance.
    """
    if n_components is None:
        _logger.info('`n_components` is not given. Determining components'
                     ' to capture 95% variance.')
        pca = PCA(**pca_kws)
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        # Index of the minimum number of components
        d = np.argmax(cumsum >= 0.95) + 1  
    else:
        pca = PCA(n_components=n_components, **pca_kws)
        pca.fit(X)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= 0.95) + 1
        if d > n_components:
            warnings.warn(f'Provided n_components={n_components} is less than'
                          f' required {d} to capture 95% variance.',
                          UserWarning)
            _logger.info(f'Resetting number of components to {d}, which'
                         ' captures 95% variance.')
    if verbose: 
        summary =ReportFactory().add_recommendations( 
            (f"Number of components ('n_components') reset to [{d}] as the"
             "  most representative variance (95%) in the dataset."
             ), 
            keys ="N-components",
            max_char_text=90 ) 
            
        print(summary)
    
    return d

def set_axes_and_feature_importances(Obj, X):
    """
    Sets the number of axes and feature importances on an object if applicable,
    especially when `X` is a pandas DataFrame.

    Parameters
    ----------
    Obj : object
        The object (typically a PCA model) to set attributes on.
    X : NDArray | DataFrame
        Data used to derive feature importances, expected to be a DataFrame
        for feature name extraction.

    Returns
    -------
    NDArray | object
        The modified object with new attributes set, or NDArray if only
        feature importances are applicable.

    Raises
    ------
    AttributeError
        If the necessary attributes are not found and cannot be set as expected.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.datasets import load_iris
    >>> from gofast.analysis.dimensionality import set_axes_and_feature_importances
    >>> X = pd.DataFrame(load_iris().data, 
                         columns=['sepal length', 'sepal width', 
                                  'petal length', 'petal width'])
    >>> pca = PCA().fit(X)
    >>> set_axes_and_feature_importances(pca, X)
    """
    try:
        setattr(Obj, 'n_axes', getattr(Obj, 'n_components_', 
                                       getattr(Obj, 'n_components', None)))
    except AttributeError as e:
        error_msg = ( 
            f"{Obj.__class__.__name__} does not have"
            " 'n_components_' or 'n_components'."
            )
        warnings.warn(error_msg)
        _logger.error(error_msg)
        raise AttributeError(error_msg) from e

    # Handling feature importances for DataFrame input
    if isinstance(X, pd.DataFrame):
        try:
            pca_components_ = getattr(Obj, 'components_')
            Obj.feature_importances_ = find_f_importances(
                np.array(X.columns), pca_components_, Obj.n_axes)
        except AttributeError as e:
            obj_type = ( 'KernelPCA' if hasattr(Obj, 'kernel') 
                        else 'LocallyLinearEmbedding' if hasattr(Obj, 'nbrs_') else ''
                        )
            error_msg =(  f"{Obj.__class__.__name__} does not have 'components_'"
                        " attribute required for calculating feature importances."
                        )
            warnings.warn(error_msg)
            _logger.error(
                f"Attempt to access 'components_' failed in"
                f" {obj_type or Obj.__class__.__name__}. Exception details: {str(e)}")
            setattr(Obj, 'feature_importances_', None)
            return Obj

    return Obj



    
    
    
    
    
    
    
    
    
    
    





