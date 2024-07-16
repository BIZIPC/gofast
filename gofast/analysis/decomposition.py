# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Steps behing the principal component analysis (PCA) and matrices decomposition 
"""

from warnings import warn 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 

from ..api.docstring import _core_docs 
from ..backends.selector import BackendSelector 
from ..tools.coreutils import _assert_all_types
from ..tools.validator import validate_positive_integer 
from ..tools.validator import check_array, parameter_validator   

__all__=["get_eigen_components", "plot_decision_regions", 
    "transform_to_principal_components", "get_total_variance_ratio" , 
    "linear_discriminant_analysis", "get_transformation_matrix"
    ]

def get_eigen_components(
        X, scale: bool = True, method: str = 'covariance', backend: str='numpy'):
    
    X = check_array(X)
    method = parameter_validator(
        "method", target_strs={'covariance', 'correlation'})(method)
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Select the matrix calculation method
    if method == 'covariance':
        matrix = np.cov(X, rowvar=False)
    elif method == 'correlation':
        matrix = np.corrcoef(X, rowvar=False)

    # Eigen decomposition based on the selected backend
    backend_selector = BackendSelector(preferred_backend=backend)
    backend = backend_selector.get_backend()
    if backend.__class__.__name__ in ['NumpyBackend', 'numpy']:
        eigen_vals, eigen_vecs = np.linalg.eigh(matrix)
    else:
        eigen_vals, eigen_vecs = backend.eig(matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_indices]
    eigen_vecs = eigen_vecs[:, sorted_indices]

    return eigen_vals, eigen_vecs, X

get_eigen_components.__doc__ = """\
Computes the eigenvalues and eigenvectors of the covariance or correlation
matrix of the dataset X.

Function extracts both eigenvalues and eigenvectors, which are fundamental 
components in many linear algebra and data analysis applications, providing a 
basis for Principal Component Analysis (PCA).

Parameters
----------
{params.X}

scale : bool, default=True
    If True, standardize the data before computing eigenvalues and eigenvectors.
method : str, default='covariance'
    Method to use for generating the matrix from which eigenvalues and eigenvectors
    are computed. Options are 'covariance' for covariance matrix and 'correlation'
    for correlation matrix.
    
backend : str, optional
    The computational backend to use. Defaults to 'numpy'. Other options are 
    'scipy', 'cupy', etc., depending on what's available.    
Returns
-------
tuple
    A tuple containing eigenvalues, eigenvectors, and the scaled (standardized) 
    feature matrix:
    - eigen_vals:  Eigenvalues in descending order. The eigenvalues from
      the covariance matrix of X if `method` is ``covariance``.
    - eigen_vecs: Eigenvectors corresponding to the eigenvalues, 
      each column is an eigenvector..
    - Xsc: The possibly scaled and transformed input data. the standardized 
      version of the input dataset X if 'scale' is ``True``.

Examples
--------
>>> import numpy as np 
>>> from sklearn.impute import SimpleImputer
>>> from sklearn.datasets import load_iris
>>> from gofast.tools.baseutils import select_features
>>> from gofast.datasets import fetch_data
>>> from gofast.analysis.decomposition import get_eigen_components

>>> X = np.random.rand(100, 5)
>>> eigen_vals, eigen_vecs, X_transformed = get_eigen_components(X)

>>> data = fetch_data("bagoue analyses")  # Encoded flow categories
>>> y = data['flow']
>>> X = data.drop(columns='flow')
>>> # Select the numerical features
>>> X = select_features(X, include='number')
>>> # Impute the missing data
>>> X = SimpleImputer().fit_transform(X)
>>> eigval, eigvecs, _ = get_eigen_components(X)
>>> print(eigval)
[1.97788909 1.34186216 1.14311674 1.02424284 0.94346533 0.92781335
 0.75249407 0.68835847 0.22168818]

>>> X = load_iris().data
>>> eigen_vals, eigen_vecs, _ = get_eigen_components(X, scale=True, method='covariance')
>>> eigen_vals.shape, eigen_vecs.shape
((4,), (4, 4))

Notes
-----
All subsequent principal components (PCs) will have the largest variance 
under the constraint that these components are uncorrelated (orthogonal) 
to each other, even if the input features are correlated. As a result, the 
principal components will be mutually orthogonal.

Note: PCA directions are highly sensitive to data scaling. It is essential 
to standardize features prior to PCA, especially if the features were measured 
on different scales and equal importance is to be assigned to all features.

The numpy function designed to operate on symmetric and non-symmetric square 
matrices may return complex eigenvalues in certain cases. For symmetric 
matrices such as covariance matrices, the `numpy.linalg.eigh` function is 
recommended as it is numerically more stable and always returns real 
eigenvalues.
""".format(params=_core_docs["params"])
   
def get_total_variance_ratio (X, view =False): 
    eigen_vals, eigen_vcs, _ = get_eigen_components(X)
    tot =sum(eigen_vals)
    # sorting the eigen values by decreasing 
    # order to rank the eigen_vectors
    var_exp = list(map( lambda x: x/ tot , sorted (eigen_vals, reverse =True)))
    #var_exp = [(i/tot) for i in sorted (eigen_vals, reverse =True)]
    cum_var_exp = np.cumsum (var_exp)
    if view: 
        plt.bar (range(1, len(eigen_vals)+1), var_exp , alpha =.5, 
                 align='center', label ='Individual explained variance')
        plt.step (range(1, len(eigen_vals)+1), cum_var_exp, where ='mid', 
                  label="Cumulative explained variance")
        plt.ylabel ("Explained variance ratio")
        plt.xlabel ('Principal component analysis')
        plt.legend (loc ='best')
        plt.tight_layout()
        plt.show () 
    
    return cum_var_exp 
    
get_total_variance_ratio.__doc__ = """\
Compute the total variance ratio.

This function calculates the ratio of each eigenvalue to the total sum of 
eigenvalues, representing the proportion of variance explained by each principal 
component. The cumulative sum of these ratios provides insight into the amount 
of variance captured as we increase the number of principal components considered.

Is the ratio of an eigenvalues :math:`\\lambda_j`, as simply the fraction of 
and eigen value, :math:`\\lambda_j` and the total sum of the eigen values as: 
 
.. math:: 
    
    \\text{explained_variance_ratio}= \\frac{\\lambda_j}{\\sum{j=1}^{d} \\lambda_j}
    
Using numpy cumsum function, we can then calculate the cumulative sum of 
explained variance which can be plot if `plot` is set to ``True`` via 
matplotlib set function.    

Parameters
----------
X : ndarray, shape (M, N)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.

view : bool, default=False
    If True, plots the individual and cumulative explained variances to provide 
    a visual representation of the variance ratio explained by the principal 
    components.

Returns
-------
cum_var_exp : ndarray
    Cumulative explained variance ratio. This array represents the cumulative 
    sum of explained variances, providing insight into how many components are 
    required to explain a certain proportion of the total variance.

Examples
--------
>>> from gofast.analysis.decomposition import get_total_variance_ratio
>>> from sklearn.datasets import load_iris
>>> X = load_iris().data
>>> cum_var_exp = get_total_variance_ratio(X, view=True)
>>> print(cum_var_exp)
array([0.92461872, 0.97768521, 0.99478782, 1.        ])

>>> # Use the X value in the example of `extract_eigen_components` function   
>>> from gofast.datasets import fetch_data 
>>> data= fetch_data("bagoue analyses") # encoded flow categories 
>>> y = data.flow ; X= data.drop(columns='flow') 
>>> # select the numerical features 
>>> X =select_features(X, include ='number')
>>> # imputed the missing data 
>>> X = SimpleImputer().fit_transform(X)
>>> cum_var = get_total_variance_ratio(X, view=True)
>>> cum_var
... array([0.26091916, 0.44042728, 0.57625294, 0.69786032, 0.80479823,
       0.89379712, 0.97474381, 1.        ])

Notes
-----
The explained variance ratio is a useful metric for dimensionality reduction, 
especially when deciding how many principal components to retain in order to 
preserve a significant amount of information about the original dataset.
"""

def get_transformation_matrix(
    n_components: int, 
    eigen_vals: np.ndarray = None, 
    eigen_vecs: np.ndarray = None,  
    X: np.ndarray = None
) -> np.ndarray:
    """
    Construct the transformation matrix from eigenpairs for a specified
    number of principal components. This matrix can then be used to 
    transform data into a lower-dimensional space defined by the 
    principal components.

    Parameters
    ----------
    n_components : int
        The number of principal components to include in the transformation
        matrix.
    eigen_vals : np.ndarray, optional
        Array of eigenvalues from the covariance matrix. Required if `X` is not
        provided.
    eigen_vecs : np.ndarray, optional
        Corresponding matrix of eigenvectors. Required if `X` is not provided.
    X : np.ndarray, optional
        Data matrix from which to compute eigenvalues and eigenvectors if they
        are not directly provided. Shape should be (n_samples, n_features).

    Returns
    -------
    np.ndarray
        The transformation matrix composed of eigenvectors corresponding
        to the top `n_components` eigenvalues. Shape is (n_features, n_components).

    Raises
    ------
    ValueError
        If required eigenvalues and eigenvectors are not provided, or if
        `n_components` exceeds the number of available eigenvalues.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> eigen_vals = np.array([2.0, 0.5])
    >>> eigen_vecs = np.array([[0.6, -0.8], [0.8, 0.6]])
    >>> get_transformation_matrix(1, eigen_vals, eigen_vecs)
    array([[0.6],
           [0.8]])

    Notes
    -----
    - It is crucial to ensure that the eigenvalues and eigenvectors are
      correctly paired and sorted in descending order of the eigenvalues
      before constructing the transformation matrix.
    - This function assumes that eigenvectors are provided as columns in
      `eigen_vecs`.
    - The function is sensitive to the order and correctness of eigenvalues
      and eigenvectors, as the transformation matrix directly influences
      the resulting dimensionality reduction.
    """
    
    # Validate input parameters
    n_components = validate_positive_integer(n_components, "n_components")
    if X is not None:
        eigen_vals, eigen_vecs, _ = get_eigen_components(X)
    elif eigen_vals is None or eigen_vecs is None:
        raise ValueError("Either both eigenvalues and eigenvectors must be provided, "
                         "or a matrix X from which they can be computed.")
    
    if eigen_vals is None or eigen_vecs is None:
        raise ValueError("Eigenvalues and eigenvectors cannot be None.")

    eigen_vals= np.asarray( eigen_vals) 
    eigen_vecs = np.asarray (eigen_vecs)
    # Ensure the length of eigenvalues and eigenvectors match
    if len(eigen_vals) != eigen_vecs.shape[1]:
        raise ValueError("Mismatch between the number of eigenvalues and the number "
                         "of columns in eigenvector matrix.")
    
    if n_components > len(eigen_vals):
        raise ValueError("n_components exceeds the number of available eigenvalues")

    # Sort eigenpairs by descending order of eigenvalues
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # Select the top 'n_components' eigenvectors to form the transformation matrix
    w_matrix = np.hstack([pair[1][:, np.newaxis] for pair in eigen_pairs[:n_components]])

    return w_matrix

def transform_to_principal_components (
        X, y=None, n_components =2, positive_class=1,
        view =False ):
    
    # select k vectors which correspond to the k largest 
    # eigenvalues , where k is the dimesionality of the new  
    # subspace (k<=d) 
    eigen_vals, eigen_vecs, X = get_eigen_components(X)
    # -> sorting the eigen values by decreasing order 
    eigen_pairs = [ (np.abs(eigen_vals[i]) , eigen_vecs[:, i]) 
                   for i in range(len(eigen_vals))]
    eigen_pairs.sort(key =lambda k :k[0], reverse =True)
    # collect two eigen vectors that correspond to the two largest 
    # eigenvalues to capture about 60% of the variance of this datasets
    
    # Check if the requested number of components is implemented
    if n_components > 2:
        w = get_transformation_matrix(n_components, eigen_vals, eigen_vecs )
    else:
        w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    # w= np.hstack((eigen_pairs[0][1][:, np.newaxis], 
    #              eigen_pairs [1][1][:, np.newaxis])
    #              ) 
    # In pratice the number of principal component has to be 
    # determined by a tradeoff between computational efficiency 
    # and the performance of the classifier.
    #-> transform X onto a PCA subspace( the pc one on two)
    X_transf = X.dot(w)
    
    if view: 
        from ..plot.utils import make_mpl_properties
        if y is None: 
            raise TypeError("Missing the target `y`")
        # markers = tuple (D_MARKERS [:len(np.unique (y))])
        # colors = tuple (D_COLORS [:len(np.unique (y))])    
        colors = make_mpl_properties(len(np.unique (y)))
        markers = make_mpl_properties (
            len(np.unique (y)), 'markers') 
        
        if positive_class not in np.unique (y): 
            raise ValueError( f"'{positive_class}' does not match any label "
                             "of the class. The positive class must be an  "
                             "integer label within the class values"
                             )
        for l, c, m in zip(np.unique (y),colors, markers ):
            plt.scatter(X_transf[y==positive_class, 0],
                        X_transf[y==positive_class, 1], c= c,
                        label =l, marker=m)
        plt.xlabel ('PC1')
        plt.ylabel ('PC2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show() 
        
    return X_transf 
  
transform_to_principal_components.__doc__ = """\
Transforms the dataset X into new principal components derived from the 
eigen decomposition of the covariance matrix of X.

Parameters
-----------
{params.X}
{params.y}
n_components: int, default=2
    The number of principal components to retain. Specifies the dimensionality 
    of the transformed data in terms of the most significant principal axes.
positive_class: int
    Specifies the class label that should be considered as the positive class 
    in binary classification tasks. This is used primarily for visualization 
    to distinguish between the positive and other classes.
view: bool, default={{'False'}}
    If set to True, visualizes the transformed principal components along with 
    the separation between the positive class and other classes. This is helpful 
    for assessing the quality of the transformation in terms of class separability.

Returns
---------
X_transf: ndarray
    The transformed dataset where each sample is now represented in the reduced 
    principal component space. This transformation often helps in visualizing 
    high-dimensional data in a two-dimensional or three-dimensional space.

Examples
---------
>>> from gofast.analysis.decomposition import transform_to_principal_components
>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> X_transformed = transform_to_principal_components(X, y, positive_class=1, view=True)
>>> print(X_transformed.shape)
(150, 2)

>>> # Use the X, y value in the example of `get_eigen_components` function  
>>> Xtransf = transform_to_principal_components(X, y=y,  positive_class = 2 , view =True)
>>> Xtransf[0] 
array([-1.0168034 ,  2.56417088])

Notes
-----
This function is particularly useful in scenarios where high-dimensional data 
needs to be visualized or analyzed in a lower-dimensional space. By projecting 
the data onto the principal components that explain the most variance, 
significant patterns and structures in the data can often be more easily 
identified.
""".format(params=_core_docs["params"])

def _decision_region (X, y, clf, resolution =.02 , ax =None ): 
    """ visuzalize the decision region """
    from ..plot.utils import make_mpl_properties
    # setup marker generator and colors map 
    colors = tuple (make_mpl_properties(len(np.unique (y))))
    markers = tuple (make_mpl_properties (
        len(np.unique (y)), 'markers') )
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface 
    x1_min , x1_max = X[:, 0].min() -1, X[:, 0].max() +1 
    x2_min , x2_max = X[:, 1].min() -1, X[:, 1].max() +1 
    
    xx1 , xx2 = np.meshgrid(np.arange (x1_min, x1_max, resolution), 
                            np.arange (x2_min, x2_max, resolution)
                            )
    z= clf.predict(np.array ([xx1.ravel(), xx2.ravel()]).T)
    z= z.reshape (xx1.shape)
    if ax is None: 
        fig, ax= plt.subplots()
    ax.contourf (xx1, xx2, z, alpha =.4, cmap =cmap )
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    # plot the examples by classes 
    for idx, cl in enumerate (np.unique (y)): 
        ax.scatter (x= X[y ==cl, 0] , y = X[y==cl, 1], 
                     alpha =.6 , 
                     color = cmap(idx), 
                     edgecolors='black', 
                     marker = markers[idx], 
                     label=cl 
                     ) 
    return ax 
        
def plot_decision_regions (
    X, y, clf, Xt =None, yt=None, random_state = 42, test_size = .3 , 
    scaling =True, split =False,  n_components =2 , view ='X',
    resolution =.02, return_expl_variance_ratio =False, return_X =False, 
    axe =None, 
    **kws 
    ): 
    view = str(view).lower().strip()
    if  view in ('xt', 'test'): 
        view ='test'
    elif view in ('x', 'train'):
        view = 'train'
    else: view =None 
    
    if split : 
        X, Xt, y, yt = train_test_split(X, y, random_state =random_state, 
                                        test_size =test_size, **kws)
       
    pca = PCA (n_components = n_components)
    # scale data 
    sc = StandardScaler() 
    X =  sc.fit_transform(X) if scaling else X 
    if Xt is not None: 
        Xt =  sc.transform(Xt) if scaling else Xt 
        
    # dimension reduction 
    X_pca = pca.fit_transform(X)
    Xt_pca = pca.transform (Xt) if ( Xt is not None)  else None 
    # fitting the classifier clf model on the reduced datasets 
    clf.fit(X_pca, y )
    # now plot the decision regions 
    if view is not None: 
        if view =='train': 
            ax = _decision_region(
                X_pca, y, clf = clf,resolution = resolution, ax= axe) 
        if view =='test':
            if Xt_pca is None: 
                raise TypeError("Cannot plot missing test sets (Xt, yt)")
            ax = _decision_region(Xt_pca, yt, clf=clf, resolution =resolution,
                                  ax =axe )
        ax.set_xlabel("PC1")
        ax.set_ylabel ("PC2")
        ax.legend (loc= 'lower left')
        plt.show ()
        
    if return_expl_variance_ratio : 
        pca =PCA(n_components =None )
        X_pca = pca.fit_transform(X)
        
        return pca.explained_variance_ratio_ 
    
    return X_pca if return_X else  ax 
   
plot_decision_regions.__doc__ = """\
Visualize decision regions for datasets transformed via PCA, showing how 
a classifier divides the feature space.

Parameters
-----------
{params.X}
{params.y}
{params.Xt}
{params.yt}
{params.clf}
random_state : int, default={{42}}
    The seed used by the random number generator for data shuffling.
test_size : float, default=0.3
    Proportion of the dataset to include in the test split.
split : bool, default=False
    If True, splits the dataset into a training set and a test set using 
    the entire dataset provided in (X, y).
n_components : int or float, default={{2}}
    The number of principal components to retain for PCA. If a float between 
    0.0 and 1.0 is provided, it specifies the fraction of variance that must 
    be retained by the PCA components.
view :{{'X', 'Xt', None}}, default={{None}}
    Specifies the subset for which to visualize decision regions:
    'X' for training data, 'Xt' for test data, or None to disable visualization.
resolution : float, default={{0.02}}
    The resolution of the grid used in plotting decision regions, determining 
    the granularity of the meshgrid.
return_expl_variance_ratio : bool, default=False
    If True, returns the explained variance ratio of the PCA transformation.
return_X : bool, default=False
    If True, returns the PCA-transformed dataset.
ax : Matplotlib.Axes, optional
    A custom Matplotlib Axes object for the plot; if not provided, a new one 
    will be created.
    
kws : dict, optional
    Additional keyword arguments passed to the scikit-learn function
    sklearn.model_selection.train_test_split.
Returns
-------
X_pca : ndarray or array-like
    The PCA-transformed dataset, or the explained variance ratios if 
    `return_X` is True or `return_expl_variance_ratio` is True.
ax : Matplotlib.Axes, optional
    The Matplotlib Axes object useful for further customization of the plot.

Examples
--------
>>> from gofast.datasets import fetch_data
>>> from sklearn.preprocessing import StandardScaler
>>> from sklearn.linear_model import LogisticRegression
>>> from gofast.tools.baseutils import select_features
>>> from gofast.analysis.decomposition import plot_decision_regions
>>> data = fetch_data("bagoue analyses")
>>> y = data['flow']
>>> X = select_features(data.drop(columns='flow'), include='number')
>>> X = StandardScaler().fit_transform(X)
>>> lr_clf = LogisticRegression(random_state=42)
>>> plot_decision_regions(X, y, clf=lr_clf, n_components=2, view='X', split=True)

Notes
-----
The function applies PCA to reduce dimensionality before plotting, which simplifies 
the visualization of the decision regions. It assumes that the classifier and PCA 
can handle the dimensionality specified by `n_components`. This function is particularly 
useful for visualizing the effectiveness of classification boundaries in a 
lower-dimensional space.
""".format(params=_core_docs["params"])

def linear_discriminant_analysis (
        X, y, n_components = 2 , view=False, verbose = 0  , return_X=True, 
 ): 
    n_components = int (_assert_all_types (n_components, int, float ))
    # standardize the features 
    eigen_vals, eigen_vcs, X = get_eigen_components(X)
    # compute the mean vectors which will use to 
    # construct the within classes scatter matrix 
    np.set_printoptions(precision=4) 
    mean_vecs =list() 
    for label in range (1, len(np.unique (y))): 
        mean_vecs.append (np.mean(X[y==label], axis =0 ))
        if verbose : 
            print('MV %s: %s\n' %(label , mean_vecs[label -1]))
            
    # compute the within class Sw using the mean vector; 
    # calculated by summing up the individual scatter matrices 
    
    d = X.shape [1]  # number of features
    SW =np.zeros ((d, d ))
    #==========================================================================
    # for label , mv in zip (range (1 , len(np.unique (y))), mean_vecs): 
    #     class_scatter = np.zeros ((d, d) )
    #     for row in X [y==label ]: 
    #         row, mv= row.reshape (d, 1), mv.reshape (d, 1)
    #         class_scatter += (row - mv ).dot(row - mv).T 
    #         SW += class_scatter 
    # if verbose : 
    #     print("within-class scatter matrix: %sx %s" % (
    #         SW.shape [0], SW.shape[1]))
    
    # the assumption that we are making when we are computing the  
    # scatter matrices  is that the labels in the training datasets 
    # are uniformly distributed. Howeverm, if we print the number 
    # of class labels, we see that this assumtions is violated. 
    # for instance : 
    # >>> print('class label distributions: %s' % np.bincount (y)[1:])
    #==========================================================================
    # the better way is to scale the individual scatter matrices, before 
    # sum them up as scatter matrix SW.  when dividing the scatter matrices 
    # by the number of classes examples , we can see that computig the  
    # scatter matrix is the same as covariance matrix 
    
    for label , mv in zip (range (1 , len(np.unique (y))), mean_vecs): 
        class_scatter =np.cov(X[y==label].T )
        SW += class_scatter 

    if verbose : 
        print("Scaled within-class scatter matrix: %sx %s" % (
            SW.shape [0], SW.shape[1]))
    # compute between classes scatter matrix SB 
    mean_overall = np.mean (X, axis = 0 )  # include the example from c classes 
    SB = np.zeros ((d, d)) 
    for i, mean_vec in enumerate( mean_vecs ): 
        n = X[y == i+1 , :].shape [0] 
        mean_vec = mean_vec.reshape (d, 1) # make column vector 
        mean_overall = mean_overall.reshape (d, 1)
        SB += n * (mean_vec - mean_overall ).dot ((mean_vec - mean_overall).T)
        
    if verbose : 
        if verbose : 
            print("Between within-class scatter matrix: %sx%s" % (
                SB.shape [0], SB.shape[1]))
    
    # select discriminant for the new feature subspace 
    eigen_vals, eigen_vecs = np.linalg.eig (np.linalg.inv (SW).dot(SB))
    # sort the eigen value in descending order 
    eigen_pairs = [ (np.abs (eigen_vals[i]), eigen_vecs [:, i])
                     for i in range(len(eigen_vals ))
                     ]
    eigen_pairs = sorted (eigen_pairs, key =lambda k: k[0], reverse =True )
    if verbose : 
        print("Eigen values in descending order:\n")
        for eig_val in eigen_pairs : 
            print(eig_val[0])
            
    if view : 
        tot = sum(eigen_vals.real )
        discr = [(i/tot ) for i in sorted (eigen_vals.real , reverse =True ) 
                 ]
        cum_discr = np.cumsum(discr) 
        plt.bar (range (1 , 1+ X.shape [1]), discr , alpha =.5 , 
                 align= 'center', label ='Individual "discriminability"'
                 )
        plt.step (range (1 , 1+ X.shape [1]), cum_discr , where ='mid', 
                  label ='Cumulative "discriminality"')
        plt.ylabel ('"Discriminability" ratio') 
        plt.xlabel ("Linear discriminants")
        plt.ylim ([ -0.1 , 1.1 ])
        plt.legend (loc ='best')
        plt.tight_layout() 
        plt.show () 

    # stack the two most discrimative eigen columns 
    # to create the transformation matrix W  for two components 
    # W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, 
    #                eigen_pairs[1][1][:, np.newaxis].real )
    #               )
    W0 = np.hstack(tuple (eigen_pairs[k][1][:, np.newaxis].real 
                          for k in range (len(eigen_pairs)))
                   ) 
    if n_components > W0.shape [1]: 
        warn (f"n_component '{n_components}' is larger than the most  "
               f"discriminant vector '{W0.shape [1]}'")
        n_components = W0.shape [1] 
        
    W = W0 [:, :n_components]
  
    if verbose: 
        print("07 first values  of W:\n", W[:5, :])
    # we can now transform the training dataset 
    # from  matrix  W 
    return  X.dot(W) if return_X else W 


linear_discriminant_analysis.__doc__ ="""\
Linear Discriminant Analysis `LDA`. 

LDA is used as a technique for feature extraction to increase the 
computational efficiency and reduce the degree of overfitting due to the 
curse of dimensionnality in non-regularized models. The general concept  
behind LDA is very similar to the principal component analysis (PCA), but  
whereas PCA attempts to find the orthogonal component axes of minimum 
variance in a dataset, the goal in LDA is to find the features subspace 
that optimize class separability. The main steps requiered to perform 
LDA are summarized below: 
    
* Standardize the d-dimensional datasets (d is the number of features)
* For each class , compute the d-dimensional mean vectors. Thus for 
  each mean feature value, :math:`\\mu_m` with respect to the examples 
  of class :math:`i`: 
        
  .. math:: 
      
      m_i = \\frac{1}{n_i} \\sum{x\\in D_i} x_m 
        
* Construct the between-class scatter matrix, :math:`S_B` and the 
  within class scatter matrix, :math:`S_W`. Individual scatter matrices 
  are scalled :math:`S_i` before we sum them up as scatter 
  matrix :math:`S_W` as:
      
  .. math:: 
         
      \\sum{i} = \\frac{1}{n_i}S_i 
        
      \\sum{i} = \\frac{1}{n_i} \\sum{x\\in D_i} (x-m_i)(x-m_i)^T
            
  The within-class is also called the covariance matrix, thus we can compute 
  the between class scatter_matrix :math:`S_B`. 
  
  .. math:: 
      
      S_B= \\sum{i}^{n_i}(m_i-m) (m_i-m)^T 
        
  where :math:`m` is the overall mean that is computed , including examples 
  from all classes. 

* Compute the eigenvectors and corresponding eigenvalues of the matrix 
  :math:`S_W^{-1}S_B`. 
* Sort the eigenvalues by decreasing order to rank the corresponding 
  eigenvectors 
* Choose the :math:`k` eigenvectors that correspond to the :math:`k` 
  largest eigenvalues to construct :math:`dxk`-dimensional 
  transformation matrix, :math:`W`; the eigenvectors are the columns 
  of this matrix. 
* project the examples onto the new_features subspaces using the 
  transformation matrix :math:`W`.  
    
Parameters 
-----------
X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
    Training set; Denotes data that is observed at training and 
    prediction time, used as independent variables in learning. 
    When a matrix, each sample may be represented by a feature vector, 
    or a vector of precomputed (dis)similarity with each training 
    sample. :code:`X` may also not be a matrix, and may require a 
    feature extractor or a pairwise metric to turn it into one  before 
    learning a model.
    
y: Array-like, shape (M,)
    Training target; Represents dependent variables in supervised learning. y must 
    be available at training time but is typically unavailable during prediction. 
    It includes class labels which are crucial for directing the LDA projection.
  
n_components: int, optional, default=2
    The number of components (dimensions) to retain in the transformed output. 
    This parameter determines the number of linear discriminants (eigenvalues) 
    to consider based on their ability to maximize class separation.
    
view: bool, optional, default=False
    If True, displays a plot of the discriminability ratio which helps in 
    visualizing the contribution of each component to class separation.
    
verbose: int, optional, default=0
    Controls the verbosity; higher values provide more detailed output 
    (useful for debugging or learning the internal operations of the function).

return_X: bool, optional, default=True
    Determines the type of return value. If True, the function returns the 
    transformed dataset (X projected onto the discriminant vectors). If False, 
    it returns the matrix W of weights used for the transformation.
    
Returns
--------
X or W: ndarray
    Depending on the value of `return_X`, returns either the transformed dataset 
    X with dimensions reduced to `n_components` or the transformation matrix W 
    consisting of the top `n_components` discriminant vectors.
    
Examples
----------
>>> from gofast.datasets import fetch_data 
>>> from sklearn.impute import SimpleImputer
>>> from sklearn.linear_model import LogisticRegression
>>> from gofast.tools.baseutils import select_features
>>> from gofast.analysis.decomposition import linear_discriminant_analysis
>>> y = data.flow ; X= data.drop(columns='flow') 
>>> # select the numerical features 
>>> X =select_features(X, include ='number')
>>> # imputed the missing data 
>>> X = SimpleImputer().fit_transform(X)
>>> Xtr= linear_discriminant_analysis (X, y , view =True)

Notes
-----
LDA assumes that the data is approximately normally distributed per class and 
that the classes have similar covariance matrices. If these assumptions are 
not met, the effectiveness of LDA as a classifier may be reduced. In practice,
LDA is quite robust and can perform well even when the normality assumption is 
somewhat violated.

In addition to its utility in dimensionality reduction, LDA is often used as 
a linear classification technique. The directions of the axes resulting from LDA 
are used to separate classes linearly.

The `view` parameter utilizes matplotlib for generating plots; ensure matplotlib 
is installed and properly configured in your environment if using this feature.

"""


      