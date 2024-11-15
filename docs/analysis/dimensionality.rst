.. _dimensionality:

Dimensionality Reduction
========================

.. currentmodule:: gofast.analysis.dimensionality

The :mod:`gofast.analysis.dimensionality` module provides tools for reducing the number 
of dimensions in a dataset, enabling data visualization and pattern detection. This 
module includes implementations of various dimensionality reduction techniques such 
as PCA, Kernel PCA, Incremental PCA, and Locally Linear Embedding.

Key Features
------------
- **Principal Component Analysis (PCA):** Standard linear dimensionality reduction 
  technique.
- **Kernel PCA (kPCA):** Non-linear dimensionality reduction using kernel methods.
- **Incremental PCA (iPCA):** PCA that processes data in batches, useful for large 
  datasets.
- **Locally Linear Embedding (LLE):** Manifold learning technique for non-linear 
  dimensionality reduction.
- **Feature Importances:** Extracts and visualizes the importance of features based on 
  PCA components.
- **Projection Visualization:** Projects high-dimensional data onto lower dimensions 
  for visualization.

Function Descriptions
---------------------

nPCA
~~~~
Normal Principal Components Analysis (PCA).

Performs Principal Components Analysis (PCA), a popular linear dimensionality reduction 
technique. PCA identifies the hyperplane that lies closest to the data and projects the 
data onto it, reducing the number of dimensions while attempting to preserve as much 
variance as possible.

The PCA transformation is defined as:

.. math::

    Z = XW

Where:
- \( Z \) is the transformed data matrix.
- \( X \) is the original data matrix.
- \( W \) is the matrix of principal components.

The principal components are the eigenvectors of the covariance matrix \( C \) of \( X \):

.. math::

    C = \frac{1}{n} X^T X

Examples:
^^^^^^^^^

.. code-block:: python

    import numpy as np
    from gofast.analysis.dimensionality import nPCA
    from sklearn.datasets import load_iris

    X = load_iris().data
    pca_result = nPCA(X, n_components=0.95, view=True, return_X=False)
    print(pca_result.components_)
    print(pca_result.feature_importances_)

.. code-block:: python

    from gofast.datasets import fetch_data
    X = fetch_data('Bagoue analysed dataset')
    pca = nPCA(X, 0.95, n_axes=3, return_X=False)
    pca.components_
    pca.feature_importances_

get_feature_importances
~~~~~~~~~~~~~~~~~~~~~~~
Retrieves the feature importances from PCA components and optionally scales them by 
the explained variance ratio.

The feature importance for each component is calculated as:

.. math::

    I_j = \sum_{i=1}^n |W_{ij}|

Where:
- \( I_j \) is the importance of feature \( j \).
- \( W_{ij} \) is the weight of feature \( j \) in component \( i \).

If `scale_by_variance` is True, the feature importances are scaled by the explained 
variance ratio:

.. math::

    I_j = \sum_{i=1}^n |W_{ij}| \cdot \text{explained_variance}_i

Examples:
^^^^^^^^^

.. code-block:: python

    import numpy as np
    from gofast.analysis.dimensionality import get_feature_importances

    features = ['feature1', 'feature2', 'feature3']
    components = np.array([[0.5, -0.8, 0.3], [0.4, 0.9, -0.1]])
    explained_variances = np.array([0.6, 0.4])
    importances = get_feature_importances(components, features, 2, True, explained_variances, True)

.. code-block:: python

    import numpy as np
    from gofast.analysis.dimensionality import get_feature_importances

    components = np.array([[0.1, 0.3, 0.7], [0.4, 0.5, 0.1]])
    importances = get_feature_importances(components)
    

iPCA
~~~~
Incremental Principal Component Analysis (iPCA).

Incremental PCA allows for processing the dataset in mini-batches, which is beneficial 
for large datasets that do not fit into memory. It is also suitable for applications 
requiring online updates of the decomposition.

Once problem with the preceding implementation of PCA is that it requires the whole 
training set to fit in memory for the SVD algorithm to run. This is useful for large 
training sets, and also applying PCA online (i.e., on the fly as a new instance arrives).

**Mathematical Formulation:**

Incremental PCA works by updating the principal components with each new batch of data 
using the following formula:

.. math::

    C_i = \frac{1}{n} \sum_{j=1}^{n} x_{ij} x_{ij}^T

Where:
- \( C_i \) is the covariance matrix for the \( i \)-th batch.
- \( x_{ij} \) is the \( j \)-th sample in the \( i \)-th batch.
- \( n \) is the number of samples in each batch.

Examples:
^^^^^^^^^

.. code-block:: python

    from gofast.analysis.dimensionality import iPCA
    from gofast.datasets import fetch_data

    X = fetch_data('Bagoue analysed data')
    X_transf = iPCA(X, n_components=2, n_batches=100, view=True)

For large datasets or online learning scenarios, `iPCA` is preferable to standard PCA 
since it does not require loading the entire dataset into memory.

kPCA
~~~~
Kernel Principal Component Analysis (kPCA).

`kPCA` performs complex nonlinear projections for dimensionality reduction.

Commonly, the kernel trick is a mathematical technique that implicitly maps instances 
into a very high-dimensional space (called the feature space), enabling nonlinear 
classification or regression with SVMs. Recall that a linear decision boundary in the 
high-dimensional feature space corresponds to a complex nonlinear decision boundary in 
the original space.

The kernel PCA transformation is defined as:

.. math::

    K_{ij} = k(x_i, x_j)

Where:
- \( K \) is the kernel matrix.
- \( k \) is the kernel function (e.g., RBF, polynomial).

The transformed data is obtained by:

.. math::

    Z = K \alpha

Where:
- \( Z \) is the transformed data matrix.
- \( \alpha \) is the matrix of eigenvectors of the kernel matrix.

Examples:
^^^^^^^^^

.. code-block:: python

    from gofast.analysis.dimensionality import kPCA
    from gofast.datasets import fetch_data 

    X = fetch_data('Bagoue analysis data')
    Xtransf = kPCA(X, n_components=None, kernel='rbf', gamma=0.04, view=True)

`kPCA` is particularly useful for non-linear dimensionality reduction, allowing for 
more complex data structures to be represented in a reduced number of dimensions.

LLE
~~~
Locally Linear Embedding (LLE).

`LLE` is a nonlinear dimensionality reduction technique based on closest neighbors. It 
is another powerful non-linear dimensionality reduction (NLDR) technique. It is a 
manifold learning technique that does not rely on projections like `PCA`. LLE works by 
measuring how each training instance linearly relates to its closest neighbors, and 
then looking for a low-dimensional representation of the training set where these local 
relationships are best preserved.

The LLE transformation can be described as:

.. math::

    X = W Y

Where:
- \( X \) is the high-dimensional data.
- \( W \) is the weight matrix representing linear relationships with neighbors.
- \( Y \) is the low-dimensional embedding.

Examples:
^^^^^^^^^

.. code-block:: python

    from gofast.analysis.dimensionality import LLE
    from gofast.datasets import fetch_data 

    X = fetch_data('Bagoue analysed')
    lle_kws = {
        'n_components': 4, 
        'n_neighbors': 5
    }
    Xtransf = LLE(X, view=True, **lle_kws)

LLE yields good results especially for unrolling twisted manifolds, making it 
particularly effective when there is too much noise.

get_most_variance_component
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Determines the number of principal components that explain at least 95% of the variance 
in the dataset.

This function performs Principal Component Analysis (PCA) on the given dataset and 
returns the number of components that capture 95% of the variance.

The cumulative explained variance is calculated as:

.. math::

    \text{cumsum} = \text{cumsum}(\text{explained\_variance\_ratio\_})

Where:
- \(\text{explained\_variance\_ratio\_}\) is the variance explained by each principal 
  component.

Examples:
^^^^^^^^^

.. code-block:: python

    from sklearn.datasets import load_iris
    from gofast.analysis.dimensionality import get_most_variance_component

    X = load_iris().data
    n_components = get_most_variance_component(X)
    print(n_components)

This function is particularly useful in scenarios where you need to reduce 
dimensionality but are unsure about the number of dimensions to retain to preserve 
significant variance.

project_ndim_vs_explained_variance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Plots the number of dimensions against the explained variance ratio of a PCA object or 
similar.

This function provides a visual representation of the explained variance vs. the number 
of dimensions for PCA-related objects.

The cumulative explained variance is plotted as:

.. math::

    \text{cumsum} = \text{cumsum}(\text{explained\_variance\_ratio\_})

Where:
- \(\text{explained\_variance\_ratio\_}\) is the variance explained by each principal 
  component.

Examples:
^^^^^^^^^

.. code-block:: python

    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    from gofast.analysis.dimensionality import project_ndim_vs_explained_variance

    X = load_iris().data
    pca = PCA().fit(X)
    project_ndim_vs_explained_variance(pca)

Ensure that the object passed to this function is fitted and contains the 
`explained_variance_ratio_` attribute necessary for plotting.
