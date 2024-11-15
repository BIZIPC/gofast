.. _decomposition:

Decomposition
=============

.. currentmodule:: gofast.analysis.decomposition

The :mod:`gofast.analysis.decomposition` module provides tools for performing matrix 
decomposition and Principal Component Analysis (PCA), among other dimensionality 
reduction techniques. These tools are essential for understanding the underlying 
structure of data, reducing the dimensionality of datasets, and extracting the most 
informative features.

Key Features
------------
- **Eigen Decomposition**: Compute eigenvalues and eigenvectors of covariance or 
  correlation matrices.
- **Variance Ratio**: Calculate the explained variance ratio to determine the importance 
  of each principal component.
- **Transformation Matrix**: Construct transformation matrices to project data into 
  lower-dimensional spaces.

Function Descriptions
---------------------

get_eigen_components
~~~~~~~~~~~~~~~~~~~~

Computes the eigenvalues and eigenvectors of the covariance or correlation matrix 
of the dataset X.

This function extracts both eigenvalues and eigenvectors, which are fundamental 
components in many linear algebra and data analysis applications, providing a 
basis for Principal Component Analysis (PCA).

Let :math:`X` be the data matrix of shape (n_samples, n_features). The covariance 
matrix :math:`\Sigma` is computed as:

.. math::

    \Sigma = \frac{1}{n-1} X^T X

The eigenvalues :math:`\lambda_i` and eigenvectors :math:`v_i` are obtained by 
solving the equation:

.. math::

    \Sigma v_i = \lambda_i v_i

Examples
--------
Example 1:

.. code-block:: python

    import numpy as np 
    from gofast.analysis.decomposition import get_eigen_components

    X = np.random.rand(100, 5)
    eigen_vals, eigen_vecs, X_transformed = get_eigen_components(X)

Example 2:

.. code-block:: python

    from sklearn.datasets import load_iris
    from gofast.analysis.decomposition import get_eigen_components

    X = load_iris().data
    eigen_vals, eigen_vecs, _ = get_eigen_components(X, scale=True, method='covariance')
    print(eigen_vals, eigen_vecs)

Principal components will have the largest variance under the constraint that these components 
are uncorrelated (orthogonal) to each other, even if the input features are correlated. PCA directions 
are highly sensitive to data scaling. It is essential to standardize features prior to PCA.

get_total_variance_ratio
~~~~~~~~~~~~~~~~~~~~~~~~

Compute the total variance ratio.

This function calculates the ratio of each eigenvalue to the total sum of eigenvalues, 
representing the proportion of variance explained by each principal component. The cumulative 
sum of these ratios provides insight into the amount of variance captured as we increase 
the number of principal components considered.

The explained variance ratio for each eigenvalue :math:`\lambda_i` is given by:

.. math::

    \text{explained_variance_ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}

The cumulative explained variance ratio is the cumulative sum of the individual explained 
variance ratios.

Examples
--------
Example 1:

.. code-block:: python

    from gofast.analysis.decomposition import get_total_variance_ratio
    from sklearn.datasets import load_iris

    X = load_iris().data
    cum_var_exp = get_total_variance_ratio(X, view=True)
    print(cum_var_exp)

Example 2:

.. code-block:: python

    from gofast.datasets import fetch_data 
    from gofast.analysis.decomposition import get_total_variance_ratio

    data = fetch_data("bagoue analyses")
    y = data.flow
    X = data.drop(columns='flow')
    X = SimpleImputer().fit_transform(X)
    cum_var = get_total_variance_ratio(X, view=True)
    print(cum_var)

The explained variance ratio is a useful metric for dimensionality reduction, 
especially when deciding how many principal components to retain to preserve 
a significant amount of information about the original dataset.

transform_to_principal_components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transforms the dataset X into new principal components derived from the 
eigen decomposition of the covariance matrix of X.

Let :math:`X` be the data matrix of shape (n_samples, n_features). The eigenvalues 
:math:`\lambda_i` and eigenvectors :math:`v_i` are computed as:

.. math::

    \Sigma v_i = \lambda_i v_i

The transformation matrix :math:`W` is constructed from the top `n_components` eigenvectors:

.. math::

    W = [v_1, v_2, \ldots, v_{n_components}]

The dataset is then projected into the new subspace as:

.. math::

    X_{\text{transformed}} = X W

This function is particularly useful in scenarios where high-dimensional data 
needs to be visualized or analyzed in a lower-dimensional space. By projecting 
the data onto the principal components that explain the most variance, significant 
patterns and structures in the data can often be more easily identified.

Examples
--------
Example 1:

.. code-block:: python

    from sklearn.datasets import load_iris
    from gofast.analysis.decomposition import transform_to_principal_components

    X, y = load_iris(return_X_y=True)
    X_transformed = transform_to_principal_components(X, y, positive_class=1, view=True)
    print(X_transformed.shape)

Example 2:

.. code-block:: python

    from gofast.datasets import fetch_data 
    from gofast.analysis.decomposition import transform_to_principal_components

    data = fetch_data("bagoue analyses")
    y = data.flow
    X = data.drop(columns='flow')
    X = SimpleImputer().fit_transform(X)
    X_transf = transform_to_principal_components(X, y=y, positive_class=2, view=True)
    print(X_transf[0])

plot_decision_regions
~~~~~~~~~~~~~~~~~~~~~

Visualize decision regions for datasets transformed via PCA, showing how 
a classifier divides the feature space.

After transforming the dataset :math:`X` into the principal component space, the 
classifier is trained on this reduced space. The decision regions are visualized 
by predicting class labels on a grid and plotting these predictions to show the 
decision boundaries.

This function applies PCA to reduce dimensionality before plotting, which simplifies 
the visualization of the decision regions. It is particularly useful for visualizing 
the effectiveness of classification boundaries in a lower-dimensional space.

Examples
--------
Example 1:

.. code-block:: python

    from gofast.datasets import fetch_data
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from gofast.tools.baseutils import select_features
    from gofast.analysis.decomposition import plot_decision_regions

    data = fetch_data("bagoue analyses")
    y = data['flow']
    X = select_features(data.drop(columns='flow'), include='number')
    X = StandardScaler().fit_transform(X)
    lr_clf = LogisticRegression(random_state=42)
    plot_decision_regions(X, y, clf=lr_clf, n_components=2, view='X', split=True)

linear_discriminant_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linear Discriminant Analysis (LDA).

LDA is used as a technique for feature extraction to increase the 
computational efficiency and reduce the degree of overfitting due to the 
curse of dimensionality in non-regularized models. The general concept  
behind LDA is very similar to the Principal Component Analysis (PCA), but  
whereas PCA attempts to find the orthogonal component axes of minimum 
variance in a dataset, the goal in LDA is to find the features subspace 
that optimize class separability.

Transforms the dataset X into a new feature space defined by the linear discriminants.

The main steps required to perform LDA are summarized below: 

1. Standardize the d-dimensional datasets (d is the number of features).
2. For each class, compute the d-dimensional mean vectors. For each mean feature value, :math:`\mu_m` with respect to the examples of class :math:`i`:

.. math:: 

    m_i = \\frac{1}{n_i} \\sum{x\\in D_i} x_m 

3. Construct the within-class scatter matrix :math:`S_W` and the between-class scatter matrix :math:`S_B`. Individual scatter matrices are scaled :math:`S_i` before summing them up as scatter matrix :math:`S_W`:

.. math:: 

    S_W = \\sum_{i=1}^{c} S_i, \\quad \\text{where} \\quad S_i = \\sum_{x \\in D_i} (x - m_i)(x - m_i)^T

The between-class scatter matrix :math:`S_B` is computed as:

.. math:: 

    S_B = \\sum_{i=1}^{c} n_i (m_i - m) (m_i - m)^T 

where :math:`m` is the overall mean computed from all classes.

4. Compute the eigenvectors and corresponding eigenvalues of the matrix :math:`S_W^{-1}S_B`.
5. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
6. Choose the :math:`k` eigenvectors that correspond to the :math:`k` largest eigenvalues to construct the :math:`d \\times k` transformation matrix :math:`W`; the eigenvectors are the columns of this matrix.
7. Project the examples onto the new feature subspace using the transformation matrix :math:`W`.

Examples
--------
Example 1:

.. code-block:: python

    from gofast.datasets import fetch_data
    from sklearn.impute import SimpleImputer
    from gofast.analysis.decomposition import linear_discriminant_analysis

    data = fetch_data("bagoue analyses")
    y = data.flow
    X = data.drop(columns='flow')
    X = SimpleImputer().fit_transform(X)
    X_transformed = linear_discriminant_analysis(X, y, view=True)
    print(X_transformed.shape)

Example 2:

.. code-block:: python

    from sklearn.datasets import load_iris
    from gofast.analysis.decomposition import linear_discriminant_analysis

    X, y = load_iris(return_X_y=True)
    X_transformed = linear_discriminant_analysis(X, y, n_components=3, view=True)
    print(X_transformed.shape)

LDA is particularly useful in scenarios where we aim to maximize class separability in a lower-dimensional space. It is often employed 
in supervised learning to preprocess data, enhancing the performance of classification algorithms.

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
