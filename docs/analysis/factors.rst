.. _factors:

Factors
=======

.. currentmodule:: gofast.analysis.factors

The :mod:`gofast.analysis.factors` module provides various functions for performing factor analysis. This includes methods like Maximum 
Likelihood Factor Analysis, Direct Oblimin Rotation, Varimax Rotation, and others. These methods are essential for data reduction and 
structure detection in large datasets, especially in the context of multivariate statistics.

Key Features
------------
- **Spectral Factor Analysis**: Extract factor loadings, common factors, and eigenvalues using spectral methods.
- **Promax Rotation**: Enhance factor interpretability through oblique rotations allowing factor correlations.
- **Ledoit-Wolf Score**: Use Ledoit-Wolf shrinkage estimator for robust covariance matrix estimation.
- **Noise Impact Evaluation**: Assess how noise affects the performance of dimensionality reduction techniques.
- **Scedastic Data Generation**: Create datasets with homoscedastic and heteroscedastic noise for model comparison.
- **Principal Axis Factoring**: Identify latent variables explaining observed correlations through PAF.
- **Varimax Rotation**: Apply orthogonal rotations to simplify factor loading structures.
- **Oblimin Rotation**: Perform oblique rotations to allow for correlated factors in factor analysis.
- **Hotelling's T-square Test**: Conduct multivariate hypothesis testing to compare group means.

Function Descriptions
---------------------

spectral_factor
~~~~~~~~~~~~~~~~

Perform Spectral Method in Factor Analysis on the given data.

The `spectral_factor` analysis function performs the Spectral Method in Factor Analysis on the input data, allowing you to obtain factor loadings, common factors, and eigenvalues.

Perform Spectral Factor Analysis on the given data to extract factor loadings, common factors, and associated eigenvalues using eigenvalue decomposition of the covariance matrix of the data.

.. math::
    C = \frac{1}{n_{\text{samples}}} X^T X

where :math:`X` is the data matrix centered by subtracting the mean of each feature. The decomposition is:

.. math::
    C = VDV^T

where :math:`V` is the matrix of eigenvectors (factor loadings) and :math:`D` is the diagonal matrix of eigenvalues. The factor loadings are obtained by:

.. math::
    L = V \sqrt{D}

and the common factors are calculated as:

.. math::
    F = X L

where :math:`L` is the matrix of loadings, and :math:`F` are the factors.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import spectral_factor
    data = np.array([[1.2, 2.3, 3.1],
                     [1.8, 3.5, 4.2],
                     [2.6, 4.0, 5.3],
                     [0.9, 1.7, 2.0]])
    loadings, factors, eigenvalues = spectral_factor(data, num_factors=2)
    print("Factor Loadings Matrix:")
    print(loadings)
    print("Common Factors Matrix:")
    print(factors)
    print("Eigenvalues:")
    print(eigenvalues)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import spectral_factor
    data = np.random.rand(100, 10)
    loadings, factors, eigenvalues = spectral_factor(data, num_factors=5)
    print("Factor Loadings Matrix:")
    print(loadings)
    print("Common Factors Matrix:")
    print(factors)
    print("Eigenvalues:")
    print(eigenvalues)

promax_rotation
~~~~~~~~~~~~~~~

Perform Promax Rotation on a factor loadings matrix to enhance the interpretability of factors in exploratory factor analysis by allowing for oblique solutions where factors can correlate.

Promax rotation is a type of oblique rotation that aims to simplify the factor structure after an initial orthogonal rotation, typically a varimax rotation. It enhances simple structure by allowing factors to correlate, adjusting the loadings matrix through a power transformation.

.. math::
    R = \left(L L^T\right)^{1/2} \cdot \left(L L^T\right)^{1/2 - 1} 
    \cdot \left(L L^T\right)^{1/2} \cdot P

where:
- :math:`R` is the rotated loadings matrix.
- :math:`L` is the original factor loadings matrix.
- :math:`P` is a transformation matrix derived from the power parameter, where diagonal elements are raised to the specified power.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import promax_rotation
    loadings = np.array([
        [0.7, 0.1, 0.3],
        [0.8, 0.2, 0.4],
        [0.4, 0.6, 0.5],
        [0.3, 0.7, 0.2]
    ])
    rotated_loadings = promax_rotation(loadings, power=4)
    print("Rotated Factor Loadings Matrix:")
    print(rotated_loadings)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import promax_rotation
    loadings = np.random.rand(20, 5)
    rotated_loadings = promax_rotation(loadings, power=4)
    print("Rotated Factor Loadings Matrix:")
    print(rotated_loadings)

ledoit_wolf_score
~~~~~~~~~~~~~~~~~

Calculate the model score using the Ledoit-Wolf shrinkage estimator for the covariance matrix of the given dataset X.

The Ledoit-Wolf shrinkage estimator is a method to improve the conditioning of covariance matrices in high-dimensional settings, balancing between the sample covariance matrix and a structured estimator (like the identity matrix) according to a calculated shrinkage coefficient. The regularized covariance is:

.. math::

    (1 - \text{shrinkage}) * \text{cov} + \text{shrinkage} * \mu * \text{np.identity(n_features)}

where :math:`\mu = \text{trace(cov)} / n_{features}` and shrinkage is given by the Ledoit and Wolf formula.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_spd_matrix
    from gofast.analysis.factors import ledoit_wolf_score
    X = make_spd_matrix(100, random_state=42)  
    score = ledoit_wolf_score(X)
    print(f"Model score: {score:.4f}")

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from sklearn.datasets import make_spd_matrix
    from gofast.analysis.factors import ledoit_wolf_score
    X = make_spd_matrix(1000, random_state=42)  
    score = ledoit_wolf_score(X, block_size=500)
    print(f"Model score: {score:.4f}")

evaluate_noise_impact_on_reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate and compare PCA and Factor Analysis (FA) scores under different noise conditions. This function evaluates the dimensionality using cross-validation scores and model likelihood comparisons with different covariance estimators.

The function simulates low-rank data, adds noise, and then assesses the performance of dimensionality reduction techniques. It particularly evaluates how well each method recovers the known rank of the data in the presence of homoscedastic (uniform across features) and heteroscedastic (varied across features) noise.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import evaluate_noise_impact_on_reduction
    X = np.random.rand(100, 50)
    results = evaluate_noise_impact_on_reduction(
        X, rank=10, sigma=0.5, step=5, random_state=42,
        verbose=1, display_plots=False
    )
    print(results['homoscedastic_noise']['pca_scores'])
    print(results['heteroscedastic_noise']['fa_scores'])

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import evaluate_noise_impact_on_reduction
    X = np.random.rand(200, 60)
    evaluate_noise_impact_on_reduction(
        X, rank=15, sigma=1.0, step=10, random_state=42
    )

make_scedastic_data
~~~~~~~~~~~~~~~~~~~

Generate a scedastic data dataset for probabilistic PCA and Factor Analysis for model comparison.

Probabilistic PCA and Factor Analysis are probabilistic models. The consequence is that the likelihood of new data can be used for model selection and covariance estimation. Here we compare PCA and FA with cross-validation on low-rank data corrupted with homoscedastic noise (noise variance is the same for each feature) or heteroscedastic noise (noise variance is different for each feature). In a second step, we compare the model likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed in recovering the size of the low-rank subspace. The likelihood with PCA is higher than FA in this case. However, PCA fails and overestimates the rank when heteroscedastic noise is present. Under appropriate circumstances, the low-rank models are more likely than shrinkage models.

Examples
--------

Example 1:
.. code-block:: python

    from gofast.analysis.factors import make_scedastic_data
    X, X_homo, X_hetero, n_components = make_scedastic_data()
    print(X.shape, X_homo.shape, X_hetero.shape)
    print(n_components)

Example 2 (more robust usage):
.. code-block:: python

    from gofast.analysis.factors import make_scedastic_data
    X, X_homo, X_hetero, n_components = make_scedastic_data(
        n_samples=2000, n_features=100, rank=20, sigma=0.7, 
        n_components=10, random_state=42
    )
    print(X.shape, X_homo.shape, X_hetero.shape)
    print(n_components)
    
    
rotated_factor
~~~~~~~~~~~~~~

Perform a simple rotated factor analysis on the dataset using an initial factor extraction method (e.g., PCA) followed by a rotation such as Varimax or Oblimin.

The initial factor extraction is performed using PCA, after which the specified rotation is applied to the factor loading matrix. This method allows for enhanced interpretability of the factors through rotations which aim to simplify the structure of the loadings.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import rotated_factor
    X = np.random.rand(100, 5)  # Simulated dataset with 100 samples and 5 features
    rotated_components = rotated_factor(X, n_components=2, rotation='varimax')
    print(rotated_components.shape)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import rotated_factor
    X = np.random.rand(200, 10)  # Simulated dataset with 200 samples and 10 features
    rotated_components = rotated_factor(X, n_components=3, rotation='oblimin', gamma=0.5)
    print(rotated_components.shape)

principal_axis_factoring
~~~~~~~~~~~~~~~~~~~~~~~~

Perform Principal Axis Factoring (PAF) on a given dataset.

Principal Axis Factoring is a technique used in factor analysis where the factors are estimated based on the correlations between variables. The objective of PAF is to identify latent variables that can explain the observed correlations among the variables.

.. math::
    \text{Maximize: } \sum_i \left( \frac{1}{n} \sum_j r_{ij}^2 \right) - 
    \left( \frac{1}{n^2} \sum_i \sum_j r_{ij}^2 \right)

where :math:`r_{ij}` is the correlation between the :math:`i`-th and :math:`j`-th variables, and :math:`n` is the number of variables.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import principal_axis_factoring
    X = np.random.rand(100, 5)  # Simulated dataset with 100 samples and 5 features
    factors = principal_axis_factoring(X, n_factors=2)
    print(factors.shape)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import principal_axis_factoring
    X = np.random.rand(200, 8)  # Simulated dataset with 200 samples and 8 features
    factors = principal_axis_factoring(X, n_factors=3, backend='scipy')
    print(factors.shape)

varimax_rotation
~~~~~~~~~~~~~~~~

Perform Varimax (orthogonal) rotation on the factor loading matrix.

Varimax rotation is an orthogonal rotation of the factor axes that maximizes the sum of the variances of the squared loadings. It simplifies the interpretation by making high loadings higher and low loadings lower within each factor.

.. math::
    \sum_i \left( \frac{1}{n} \sum_j a_{ij}^2 \right)^2 - \left( \frac{1}{n^2} 
    \sum_i \sum_j a_{ij}^2 \right)^2

where :math:`a_{ij}` is the loading of the :math:`j`-th variable on the :math:`i`-th factor, and :math:`n` is the number of variables.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import varimax_rotation
    # Simulated factor loading matrix (for illustration purposes)
    factor_loading_matrix = np.array([
        [0.7, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.3, 0.7]
    ])
    rotated_matrix = varimax_rotation(factor_loading_matrix)
    # Display the rotated factor loading matrix
    print("Rotated Factor Loading Matrix:")
    print(rotated_matrix)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import varimax_rotation
    # Simulated factor loading matrix (for illustration purposes)
    factor_loading_matrix = np.random.rand(10, 3)
    rotated_matrix = varimax_rotation(factor_loading_matrix, gamma=1.0, q=50, tol=1e-8)
    # Display the rotated factor loading matrix
    print("Rotated Factor Loading Matrix:")
    print(rotated_matrix)

oblimin_rotation
~~~~~~~~~~~~~~~~

Perform Oblimin (oblique) rotation on the factor loading matrix to allow the factors to be correlated. This rotation method optimizes a criterion that balances the simplicity of factors with their correlation.

The mathematical formulation of the Oblimin rotation objective is:

.. math::
    \sum_i \left( \sum_j (a_{ij}^2 - \gamma a_{ij}^4) \right)^2 \rightarrow \text{Maximize}

Where :math:`a_{ij}` is the loading of the j-th variable on the i-th factor, and :math:`\gamma` is a parameter that controls the degree of correlation among the factors.

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import oblimin_rotation
    # Simulated factor loading matrix (for illustration purposes)
    factor_loading_matrix = np.array([
        [0.7, 0.1, 0.3],
        [0.8, 0.2, 0.4],
        [0.4, 0.6, 0.2],
        [0.3, 0.7, 0.5]
    ])
    # Apply Oblimin Rotation with a specified gamma value (degree of correlation)
    gamma_value = 0.2
    rotated_matrix = oblimin_rotation(factor_loading_matrix, gamma=gamma_value)
    # Display the rotated factor loading matrix
    print("Original Factor Loading Matrix:")
    print(factor_loading_matrix)
    print("\nRotated Factor Loading Matrix (Oblimin Rotation):")
    print(rotated_matrix)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import oblimin_rotation
    # Simulated factor loading matrix (for illustration purposes)
    factor_loading_matrix = np.random.rand(10, 3)
    rotated_matrix = oblimin_rotation(factor_loading_matrix, gamma=0.5, max_iter=50, tol=1e-8)
    # Display the rotated factor loading matrix
    print("Rotated Factor Loading Matrix (Oblimin Rotation):")
    print(rotated_matrix)

evaluate_dimension_reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate dimensionality reduction performance using PCA (Principal Component Analysis) and FA (Factor Analysis) for a dataset over a specified range or number of components.

`evaluate_dimension_reduction` applies cross-validation to assess the effectiveness of each method.

Examples
--------

Example 1:
.. code-block:: python

    from sklearn.datasets import load_iris
    from gofast.analysis.factors import evaluate_dimension_reduction
    X, _ = load_iris(return_X_y=True)
    pca_scores, fa_scores = evaluate_dimension_reduction(
        X, n_features=X.shape[1], n_components=3)
    print(pca_scores, fa_scores)

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import evaluate_dimension_reduction
    X = np.random.rand(200, 10)
    pca_scores, fa_scores = evaluate_dimension_reduction(
        X, n_features=X.shape[1], n_components=[2, 4, 6, 8], cv=10)
    print(pca_scores, fa_scores)

samples_hotellings_t_square
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform Two-Sample Hotelling's T-Square test to compare two multivariate samples.

Hotelling's T-Square test assesses whether two multivariate samples come from populations with equal means. It is a multivariate extension of the two-sample t-test.

.. math::
    T^2 = n_1 \cdot n_2 \cdot (\mathbf{\bar{x}}_1 - \mathbf{\bar{x}}_2)^T 
    \cdot \mathbf{S}_w^{-1} \cdot (\mathbf{\bar{x}}_1 - \mathbf{\bar{x}}_2)

.. math::
    \text{Degrees of Freedom (Numerator):} \quad df_1 = p

.. math::
    \text{Degrees of Freedom (Denominator):} \quad df_2 = n_1 + n_2 - p - 1

.. math::
    \text{P-Value:} \quad p = 1 - F\left(T^2 \cdot \frac{df_2}
    {df_1 \cdot df_2 - p + 1}, df_1, df_2\right)

Examples
--------

Example 1:
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import samples_hotellings_t_square
    sample1 = np.array([[1.2, 2.3], [1.8, 3.5], [2.6, 4.0]])
    sample2 = np.array([[0.9, 2.0], [1.5, 3.2], [2.3, 3.8]])
    statistic, df1, df2, p_value = samples_hotellings_t_square(sample1, sample2)
    print(f"Hotelling's T-Square statistic: {statistic:.4f}")
    print(f"Degrees of freedom (numerator): {df1}")
    print(f"Degrees of freedom (denominator): {df2}")
    print(f"P-value: {p_value:.4f}")

Example 2 (more robust usage):
.. code-block:: python

    import numpy as np
    from gofast.analysis.factors import samples_hotellings_t_square
    sample1 = np.random.rand(50, 5)
    sample2 = np.random.rand(60, 5)
    statistic, df1, df2, p_value = samples_hotellings_t_square(sample1, sample2)
    print(f"Hotelling's T-Square statistic: {statistic:.4f}")
    print(f"Degrees of freedom (numerator): {df1}")
    print(f"Degrees of freedom (denominator): {df2}")
    print(f"P-value: {p_value:.4f}")



