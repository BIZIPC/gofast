# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Factor Analysis (FA)
====================

Here, we give an overview of implementations with vizualization of some factor 
analyses. For some remaining factor analysis methods, including Maximum 
Likelihood Factor Analysis, Direct Oblimin Rotation, Varimax Rotation, and 
others, it's recommended to refer to specialized texts in statistics or use 
established statistical software like R, Python's statsmodels, or scikit-learn 
(for methods it supports).

Implementing these methods requires deep knowledge in statistics and numerical 
methods, and often these implementations are already optimized in existing 
software libraries. For learning and research purposes, you might explore 
academic textbooks or online courses that cover advanced statistical methods 
and factor analysis in detail.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import f as statsf 

from sklearn.decomposition import PCA, FactorAnalysis 
from sklearn.covariance import LedoitWolf 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..backends.selector import BackendSelector 
from ..api.docstring import _core_docs
from ..api.types import ArrayLike , Union, Iterable 
from ..api.util import to_snake_case 
from ..tools.validator import check_array, validate_positive_integer
from ..tools.validator import ensure_2d, parameter_validator 

__all__=[ 
    "ledoit_wolf_score",  
    "evaluate_noise_impact_on_reduction", 
    "make_scedastic_data", 
    "rotated_factor", 
    "principal_axis_factoring", 
    "varimax_rotation", 
    "oblimin_rotation", 
    "evaluate_dimension_reduction", 
    "samples_hotellings_t_square", 
    "promax_rotation", 
    "spectral_factor", 
    ]

def spectral_factor(
    data, /,  
    num_factors=None, 
    backend: str="numpy", 
    ):
    """
    Perform Spectral Method in Factor Analysis on the given data.
    
    The `spectral_factor` analysis function performs the Spectral Method in 
    Factor Analysis on the input data, allowing you to obtain factor 
    loadings, common factors, and eigenvalues. 

    Perform Spectral Factor Analysis on the given data to extract factor
    loadings, common factors, and associated eigenvalues using eigenvalue
    decomposition of the covariance matrix of the data.

    Parameters
    ----------
    data : ndarray
        The input data matrix with shape (n_samples, n_features), where 
        n_samples is the number of observations and n_features is the 
        number of variables.
    num_factors : int, optional
        The number of factors to extract. If None, the number of factors
        is determined as the minimum of n_samples and n_features.
    backend : str, optional
        The computational backend to use for eigenvalue decomposition.
        Currently only 'numpy' is supported. Default is "numpy".

    Returns
    -------
    loadings : ndarray
        The factor loadings matrix with shape (n_features, num_factors).
        These are the eigenvectors scaled by the square root of the 
        corresponding eigenvalues.
    factors : ndarray
        The common factors matrix with shape (n_samples, num_factors).
        These represent the projection of the data onto the factor loadings.
    eigenvalues : ndarray
        The eigenvalues associated with the extracted factors, indicating
        the variance explained by each factor.

    Notes
    -----
    Spectral Factor Analysis involves the eigenvalue decomposition of the
    covariance matrix of the data, defined as:
    
    .. math::
        C = \frac{1}{n_{\text{samples}}} X^T X

    where :math:`X` is the data matrix centered by subtracting the mean
    of each feature. The decomposition is:

    .. math::
        C = VDV^T

    where :math:`V` is the matrix of eigenvectors (factor loadings) and
    :math:`D` is the diagonal matrix of eigenvalues. The factor loadings are
    obtained by:

    .. math::
        L = V \sqrt{D}

    and the common factors are calculated as:

    .. math::
        F = X L

    where :math:`L` is the matrix of loadings, and :math:`F` are the factors.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import spectral_factor
    >>> data = np.array([[1.2, 2.3, 3.1],
    ...                  [1.8, 3.5, 4.2],
    ...                  [2.6, 4.0, 5.3],
    ...                  [0.9, 1.7, 2.0]])
    >>> loadings, factors, eigenvalues = spectral_factor(data, num_factors=2)
    >>> print("Factor Loadings Matrix:")
    >>> print(loadings)
    >>> print("Common Factors Matrix:")
    >>> print(factors)
    >>> print("Eigenvalues:")
    >>> print(eigenvalues)
    Factor Loadings Matrix:
    [[-0.73841609  0.11161432]
     [-1.05205481 -0.11480111]
     [-1.41795179  0.02705258]]
    Common Factors Matrix:
    [[ 1.69863184e+00  3.69563550e-03]
     [-1.56663056e+00 -3.73392712e-02]
     [-4.24313782e+00  2.43094659e-02]
     [ 4.11113653e+00  9.33416979e-03]]
    Eigenvalues:
    [3.66266495 0.02636889]
    
    """
    data = check_array(data )
    # Calculate the covariance matrix of the data
    cov_matrix = np.cov(data, rowvar=False)

    # Perform eigenvalue decomposition on the covariance matrix
    # based on the selected backend
    backend_selector = BackendSelector(preferred_backend=backend)
    backend = backend_selector.get_backend()
    if backend.__class__.__name__ in ['NumpyBackend', 'numpy']:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    else:
        eigenvalues, eigenvectors = backend.eig(cov_matrix)
        
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Determine the number of factors
    if num_factors is None:
        num_factors = min(data.shape[0], data.shape[1])
    else:
        num_factors = min(num_factors, data.shape[1])

    # Extract the appropriate number of eigenvectors and scale for loadings
    selected_eigenvectors = eigenvectors[:, :num_factors]
    sqrt_eigenvalues = np.sqrt(eigenvalues[:num_factors])
    loadings = selected_eigenvectors * sqrt_eigenvalues[np.newaxis, :]
    
    # Calculate the factor scores
    factors = np.dot(data - np.mean(data, axis=0), loadings)

    return loadings, factors, eigenvalues[:num_factors]

def promax_rotation(loadings: ArrayLike, power: int=4):
    """
    Perform Promax Rotation on a factor loadings matrix to enhance the
    interpretability of factors in exploratory factor analysis by allowing
    for oblique solutions where factors can correlate.

    Promax rotation is a type of oblique rotation that aims to simplify the
    factor structure after an initial orthogonal rotation, typically a varimax
    rotation. It enhances simple structure by allowing factors to correlate, 
    adjusting the loadings matrix through a power transformation.

    Parameters
    ----------
    loading_ar2d : ndarray
        The factor loadings matrix with shape (n_features, num_factors).
        This matrix represents initial factor loadings typically obtained 
        from a PCA or an initial factor analysis.
    power : int, optional
        The power parameter for Promax rotation, which enhances the contrast 
        between large and small loadings. A common choice is 4. Default is 4.

    Returns
    -------
    rotated_loadings : ndarray
        The rotated factor loadings matrix after applying Promax rotation.

    Notes
    -----
    Promax rotation involves raising the loadings to a specified power 
    to increase the differentiation between higher and lower loadings, thus
    enhancing interpretability. The mathematical formulation for Promax 
    rotation can be expressed as:

    .. math::
        R = \left(L L^T\right)^{1/2} \cdot \left(L L^T\right)^{1/2 - 1} 
        \cdot \left(L L^T\right)^{1/2} \cdot P

    where:
    - :math:`R` is the rotated loadings matrix.
    - :math:`L` is the original factor loadings matrix.
    - :math:`P` is a transformation matrix derived from the power parameter,
      where diagonal elements are raised to the specified power.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import promax_rotation
    >>> loadings = np.array([
    ...     [0.7, 0.1, 0.3],
    ...     [0.8, 0.2, 0.4],
    ...     [0.4, 0.6, 0.5],
    ...     [0.3, 0.7, 0.2]
    ... ])
    >>> rotated_loadings = promax_rotation(loadings, power=4)
    >>> print("Rotated Factor Loadings Matrix:")
    >>> print(rotated_loadings)
    Rotated Factor Loadings Matrix:
    [[ 0.7+1.04773790e-10j  1.6-2.04890966e-09j 24.3-9.42964107e-09j]
     [ 0.8+1.62981451e-09j  3.2+2.90572643e-08j 32.4+1.28243119e-07j]
     [ 0.4+3.49245965e-11j  9.6+1.30385160e-09j 40.5+1.88592821e-09j]
     [ 0.3-5.19794412e-09j 11.2-1.12690032e-08j 16.2-1.80106144e-07j]]
    """
    power = validate_positive_integer(power, "power", include_zero= True)

    loadings = check_array(loadings)
    # Perform Promax rotation using the specified power parameter
    L = np.asarray(loadings)
    num_factors = L.shape[1]

    # Compute L * L' (transpose)
    LLT = np.dot(L, L.T)

    # Compute the square root of LLT and its inverse
    LLT_sqrt = linalg.sqrtm(LLT)
    LLT_inv_sqrt = linalg.inv(LLT_sqrt)

    # P matrix should be the identity matrix raised to the given power
    # and must be aligned in dimension with the number of factors.
    P = np.diag(np.power(np.arange(1, num_factors + 1), power))

    # Compute the transformed loadings
    # L * P involves the original loadings and the transformation matrix
    transformed_loadings = np.dot(L, P)

    # Rotate using the formula: sqrtm(LLT) * inv(sqrtm(LLT)) * transformed_loadings
    rotated_loadings = np.dot(np.dot(LLT_sqrt, LLT_inv_sqrt), transformed_loadings)

    return rotated_loadings

def ledoit_wolf_score(
    X: ArrayLike, store_precision: bool = True, 
    assume_centered: bool = False,
    block_size: int = 1000, **kwargs
    ):
    """
    Calculate the model score using the Ledoit-Wolf shrinkage estimator for 
    the covariance matrix of the given dataset X.

    Parameters
    ----------
    X : ArrayLike
        Input data where rows represent samples and columns represent features.
    store_precision : bool, default=True
        Specify if the estimated precision matrix should be stored.
    assume_centered : bool, default=False
        If True, it is assumed that the data are centered. If False, the data
        will be centered before computing the covariance estimate.
    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split during 
        the Ledoit-Wolf estimation. This is a memory optimization and does not 
        affect the result.
    **kwargs : dict
        Additional keyword arguments to be passed to the LedoitWolf estimator.

    Returns
    -------
    float
        The mean cross-validated score of the Ledoit-Wolf estimator on the data.

    Notes
    -----
    The Ledoit-Wolf shrinkage estimator is a method to improve the conditioning of 
    covariance matrices in high-dimensional settings, balancing between the sample 
    covariance matrix and a structured estimator (like the identity matrix) according 
    to a calculated shrinkage coefficient. The regularised covariance is:
        
    .. math::
        
        (1 - text{shrinkage}) * \text{cov} + \text{shrinkage} * \mu * \text{np.identity(n_features)}
        
    where :math:`\mu = \text{trace(cov)} / n_{features}`
    and shrinkage is given by the Ledoit and Wolf formula
     
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_spd_matrix
    >>> from gofast.analysis.factors import ledoit_wolf_score
    >>> # Generate a random symmetric positive-definite matrix
    >>> X = make_spd_matrix(100, random_state=42)  
    >>> score = ledoit_wolf_score(X)
    >>> print(f"Model score: {score:.4f}")
    Model score: 72.1523
    """
    # Initialize the LedoitWolf covariance estimator with given parameters
    lw_estimator = LedoitWolf(
        store_precision=store_precision, assume_centered=assume_centered,
        block_size=block_size, **kwargs
    )
    # Perform the fitting and scoring
    # Note: cross_val_score requires a model, so we need to adapt the covariance
    # estimator to fit this. Typically, cross_val_score is used with predictive
    # models, so we would wrap this in a way that can be used with 
    # cross_val_score or simply use .fit() and .score() methods if adapting 
    # isn't trivial.
    X = ensure_2d (check_array(X))
    lw_estimator.fit(X)  # Fit the Ledoit-Wolf model to data
    score = np.mean(cross_val_score(lw_estimator, X))  # Compute cross-validated scores

    return score

def evaluate_noise_impact_on_reduction(
    X: ArrayLike,
    rank: int = 10,
    sigma: float = 1.0,
    step: int = 5,
    random_state: int = 42,
    verbose: int = 0,
    display_plots: bool = True
):
    from ..models.utils import shrink_covariance_cv_score 
    
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    n_components = np.arange(0, n_features, step)
    
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X_base = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    X_homo = X_base + sigma * rng.randn(n_samples, n_features)
    X_hetero = X_base + rng.randn(n_samples, n_features) * (
        sigma * rng.rand(n_features) + sigma / 2.)

    results = {}

    for X_noise, title in [(X_homo, 'Homoscedastic Noise'), (
            X_hetero, 'Heteroscedastic Noise')]:
        pca_scores, fa_scores = evaluate_dimension_reduction(X_noise, n_features)
        pca_mle = PCA(svd_solver='full', n_components='mle').fit(X_noise)
        n_components_pca_mle = pca_mle.n_components_
 
        if verbose:
            print(f"Best n_components by PCA CV = {np.argmax(pca_scores)}")
            print(f"Best n_components by Factor Analysis CV = {np.argmax(fa_scores)}")
            print(f"Best n_components by PCA MLE = {n_components_pca_mle}")

        if display_plots:
            plt.figure()
            plt.plot(n_components, pca_scores, 'b', label='PCA scores')
            plt.plot(n_components, fa_scores, 'r', label='FA scores')
            plt.axvline(rank, color='g', label=f'TRUTH: {rank}', linestyle='-')
            plt.axvline(np.argmax(pca_scores), color='b', 
                        label=f'PCA CV: {np.argmax(pca_scores)}', linestyle='--')
            plt.axvline(np.argmax(fa_scores), color='r',
                        label=f'Factor Analysis CV: {np.argmax(fa_scores)}', linestyle='--')
            plt.axvline(n_components_pca_mle, color='k', 
                        label=f'PCA MLE: {n_components_pca_mle}', linestyle='--')
            plt.axhline(shrink_covariance_cv_score(X_noise),
                        color='violet', label='Shrunk Covariance MLE', linestyle='-.')
            plt.axhline(ledoit_wolf_score(X_noise), color='orange', 
                        label='Ledoit Wolf MLE', linestyle='-.')
            plt.xlabel('Number of Components')
            plt.ylabel('CV Scores')
            plt.legend(loc='lower right')
            plt.title(title)
            plt.show()

        # Collecting results for return when not displaying plots
        results[to_snake_case(title)] = {
            'pca_scores': pca_scores,
            'fa_scores': fa_scores,
            'n_components_pca_mle': int(n_components_pca_mle),
            'X_homo': X_homo,
            'X_hetero': X_hetero
        }

    if not display_plots:
        return results
    
evaluate_noise_impact_on_reduction.__doc__="""\
Generate and compare PCA and Factor Analysis (FA) scores under
different noise conditions. This function evaluates the dimensionality
using cross-validation scores and model likelihood comparisons with
different covariance estimators.

The function simulates low-rank data, adds noise, and then assesses the
performance of dimensionality reduction techniques. It particularly
evaluates how well each method recovers the known rank of the data
in the presence of homoscedastic (uniform across features) and
heteroscedastic (varied across features) noise.

Parameters
----------
{params.X}

rank : int, optional
    Rank of the data to simulate the underlying low-dimensional structure.
    Default is 10.
sigma : float, optional
    Standard deviation of the added noise, which affects the difficulty
    of the dimensionality reduction problem. Default is 1.0.
step : int, optional
    Step size for generating the range of component numbers to evaluate.
    Default is 5.
random_state : int, optional
    Seed for the random number generator to ensure reproducibility.
    Default is 42.
    
{params.verbose}
display_plots : bool, optional
    Whether to display plots of the results. If True, plots are shown
    and the function does not return any values. If False, returns
    detailed results in a dictionary. Default is True.

Returns
-------
dict, optional
    A dictionary containing PCA and FA scores, the number of components
    determined to be optimal by PCA MLE, and the datasets generated with
    homoscedastic and heteroscedastic noise. This return is conditional
    on `display_plots` being False.

Notes
-----
- This function is useful in scenarios where understanding the impact
  of noise on dimensionality reduction algorithms is critical, such as
  in feature extraction phases of machine learning pipelines.
- The homoscedastic noise scenario tests the robustness of PCA and FA
  against uniform noise, while the heteroscedastic scenario presents a
  more challenging environment where noise levels vary per feature,
  potentially impacting the accuracy of rank recovery.

Examples
--------
>>> import numpy as np
>>> from gofast.analysis.factors import evaluate_noise_impact_on_reduction
>>> X = np.random.rand(100, 50)
>>> results = evaluate_noise_impact_on_reduction(
...     X, rank=10, sigma=0.5, step=5, random_state=42,
...     verbose=1, display_plots=False
... )
>>> print(results['homoscedastic_noise']['pca_score'])
>>> print(results['heteroscedastic_noise']['fa_scores'])
Best n_components by PCA CV = 2
Best n_components by Factor Analysis CV = 2
Best n_components by PCA MLE = 10
Best n_components by PCA CV = 2
Best n_components by Factor Analysis CV = 2
Best n_components by PCA MLE = 11
[-53.05972870584175, -52.566332485632714, -50.52484170712419, ...]
[-52.35497867106518, -51.91787494749351, -50.63633551613926, ...]
>>> evaluate_noise_impact_on_reduction(
...     X, rank=10, sigma=0.5, step=5, random_state=42,
... )
""".format(
    params =_core_docs["params"]
)    
    
def make_scedastic_data (
    n_samples: int= 1000, 
    n_features: int=50, 
    rank: int =  10, 
    sigma: float=1., 
    n_components: Union[int, Iterable[int]] = None, 
    random_state =42
   ): 
    """ Generate a scedatic data data for probabilistic PCA and Factor Analysis for  
    model comparison. 
    
    
    Probabilistic PCA and Factor Analysis are probabilistic models. The consequence 
    is that the likelihood of new data can be used for model selection and 
    covariance estimation. Here we compare PCA and FA with cross-validation on 
    low rank data corrupted with homoscedastic noise 
    (noise variance is the same for each feature) or heteroscedastic noise 
    (noise variance is the different for each feature). In a second step we compare 
    the model likelihood to the likelihoods obtained from shrinkage covariance 
    estimators.
    
    One can observe that with homoscedastic noise both FA and PCA succeed in 
    recovering the size of the low rank subspace. The likelihood with PCA is 
    higher than FA in this case. However PCA fails and overestimates the rank 
    when heteroscedastic noise is present. Under appropriate circumstances the 
    low rank models are more likely than shrinkage models.
    
    The automatic estimation from Automatic Choice of Dimensionality for PCA. 
    NIPS 2000: 598-604 by Thomas P. Minka is also compared.
    
    # Authors: Alexandre Gramfort & Denis A. Engemann
    # License: BSD 3 clause
    
    edited by LKouadio on Tue Oct 11 16:54:26 2022
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate.
    n_features : int, default=50
        Number of features.
    rank : int, default=10
        Rank of the matrix to simulate low-rank structure.
    sigma : float, default=1.0
        Base level of noise variance.
    n_components : int or Iterable[int], optional
       The number of components to consider for dimensionality reduction.
       If None, defaults to a range of values from 0 to n_features, stepping by 5.
       If an int, a range from 0 to n_features stepping by the given int.
       If an iterable, used directly.
    random_state : int, default=42
        Controls the randomness of the data generation.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The generated samples.
    X_homo : ndarray, shape (n_samples, n_features)
        Samples with homoscedastic noise added.
    X_hetero : ndarray, shape (n_samples, n_features)
        Samples with heteroscedastic noise added.
    n_components : ndarray
        Array containing options for the number of components.

    Notes
    -----
    - Homoscedastic noise makes both PCA and FA likely to recover the true rank.
    - Heteroscedastic noise may lead PCA to overestimate the rank, while FA may 
      perform better.
    - The likelihood comparison between PCA, FA, and shrinkage models highlights
      the conditions under which low-rank models outperform others.

    Examples
    --------
    >>> from gofast.analysis.factors import make_scedastic_data
    >>> X, X_homo, X_hetero, n_components = make_scedastic_data()
    >>> print(X.shape, X_homo.shape, X_hetero.shape)
    >>> print(n_components)
    (1000, 50) (1000, 50) (1000, 50)
    [ 0  5 10 15 20 25 30 35 40 45]
    """
    # Validate parameters
    for value, name in zip([n_samples, n_features, rank],
                           ["n_samples", "n_features", "rank"]):
        validate_positive_integer(value, name)

    # Create the data
    rng = np.random.RandomState(random_state )
    U, _, _ = linalg.svd(rng.randn(n_features, n_features))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    
    # Adding homoscedastic noise
    X_homo = X + sigma * rng.randn(n_samples, n_features)
    
    # Adding heteroscedastic noise
    sigmas = sigma * rng.rand(n_features) + sigma / 2.
    X_hetero = X + rng.randn(n_samples, n_features) * sigmas
    
    # Handling n_components parameter
    if n_components is None:
         # Fit the models
        n_components = np.arange(0, n_features, 5)
    elif isinstance(n_components, int):
        n_components = np.arange(0, n_features, n_components)
    elif not all(isinstance(x, int) and x >= 0 for x in n_components):
        raise ValueError(
            "n_components must be a positive integer or an iterable"
            " of positive integers")
   
    return X, X_homo, X_hetero , n_components
 
def rotated_factor(
    X: ArrayLike, 
    n_components: int=2, 
    rotation:str='varimax',
    gamma: float=1.0,
    q: float=20, 
    max_iter: int=20, 
    tol: float=1e-6
    ):
    """
    Perform a simple rotated factor analysis on the dataset using an initial
    factor extraction method (e.g., PCA) followed by a rotation such as Varimax
    or Oblimin.

    Parameters
    ----------
    X : array_like
        Input data where rows represent samples and columns represent features.
    n_components : int, optional
        The number of factors to extract. Default is 2.
    rotation : str, optional
        The type of rotation to apply. Options include 'varimax', 'oblimin', etc.
        Default is 'varimax'.
    gamma : float, optional
        The rotation parameter controlling the degree of correlation between
        factors. Relevant for certain rotation types like 'oblimin'. Default is 1.0.
    q : int, optional
        The maximum number of iterations for convergence in the rotation algorithm.
        Default is 20.
    max_iter : int, optional
        The maximum number of iterations for the rotation algorithm. Default is 20.
    tol : float, optional
        Tolerance for the convergence of the rotation algorithm. Default is 1e-6.

    Returns
    -------
    rotated_components : ndarray
        The rotated factor components as an array.

    Notes
    -----
    The initial factor extraction is performed using PCA, after which the specified
    rotation is applied to the factor loading matrix. This method allows for enhanced
    interpretability of the factors through rotations which aim to simplify the
    structure of the loadings.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import rotated_factor
    >>> X = np.random.rand(100, 5)  # Simulated dataset with 100 samples and 5 features
    >>> rotated_components = rotated_factor(X, n_components=2, rotation='varimax')
    >>> print(rotated_components.shape)
    (100, 2)
    """
    # Ensure the input is a numpy array and scale it
    X = StandardScaler().fit_transform(X)

    # Initial factor extraction using PCA
    pca = PCA(n_components=n_components)
    loadings = pca.fit_transform(X)
    
    rotation = parameter_validator("rotation", target_strs = {
        "varimax", "oblimin", "none"}) (rotation)

    # Apply rotation based on the specified type
    if rotation == 'varimax':
        rotated_components = varimax_rotation(
            loadings, gamma=gamma, q=q, tol=tol)
    elif rotation == 'oblimin':
        rotated_components = oblimin_rotation(
            loadings, gamma=gamma, max_iter=max_iter, tol=tol)
    else:
        # Default to returning the PCA components if no valid rotation is specified
        rotated_components = loadings

    return rotated_components
   
def principal_axis_factoring(
    X: ArrayLike, 
    n_factors: int=2, 
    backend: str='numpy'
    ):
    r"""
    Perform Principal Axis Factoring (PAF) on a given dataset.

    Principal Axis Factoring is a technique used in factor analysis where the 
    factors are estimated based on the correlations between variables. 
    
    The objective of PAF is to identify latent variables that can explain the 
    observed correlations among the variables.

    Mathematically, the PAF objective can be represented as:

    .. math::
        \text{Maximize: } \sum_i \left( \frac{1}{n} \sum_j r_{ij}^2 \right) - 
        \left( \frac{1}{n^2} \sum_i \sum_j r_{ij}^2 \right)

    where :math:`r_{ij}` is the correlation between the :math:`i`-th and :math:`j`-th 
    variables, and :math:`n` is the number of variables.

    Parameters
    ----------
    X : array_like
        Input data where rows represent samples and columns represent features.
    n_factors : int, optional
        The number of factors to extract. Defaults to 2.
        
    backend : str, optional
        The computational backend to use. Defaults to 'numpy'. Other options are 
        'scipy', 'cupy', etc., depending on what's available.   
    Returns
    -------
    factors : ndarray
        The extracted factors, represented as an array with dimensions (n_features, n_factors),
        where each column represents a factor.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import principal_axis_factoring
    >>> X = np.random.rand(100, 5)  # Simulated dataset with 100 samples and 5 features
    >>> factors = principal_axis_factoring(X, n_factors=2)
    >>> print(factors.shape)
    (5, 2)
    """
    # Ensure that X is a numpy array and validate the number of factors
    X = check_array(X) 
    n_factors = validate_positive_integer(n_factors, "n_factors")
    # Compute the correlation matrix, not covariance, as PAF uses correlations
    correlation_matrix = np.corrcoef(X, rowvar=False)

    # Perform eigenvalue decomposition on the correlation matrix
    # based on the selected backend
    backend_selector = BackendSelector(preferred_backend=backend)
    backend = backend_selector.get_backend()
    if backend.__class__.__name__ in ['NumpyBackend', 'numpy']:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    else:
        eigenvalues, eigenvectors = backend.eig(correlation_matrix)
        
    # Select the top n_factors eigenvectors (principal components)
    idx = np.argsort(eigenvalues)[::-1]
    factors = eigenvectors[:, idx[:n_factors]]

    return factors

def varimax_rotation(
    loadings: ArrayLike, /,  
    gamma: float=1.0, 
    q: int=20, 
    tol: float=1e-6
    ):
    r"""
    Perform Varimax (orthogonal) rotation on the factor loading matrix.

    Varimax rotation is an orthogonal rotation of the factor axes that maximizes 
    the sum of the variances of the squared loadings. It simplifies the interpretation 
    by making high loadings higher and low loadings lower within each factor.

    The Varimax Rotation Objective is to maximize:
    
    .. math::
        \sum_i \left( \frac{1}{n} \sum_j a_{ij}^2 \right)^2 - \left( \frac{1}{n^2} 
        \sum_i \sum_j a_{ij}^2 \right)^2

    where :math:`a_{ij}` is the loading of the :math:`j`-th variable on the 
    :math:`i`-th factor, and :math:`n` is the number of variables.
    
    Parameters
    ----------
    loadings : array_like
        The factor loading matrix obtained from factor analysis where
        rows represent variables and columns represent factors.
    gamma : float, optional
        The normalization parameter for the rotation. Defaults to 1.0, 
        representing Varimax orthogonal rotation.
    q : int, optional
        The maximum number of iterations allowed for convergence. Defaults to 20.
    tol : float, optional
        The tolerance for convergence; rotation stops if changes are below this threshold.
        Defaults to 1e-6.

    Returns
    -------
    rotated_matrix : array_like
        The rotated factor loading matrix, enhancing factor interpretability by
        maximizing loadings variance within factors.

    Notes
    -----
    Varimax rotation is particularly useful in factor analysis and other dimensionality 
    reduction techniques where interpretability of the factors is crucial. This rotation 
    tends to produce factors that are easier to interpret due to the clearer partitioning 
    of high and low loadings, making it one of the most popular rotation methods in 
    exploratory factor analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import varimax_rotation
    >>> # Simulated factor loading matrix (for illustration purposes)
    >>> factor_loading_matrix = np.array([
    ...     [0.7, 0.1],
    ...     [0.8, 0.2],
    ...     [0.4, 0.6],
    ...     [0.3, 0.7]
    ... ])
    >>> rotated_matrix = varimax_rotation(factor_loading_matrix)
    >>> # Display the rotated factor loading matrix
    >>> print("Rotated Factor Loading Matrix:")
    >>> print(rotated_matrix)
    Rotated Factor Loading Matrix:
    [[ 0.70710583 -0.00116139]
     [ 0.82038213  0.08350549]
     [ 0.48171462  0.53661068]
     [ 0.39704774  0.64988698]]
    """
    loading_matrix = np.asarray(loadings)
    n_rows, n_cols = loading_matrix.shape
    rotation_matrix = np.eye(n_cols)
    var = 0
    q = validate_positive_integer(q, "q")
    for _ in range(q):
        factor_loading_rotated = np.dot(loading_matrix, rotation_matrix)
        lambda_squared = (factor_loading_rotated ** 2).sum(axis=0)

        # Calculate the term for matrix multiplication
        tmp = factor_loading_rotated ** 3
        for j in range(n_cols):
            tmp[:, j] *= lambda_squared[j]

        # Adjusted to compute the off-diagonal part of the numerator matrix
        A = np.dot(factor_loading_rotated.T, tmp) / n_rows
        B = np.diag(lambda_squared) / n_rows
        T = A - B @ B

        # SVD on the target matrix T, used in rotation calculation
        u, s, v = np.linalg.svd(T)
        rotation_matrix = np.dot(u, v)

        var_new = np.sum(s)
        if np.abs(var_new - var) < tol:
            break
        var = var_new

    return np.dot(loading_matrix, rotation_matrix)

def oblimin_rotation(
    loadings, /, 
    gamma=0.0, 
    max_iter=100, 
    tol=1e-6
    ):
    """
    Perform Oblimin (oblique) rotation on the factor loading matrix
    to allow the factors to be correlated. This rotation method optimizes
    a criterion that balances the simplicity of factors with their correlation.

    The mathematical formulation of the Oblimin rotation objective is:

    .. math::
        \sum_i \left( \sum_j (a_{ij}^2 - \gamma a_{ij}^4) \right)^2 \rightarrow \text{Maximize}

    Where :math:`a_{ij}` is the loading of the j-th variable on the i-th factor, 
    and :math:`\gamma` is a parameter that controls the degree of correlation 
    among the factors.

    Parameters
    ----------
    loadings : array_like
        The factor loading matrix obtained from factor analysis where
        rows represent variables and columns represent factors.
    gamma : float, optional
        The rotation parameter controlling the degree of correlation between 
        factors. Defaults to 0.0, indicating orthogonal rotation.
    max_iter : int, optional
        The maximum number of iterations allowed for convergence. Defaults to 100.
    tol : float, optional
        The tolerance for convergence. If the change in the factor loading 
        matrix is less than this value, the rotation process is stopped. 
        Defaults to 1e-6.

    Returns
    -------
    rotated_matrix : array_like
        The rotated factor loading matrix that has potentially improved
        interpretability due to oblique rotation allowing correlated factors.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis.factors import oblimin_rotation
    >>> # Simulated factor loading matrix (for illustration purposes)
    >>> factor_loading_matrix = np.array([
    ...     [0.7, 0.1, 0.3],
    ...     [0.8, 0.2, 0.4],
    ...     [0.4, 0.6, 0.2],
    ...     [0.3, 0.7, 0.5]
    ... ])
    >>> # Apply Oblimin Rotation with a specified gamma value (degree of correlation)
    >>> gamma_value = 0.2
    >>> rotated_matrix = oblimin_rotation(factor_loading_matrix, gamma=gamma_value)
    >>> # Display the rotated factor loading matrix
    >>> print("Original Factor Loading Matrix:")
    >>> print(factor_loading_matrix)
    >>> print("\nRotated Factor Loading Matrix (Oblimin Rotation):")
    >>> print(rotated_matrix)
    Original Factor Loading Matrix:
    [[0.7 0.1 0.3]
     [0.8 0.2 0.4]
     [0.4 0.6 0.2]
     [0.3 0.7 0.5]]

    Rotated Factor Loading Matrix (Oblimin Rotation):
    [[ 0.43321922 -0.06261186  0.06028775]
     [ 0.4599841  -0.02318638  0.13215792]
     [ 0.19079819  0.47188445 -0.2536284 ]
     [-0.05047662  0.38426712  0.37307077]]
    """

    factor_loading_matrix = check_array(loadings)
    n_vars, n_factors = factor_loading_matrix.shape
    rotated_matrix = factor_loading_matrix.copy()
    converged = False

    for _ in range(max_iter):
        prev_matrix = rotated_matrix.copy()

        for i in range(n_factors):
            for j in range(n_factors):
                if i != j:
                    cov_ij = np.sum(rotated_matrix[:, i] * rotated_matrix[:, j])
                    cov_ii = np.sum(rotated_matrix[:, i] ** 2)
                    cov_jj = np.sum(rotated_matrix[:, j] ** 2)

                    delta = (cov_ij - gamma * cov_ii * cov_jj) / cov_ii

                    rotated_matrix[:, i] = rotated_matrix[:, i] - delta * rotated_matrix[:, j]

        # Check for convergence
        if np.allclose(prev_matrix, rotated_matrix, atol=tol):
            converged = True
            break

    if not converged:
        print("Warning: Oblimin rotation did not converge.")

    return rotated_matrix

def evaluate_dimension_reduction(
        X: ArrayLike, n_features: int, n_components: int=5, cv: int=5 ):
    if isinstance(n_components, int):
        n_components = np.arange(1, n_features, n_components)
    elif not all(isinstance(x, int) and x > 0 for x in n_components):
        raise ValueError("n_components must be a positive integer or an"
                         " iterable of positive integers")
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []

    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        # Adding default 5-fold cross-validation
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=cv)))  
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=cv)))

    return np.array(pca_scores), np.array(fa_scores)


evaluate_dimension_reduction.__doc__ ="""\
Evaluate dimensionality reduction performance using PCA (Principal Component Analysis) 
and FA (Factor Analysis) for a dataset over a specified range or number of 
components. 
`evaluate_dimension_reduction` applies cross-validation to assess the 
effectiveness of each method.
  
Parameters 
-----------
{params.X}

n_features : int
    The total number of features in the dataset. This is used to determine 
    the range of  components if `n_components` is given as an integer.
n_components : int or iterable, optional
    Specifies the number of principal components to retain. If an integer, this 
    sets the step size for the range generated between 1 and `n_features`. If 
    an iterable, it specifies the exact numbers of components to evaluate. 
    Default is 5.
cv : int, optional
    Number of folds in the cross-validator. This parameter controls the number 
    of folds used in the stratified k-fold cross-validation process for 
    evaluating the models. Default is 5.
Returns
-------
tuple of lists
    Returns two lists containing the cross-validated scores for PCA and FA 
    respectively:
    - pca_scores: A list of mean cross-validated scores of PCA for each number 
      of components tested.
    - fa_scores: A list of mean cross-validated scores of Factor Analysis for 
      each number of components tested.

Raises
------
ValueError
    If `n_components` is neither a positive integer nor an iterable of positive 
    integers.

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from gofast.analysis.factors import evaluate_dimension_reduction
>>> X, _ = load_iris(return_X_y=True)
>>> pca_scores, fa_scores = evaluate_dimension_reduction(
...      X, n_features=X.shape[1], n_components=3)
>>> print(pca_scores, fa_scores)
[-3.7040051817034483] [-3.486523429414041]

""".format(params =_core_docs["params"])

def samples_hotellings_t_square(sample1: np.ndarray, sample2: np.ndarray):
    r"""
    Perform Two-Sample Hotelling's T-Square test to compare two
    multivariate samples.

    Hotelling's T-Square test assesses whether two multivariate samples come
    from populations with equal means. It is a multivariate extension of
    the two-sample t-test.

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

    Where:
    - :math:`T^2` is the Hotelling's T-Square statistic.
    - :math:`n_1` and :math:`n_2` are the sample sizes of the two groups.
    - :math:`\mathbf{\bar{x}}_1` and :math:`\mathbf{\bar{x}}_2` are the sample means
      of the two groups.
    - :math:`p` is the number of variables (dimensions).
    - :math:`\mathbf{S}_w` is the pooled within-group covariance matrix.
    - :math:`df_1` is the degrees of freedom (numerator).
    - :math:`df_2` is the degrees of freedom (denominator).
    - :math:`F` is the F-distribution cumulative distribution function.

    Parameters
    ----------
    sample1 : array_like
        The first multivariate sample with shape (n1, p), where n1 is the number
        of samples and p is the number of variables.
    sample2 : array_like
        The second multivariate sample with shape (n2, p), where n2 is the number
        of samples and p is the number of variables.

    Returns
    -------
    statistic : float
        The Hotelling's T-Square statistic.
    df1 : int
        Degrees of freedom numerator.
    df2 : int
        Degrees of freedom denominator.
    p_value : float
        The p-value of the test.
    
    Notes
    -----
    The Hotelling's T-Square test is particularly useful when dealing with
    multivariate data where multiple outcomes are interrelated. It extends
    the concept of the t-test to higher dimensions and provides a method to
    test the hypothesis that two groups have the same mean vector in the
    multidimensional space. This test is commonly used in quality control,
    behavioral science, and social sciences.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.analysis import samples_hotellings_t_square
    >>> sample1 = np.array([[1.2, 2.3], [1.8, 3.5], [2.6, 4.0]])
    >>> sample2 = np.array([[0.9, 2.0], [1.5, 3.2], [2.3, 3.8]])
    >>> statistic, df1, df2, p_value = samples_hotellings_t_square(sample1, sample2)
    >>> print(f"Hotelling's T-Square statistic: {statistic:.4f}")
    >>> print(f"Degrees of freedom (numerator): {df1}")
    >>> print(f"Degrees of freedom (denominator): {df2}")
    >>> print(f"P-value: {p_value:.4f}")
    Hotelling's T-Square statistic: 0.4912
    Degrees of freedom (numerator): 2
    Degrees of freedom (denominator): 3
    P-value: 0.7641
    """
    n1, p = sample1.shape
    n2 = sample2.shape[0]

    # Calculate sample means
    mean1 = np.mean(sample1, axis=0)
    mean2 = np.mean(sample2, axis=0)

    # Calculate sample covariances
    cov1 = np.cov(sample1, rowvar=False, ddof=1)
    cov2 = np.cov(sample2, rowvar=False, ddof=1)

    # Pooled within-group covariance matrix
    pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

    # Calculate Hotelling's T-Square statistic
    mean_diff = mean1 - mean2
    inv_pooled_cov = np.linalg.inv(pooled_cov)
    t_square = (n1 * n2) / (n1 + n2) * np.dot(mean_diff, np.dot(inv_pooled_cov, mean_diff))

    # Degrees of freedom
    df1 = p
    df2 = n1 + n2 - p - 1

    # Calculate p-value using the F-distribution
    f_stat = t_square * (df2 / (df1 * df2 - p + 1))
    p_value = 1 - statsf.cdf(f_stat, df1, df2)

    return t_square, df1, df2, p_value


