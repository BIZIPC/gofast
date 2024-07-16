# -*- coding: utf-8 -*-
"""
test_factors.py 
"""

import numpy as np
import pytest
from sklearn.datasets import make_spd_matrix

from gofast.analysis.factors import spectral_factor, promax_rotation 
from gofast.analysis.factors import ledoit_wolf_score 
from gofast.analysis.factors import evaluate_noise_impact_on_reduction  
from gofast.analysis.factors import make_scedastic_data
from gofast.analysis.factors import rotated_factor, principal_axis_factoring
from gofast.analysis.factors import varimax_rotation, oblimin_rotation 
from gofast.analysis.factors import evaluate_dimension_reduction 
from gofast.analysis.factors import samples_hotellings_t_square 

def test_samples_hotellings_t_square_basic():
    sample1 = np.random.rand(10, 5)
    sample2 = np.random.rand(10, 5)
    stat, df1, df2, p_value = samples_hotellings_t_square(sample1, sample2)
    assert isinstance(stat, float) and isinstance(p_value, float), \
        "Statistic and p-value should be floats"
    assert df1 > 0 and df2 > 0, "Degrees of freedom should be positive"

def test_samples_hotellings_t_square_identical():
    sample = np.random.rand(10, 5)
    stat, df1, df2, p_value = samples_hotellings_t_square(sample, sample)
    assert p_value == 1.0, "P-value should be 1 for identical samples"

def test_evaluate_dimension_reduction_basic():
    X = np.random.rand(100, 50)
    pca_scores, fa_scores = evaluate_dimension_reduction(X, 50)
    assert isinstance(pca_scores, np.ndarray) and isinstance(
        fa_scores, np.ndarray), "Outputs should be lists"
    assert len(pca_scores) > 0 and len(fa_scores) > 0,\
        "Should have non-zero length scores"

def test_varimax_rotation_basic():
    loadings = np.random.rand(10, 2)
    rotated = varimax_rotation(loadings)
    assert rotated.shape == loadings.shape, \
        "Varimax rotation should not change dimensions"

def test_oblimin_rotation_basic():
    loadings = np.random.rand(10, 2)
    rotated = oblimin_rotation(loadings)
    assert rotated.shape == loadings.shape,\
        "Oblimin rotation should not change dimensions"


def test_principal_axis_factoring_basic():
    X = np.random.rand(100, 10)
    factors = principal_axis_factoring(X, n_factors=2)
    assert factors.shape == (10, 2), "Should return two factors for 10 features"

def test_rotated_factor_basic():
    X = np.random.rand(100, 10)
    rotated = rotated_factor(X, n_components=2)
    assert rotated.shape == (100, 2), "Output should have two components for 100 samples"

def test_rotated_factor_invalid_rotation():
    X = np.random.rand(10, 5)
    with pytest.raises(ValueError):
        rotated_factor(X, n_components=2, rotation='invalid_rotation')

def test_make_scedastic_data_basic():
    X, X_homo, X_hetero, n_components = make_scedastic_data()
    assert X.shape == (1000, 50), "Check X dimensions"
    assert X_homo.shape == X.shape and X_hetero.shape == X.shape, ( 
        "Homoscedastic and Heteroscedastic noise added data should match X dimensions"
        )
    assert len(n_components) > 0, "Should have non-zero n_components options"

def test_make_scedastic_data_extreme_params():
    X, X_homo, X_hetero, n_components = make_scedastic_data(
        n_samples=1, n_features=1, rank=1, sigma=0)
    assert X.shape == (1, 1), ( 
        "Single sample, single feature should be reflected in output shape"
        )
    assert np.allclose(X, X_homo), "With sigma=0, X and X_homo should be identical"
    assert np.allclose(X, X_hetero), ( 
        "X and X_hetero should be nearly identical if sigma is effectively 0"
        )

def test_evaluate_noise_impact_on_reduction():
    # Create a random dataset
    X = np.random.rand(100, 50)
    
    # Define parameters for the test
    rank = 10
    sigma = 0.5
    step = 5
    random_state = 42
    verbose = 1
    
    # Test without plotting to check the function's return values
    results = evaluate_noise_impact_on_reduction(
        X,
        rank=rank,
        sigma=sigma,
        step=step,
        random_state=random_state,
        verbose=verbose,
        display_plots=False
    )
    # Assertions to check if the output is correctly structured and contains expected keys
    assert isinstance(results, dict), "Output should be a dictionary"
    assert 'homoscedastic_noise' in results,\
        "Results should contain 'homoscedastic_noise'"
    assert 'heteroscedastic_noise' in results,\
        "Results should contain 'heteroscedastic_noise'"
    assert 'pca_scores' in results['homoscedastic_noise'], ( 
        "Each noise condition should contain PCA scores" )
    assert 'fa_scores' in results['heteroscedastic_noise'], ( 
        "Each noise condition should contain FA scores" ) 
    
    # Checking specific data types and values
    assert isinstance(results['homoscedastic_noise']['pca_scores'], np.ndarray
                      ), "PCA scores should be a numpy array"
    assert results['homoscedastic_noise']['pca_scores'].shape[0] > 0, ( 
        "PCA scores array should not be empty")
    assert isinstance(results['homoscedastic_noise']['n_components_pca_mle'],
                      (np.integer, int)), "MLE components should be an integer"
    
def test_ledoit_wolf_score_basic_functionality():
    data = np.random.rand(100, 10)
    score = ledoit_wolf_score(data)
    assert isinstance(score, float)

@pytest.mark.parametrize("assume_centered", [True, False])
def test_ledoit_wolf_score_centering(assume_centered):
    data = np.random.rand(100, 10)
    score = ledoit_wolf_score(data, assume_centered=assume_centered)
    assert isinstance(score, float)

def test_ledoit_wolf_score_store_precision():
    data = make_spd_matrix(10)  # Generates a symmetric positive definite matrix
    score = ledoit_wolf_score(data, store_precision=True)
    assert isinstance(score, float)

def test_promax_rotation_invalid_input():
    with pytest.raises(TypeError):
        promax_rotation("not a matrix")  # Invalid type
    with pytest.raises(ValueError):
        promax_rotation(np.array([1, 2, 3]))  # 1D array instead of 2D

@pytest.mark.parametrize("power", [1, 2, 4, 1])
def test_promax_rotation_various_powers(power):
    loadings = np.array([[0.7, 0.1, 0.3], [0.8, 0.2, 0.4],
                         [0.4, 0.6, 0.5], [0.3, 0.7, 0.2]])
    rotated_loadings = promax_rotation(loadings, power=power)
    assert rotated_loadings.shape == loadings.shape

def test_promax_rotation_zero_negative_power():
    loadings = np.array([[0.5, 0.6], [0.7, 0.8]])
    # Assuming the function should handle zero or negative powers correctly
    with pytest.raises(ValueError):
        promax_rotation(loadings, power=-1)

def test_spectral_fa_invalid_input():
    with pytest.raises(TypeError):
        # Passing a string instead of an ndarray
        spectral_factor("not an array")  
    with pytest.raises(ValueError):
        spectral_factor(np.array([1, 2, 3]))  # Passing a 1D array

@pytest.mark.parametrize("num_factors", [None, 1, 2, 3])
def test_spectral_fa_various_factors(num_factors):
    data = np.array([[1, 2], [3, 4], [5, 6]])
    loadings, factors, eigenvalues = spectral_factor(
        data, num_factors=num_factors)
    n_features = data.shape[1]
    if num_factors is None:
        num_factors = n_features
    assert loadings.shape == (n_features, min(num_factors, n_features))
    assert factors.shape == (data.shape[0], min(num_factors, n_features))
    assert eigenvalues.shape == (min(num_factors, n_features),)

def test_spectral_fa_zero_size():
    data = np.array([[]])
    with pytest.raises(ValueError):
        spectral_factor(data)

def test_spectral_fa_high_dimensional():
    # High dimensional data where features > samples
    data = np.random.rand(10, 100)
    loadings, factors, eigenvalues = spectral_factor(data, num_factors=5)
    assert loadings.shape == (100, 5)
    assert factors.shape == (10, 5)
    assert eigenvalues.shape == (5,)

def test_spectral_fa_output_consistency():
    data = np.random.rand(100, 5)
    loadings, factors, eigenvalues = spectral_factor(data)
    assert all(np.diff(eigenvalues) <= 0)  # Check eigenvalues are sorted in descending order

if __name__=='__main__': 
    
    pytest.main([__file__])