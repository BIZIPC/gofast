# -*- coding: utf-8 -*-
"""
test_hierarchical_linear_model.py
@author: Daniel
"""


import numpy as np
import pytest
from scipy.stats import norm

from gofast.stats.probs import hierarchical_linear_model  
from gofast.stats.probs import stochastic_volatility_model 
from gofast.stats.probs import uniform_sampling, poisson_logpmf
from gofast.stats.probs import binomial_pmf, normal_cdf, normal_pdf 

def is_pymc3_installed (): 
    try : 
        import  pymc3 # noqa
        return True 
    except: return False 
    
PYMC3_INSTALLED= is_pymc3_installed() 

@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_hierarchical_linear_model_basic():
    # Dummy data
    X = np.random.normal(size=(100, 3))
    y = np.random.normal(size=100)
    groups = np.random.randint(0, 5, size=100)

    # Test without sampling
    model = hierarchical_linear_model(X, y, groups, n_samples=0, 
                                      return_inference_data=False)
    assert model is not None, "Model should not be None"
    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_stochastic_volatility_model_basic():
    # Dummy data
    returns = np.random.normal(0, 1, 100)

    # Test without sampling
    model = stochastic_volatility_model(returns, sampling=False,
                                        return_inference_data=False)
    assert model is not None, "Model should not be None"
    
@pytest.mark.slow  # Use this decorator if you want to skip slow tests with pytest -k "not slow"    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'hierarchical_linear_model'")
    )
def test_hierarchical_linear_model_with_sampling():
    # Dummy data
    import  pymc3 as pm  # 
    X = np.random.normal(size=(20, 2))  # Reduced size for speed
    y = np.random.normal(size=20)
    groups = np.random.randint(0, 3, size=20)  # Reduced number of groups for speed

    # Test with sampling
    model, trace = hierarchical_linear_model(X, y, groups, sampling=True, n_samples=50, 
                                             return_inference_data=False)
    
    assert model is not None, "Model should not be None"
    assert isinstance(trace, pm.backends.base.MultiTrace), "Trace should be a MultiTrace object"
    assert len(trace) == 50, "Trace should contain the specified number of samples"

@pytest.mark.slow    
@pytest.mark.skipif (not PYMC3_INSTALLED, reason=(
    "Need pymc3 to be installed for running 'stochastic_volatility_model'")
    )
def test_stochastic_volatility_model_with_sampling():
    import  pymc3 as pm  # 
    # Dummy data
    returns = np.random.normal(0, 1, 50)  # Reduced size for speed

    # Test with sampling
    model, trace = stochastic_volatility_model(
        returns, sampling=True, n_samples=50, return_inference_data=False)

    assert model is not None, "Model should not be None"
    assert isinstance(trace, pm.backends.base.MultiTrace), "Trace should be a MultiTrace object"
    assert len(trace) == 50, "Trace should contain the specified number of samples"

# Tests for uniform_sampling
def test_uniform_sampling_range():
    low, high, size = 0, 10, 1000
    samples = uniform_sampling(low, high, size)
    assert np.all(samples >= low) and np.all(samples < high), "Samples outside specified range"

def test_uniform_sampling_size():
    size = 100
    samples = uniform_sampling(0, 10, size)
    assert len(samples) == size, "Incorrect number of samples generated"

# Tests for poisson_logpmf
def test_poisson_logpmf_positive_lambda():
    lambda_param = 5
    data = 3
    result = poisson_logpmf(data, lambda_param)
    assert result < 0, "Log PMF should be negative"

# Tests for binomial_pmf
def test_binomial_pmf_sum_to_one():
    trials, p_success = 10, 0.5
    pmf_values = [binomial_pmf(trials, p_success, k) for k in range(trials + 1)]
    assert np.isclose(sum(pmf_values), 1), "PMF values do not sum to 1"

# Tests for normal_cdf
def test_normal_cdf_limits():
    data = np.array([-10, 10])
    cdf_values = normal_cdf(data)
    assert np.isclose(cdf_values[0], 0, atol=1e-2) and np.isclose(
        cdf_values[1], 1, atol=1e-2), "CDF limits incorrect"


def test_normal_pdf_values():
    """
    Test the normal_pdf function for specific values and parameters.
    """
    data = np.array([0, 1, -1])
    mean = 0
    std_dev = 1
    scale = 1
    loc = 0
    expected_pdf_values = norm.pdf(data, mean, std_dev)

    pdf_values = normal_pdf(data, mean, std_dev, scale, loc)

    assert np.allclose(pdf_values, expected_pdf_values),( 
        "PDF values do not match expected values for standard normal distribution.")

def test_normal_pdf_scale():
    """
    Test the normal_pdf function with a scaling factor.
    """
    data = np.array([0])
    mean = 0
    std_dev = 1
    scale = 2
    loc = 0
    expected_scaled_pdf_value = norm.pdf(data, mean, std_dev) * scale

    scaled_pdf_values = normal_pdf(data, mean, std_dev, scale, loc)

    assert np.allclose(scaled_pdf_values, expected_scaled_pdf_value), ( 
        "Scaled PDF values do not match expected values.")

def test_normal_pdf_loc():
    """
    Test the normal_pdf function with a location shift.
    """
    data = np.array([2])
    mean = 0
    std_dev = 1
    scale = 1
    loc = 2  # Shifting the data so that 2 becomes the "new zero"
    expected_shifted_pdf_value = norm.pdf(0, mean, std_dev)
    shifted_pdf_values = normal_pdf(data, mean, std_dev, scale, loc)
    assert np.allclose(shifted_pdf_values, expected_shifted_pdf_value), ( 
        "Shifted PDF values do not match expected values."
        )


if __name__=='__main__': 
    pytest.main([__file__]) 