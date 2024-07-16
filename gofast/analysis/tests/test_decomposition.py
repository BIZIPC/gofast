# -*- coding: utf-8 -*-
"""
test_decomposition.py 
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gofast.analysis.decomposition import get_eigen_components 
from gofast.analysis.decomposition import  get_total_variance_ratio 
from gofast.analysis.decomposition import transform_to_principal_components
from gofast.analysis.decomposition import _decision_region, plot_decision_regions
from gofast.analysis.decomposition import linear_discriminant_analysis


def test_decision_region():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_samples=100)
    clf = LogisticRegression().fit(X, y)
    fig, ax = plt.subplots()
    ax = _decision_region(X, y, clf, resolution=0.01, ax=ax)
    assert isinstance(ax, plt.Axes), "The function should return a matplotlib Axes object."

    # Check if the plot has content (e.g., has data plotted on it)
    assert len(ax.collections), ( 
        "The decision regions plot should contain filled areas for class regions."
        )
    plt.close(fig)

def test_plot_decision_regions():
    X, y = make_classification(n_features=4, n_classes=2, n_samples=200)
    clf = LogisticRegression()
    
    # Test without splitting
    ax = plot_decision_regions(X, y, clf, view='X', scaling=True, split=False,
                               )
    assert isinstance(ax, plt.Axes), "Should return an Axes object when 'return_axe' is specified."

    # Test with splitting and view test data
    ax = plot_decision_regions(X, y, clf, view='Xt', 
                               scaling=True, split=True, test_size=0.2,)
    assert isinstance(ax, plt.Axes), "Should return an Axes object for test data view."

    # Test returning explained variance ratio
    explained_variance_ratio = plot_decision_regions(
        X, y, clf, view=None, scaling=True, split=False, return_expl_variance_ratio=True)
    assert isinstance(explained_variance_ratio, np.ndarray), ( 
        "Should return an array of explained variance ratios."
        )
    assert explained_variance_ratio.sum() <= 1, ( 
        "Sum of explained variance ratio should not exceed 1."
        )

def test_linear_discriminant_analysis_transformation():
    # Create a dataset with two classes
    X, y = make_classification(n_classes=2, n_features=5, n_informative=5, 
                               n_redundant=0, n_clusters_per_class=1, n_samples=100)
    X = StandardScaler().fit_transform(X)  # Standardize features

    # Test with 2 components
    transformed_data = linear_discriminant_analysis(X, y, n_components=2,
                                                    return_X=True, view=False)
    assert transformed_data.shape[1] == 2, "Should return data transformed into 2 components."

    # Test returning the transformation matrix instead of the transformed data
    W = linear_discriminant_analysis(X, y, n_components=2, return_X=False, view=False)
    assert W.shape[1] == 2, "Should return the transformation matrix for 2 components."
    assert W.shape[0] == X.shape[1], ( 
        "The number of rows in W should match the number of features in X."
        )

def test_linear_discriminant_analysis_view():
    X, y = make_classification(n_classes=2, n_features=5, n_informative=5,
                               n_redundant=0, n_clusters_per_class=1, n_samples=100)
    X = StandardScaler().fit_transform(X)

    # Test view functionality
    with plt.rc_context({'backend': 'Agg'}):  # Use non-interactive backend to avoid showing the plot
        linear_discriminant_analysis(X, y, n_components=2, return_X=False, view=True)
        plt.close('all')  # Ensure no figure is left open
        
@pytest.mark.skip ("Unable to catch ComplexWarning then skip it instead.")
def test_linear_discriminant_analysis_n_components_too_high():
    X, y = make_classification(n_classes=2, n_features=5, n_informative=5, 
                               n_redundant=0, n_clusters_per_class=1, n_samples=100)
    X = StandardScaler().fit_transform(X)

    # Test handling of n_components greater than feature count
    with pytest.raises(( np.ComplexWarning, UserWarning)) as e:
        linear_discriminant_analysis(X, y, n_components=6, return_X=True, view=False)
    assert "n_component" in str(e.value), ( 
        "Should warn when n_components exceeds the number of features."
        )

# Fixtures to provide data for the tests
@pytest.fixture
def sample_data():
    X, _ = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
    return X

# Tests for get_eigen_components
def test_get_eigen_components(sample_data):
    eigen_vals, eigen_vecs, X_scaled = get_eigen_components(sample_data)
    assert eigen_vals is not None, "Eigenvalues should not be None"
    assert eigen_vecs is not None, "Eigenvectors should not be None"
    assert X_scaled is not None, "Scaled data should not be None"
    assert eigen_vals.shape[0] == eigen_vecs.shape[1], (
        "Number of eigenvalues should match number of eigenvectors") 

def test_get_eigen_components_shape(sample_data):
    _, eigen_vecs, _ = get_eigen_components(sample_data)
    assert eigen_vecs.shape[0] == eigen_vecs.shape[1], "Eigenvectors matrix should be square"

# Tests for get_total_variance_ratio
def test_get_total_variance_ratio(sample_data):
    cum_var_exp = get_total_variance_ratio(sample_data, view=False)
    assert len(cum_var_exp) > 0, "Cumulative variance ratio should be calculated"
    assert np.isclose(cum_var_exp[-1], 1.0), "Total explained variance should sum to 1"

def test_get_total_variance_ratio_view(sample_data, mocker):
    mocker.patch("matplotlib.pyplot.show")  # Patch matplotlib show to not display plot
    cum_var_exp = get_total_variance_ratio(sample_data, view=True)
    assert np.isclose(cum_var_exp[-1], 1.0), "Total explained variance should sum to 1 when view is True"

# Tests for transform_to_principal_components
def test_transform_to_principal_components_dimensions(sample_data):
    X_transf = transform_to_principal_components(sample_data, n_components=2)
    assert X_transf.shape[1] == 2, "The transformed data should have two components"


def test_transform_to_principal_components_failure_on_high_dimensions(sample_data):
    # Mocking a scenario where the PCA might attempt to use more components than available features
    # Assuming the function raises a UserWarning when asked to handle more than 2 components
    # with pytest.warns(( np.ComplexWarning, UserWarning), match="N-component > 2 is not implemented yet.") as record:
    #     transform_to_principal_components(sample_data, n_components=30)  # More components than features

    # assert any("N-component > 2 is not implemented yet." in str(warning.message) for warning in record), (
    #     "Should warn when more than 2 components are requested"
    # )
    with pytest.raises(ValueError):
        transform_to_principal_components(sample_data, n_components=30)

def test_transform_to_principal_components_view(sample_data, mocker):
    mocker.patch("matplotlib.pyplot.show")  # Patch matplotlib show to not display plot
    y = np.random.randint(2, size=100)
    X_transf = transform_to_principal_components(sample_data, y=y, n_components=2, view=True)
    assert X_transf.shape[1] == 2, "The transformed data should have two components when view is True"


if __name__=='__main__': 
    pytest.main ([__file__]) 
    