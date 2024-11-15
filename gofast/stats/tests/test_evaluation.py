# -*- coding: utf-8 -*-
# test_evaluation.py

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils._param_validation import InvalidParameterError 
from scipy.stats import entropy
import matplotlib.pyplot as plt

from gofast.stats.evaluation import (
    BayesianMethods,
    DistributionComparison,
    ErrorAnalysis,
    FeatureImportanceTests,
    GoodnessOfFit,
    InformationCriteria,
    ModelComparison,
    NormalityTests,
    ProbabilisticModels,
    RegressionModel,
    ResidualAnalysis,
    SequentialTesting,
    TimeSeriesTests,
    VarianceComparison,
    NonParametrics,
    ModelRobustness,
)
#
# Conditional imports for BayesianMethods
try:
    import pymc3 as pm
    import arviz as az # noqa
    PYMCMC_AVAILABLE = True
except ImportError:
    PYMCMC_AVAILABLE = False


try:
    from scipy.stats import jensenshannon
    # This is import older version. In newest version, jensenshannon 
    # has been moved to distance module:
    # from scipy.stats.distance import jensenshannon 
    
except: 
    # neverthess we can mannually implement this function as 
    # below based on entropy 
    
    def jensenshannon(p, q):
        """Compute Jensen-Shannon divergence between'
        two probability distributions."""
        p = np.array(p)
        q = np.array(q)
        m = 0.5 * (p + q)
        return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))
    
# Fixtures for common data
@pytest.fixture
def regression_data():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1.2, 1.9, 3.0, 4.1, 5.1])
    return X, y

@pytest.fixture
def classification_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

@pytest.fixture
def time_series_data():
    np.random.seed(0)
    return np.random.normal(loc=0, scale=1, size=100)

@pytest.fixture
def error_analysis_data():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred1 = np.array([2.5, 0.0, 2.1, 7.8])
    y_pred2 = np.array([3.2, -0.3, 1.8, 6.9])
    return y_true, y_pred1, y_pred2

@pytest.fixture
def permutation_importance_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

@pytest.fixture
def variance_comparison_data():
    group1 = np.array([10, 12, 14, 16, 18])
    group2 = np.array([20, 22, 24, 26, 28])
    group3 = np.array([30, 32, 34, 36, 38])
    return group1, group2, group3

@pytest.fixture
def normality_data():
    return np.random.normal(loc=0, scale=1, size=100)

@pytest.fixture
def nonparametric_data():
    group1 = np.array([10, 12, 14, 16, 18])
    group2 = np.array([20, 22, 24, 26, 28])
    group3 = np.array([30, 32, 34, 36, 38])
    return group1, group2, group3

@pytest.fixture
def model_comparison_data():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 1, 0, 1])
    return X, y

@pytest.fixture
def model_robustness_data():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 1, 0, 1])
    return X, y

# Fixtures for SequentialTesting
@pytest.fixture
def sequential_testing_data_accept_h1():
    """
    Fixture for SequentialTesting with data leading to 'Accept H1'.
    """
    # Data that should lead to accepting H1
    # Assuming p1 > p0, so a series of successes should 
    # accumulate the log-likelihood ratio
    data = [1] * 10  # all successes
    return data

@pytest.fixture
def sequential_testing_data_accept_h0():
    """
    Fixture for SequentialTesting with data leading to 'Accept H0'.
    """
    # Data that should lead to accepting H0
    # Assuming p1 > p0, so a series of failures should
    # accumulate the log-likelihood ratio
    data = [0] * 10  # all failures
    return data

@pytest.fixture
def sequential_testing_data_inconclusive():
    """
    Fixture for SequentialTesting with data leading to 'Inconclusive'.
    """
    # Data that leads to an inconclusive decision
    # A mix of successes and failures that do not cross the thresholds
    data = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    return data

# Fixtures for DistributionComparison
@pytest.fixture
def distribution_comparison_data_identical():
    """
    Fixture for DistributionComparison with identical distributions.
    """
    P = [0.2, 0.3, 0.5]
    Q = [0.2, 0.3, 0.5]
    return P, Q

@pytest.fixture
def distribution_comparison_data_different():
    """
    Fixture for DistributionComparison with different distributions.
    """
    P = [0.1, 0.4, 0.5]
    Q = [0.2, 0.2, 0.6]
    return P, Q

@pytest.fixture
def distribution_comparison_data_non_normalized():
    """
    Fixture for DistributionComparison with non-normalized distributions.
    """
    P = [2, 3, 5]  # sums to 10
    Q = [1, 2, 7]  # sums to 10
    return P, Q

# Mock plt.show to prevent actual plotting during tests
@pytest.fixture(autouse=True)
def mock_plot_show(monkeypatch):
    """
    Fixture to mock plt.show() to prevent plots from appearing during tests.
    """
    monkeypatch.setattr(plt, "show", lambda: None)

# Test BayesianMethods only if pymc3 is available
@pytest.mark.skipif(not PYMCMC_AVAILABLE, reason="pymc3 is not installed.")
def test_BayesianMethods():

    # Define two simple Bayesian models
    with pm.Model() as model_a:
        mu = pm.Normal('mu', mu=0, sigma=1)
        pm.Normal('obs', mu=mu, sigma=1, observed=np.random.randn(100))

    with pm.Model() as model_b:
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)
        pm.Normal('obs', mu=mu, sigma=sigma, observed=np.random.randn(100))

    models = {'ModelA': model_a, 'ModelB': model_b}
    bm = BayesianMethods(fit_models=True, draws=100, tune=100, random_seed=42, cores=1)
    bm.run(models)

    # Check if traces are recorded
    assert 'ModelA' in bm.traces_
    assert 'ModelB' in bm.traces_

    # Test bayes_factor method
    bf = bm.bayes_factor('ModelA', 'ModelB', method='log_likelihood')
    assert isinstance(bf, float)

    # Test highest_posterior_density_interval method
    hpd = bm.highest_posterior_density_interval('ModelA', 'mu', credible_interval=0.95)
    assert isinstance(hpd, np.ndarray)
    assert hpd.shape == (2,)

    # Since plot does not return anything, just ensure it runs without error
    bm.plot('ModelA', 'mu')

# Test TimeSeriesTests
def test_TimeSeriesTests(time_series_data):
    tst = TimeSeriesTests(lags=10)
    tst.run(time_series_data)

    # Test augmented_dickey_fuller
    adf_result = tst.augmented_dickey_fuller()
    assert isinstance(adf_result, tuple)
    assert len(adf_result) == 6  # adf_stat, pvalue, usedlag, nobs, critical_values, icbest

    # Test ljung_box
    lb_stat, lb_pvalue = tst.ljung_box()
    assert isinstance(lb_stat, np.ndarray)
    assert isinstance(lb_pvalue, np.ndarray)
    assert len(lb_stat) == 10  # as lags=10

    # Test plot
    tst.plot()

# Test ErrorAnalysis
def test_ErrorAnalysis(error_analysis_data):
    y_true, y_pred1, y_pred2 = error_analysis_data
    ea = ErrorAnalysis(alpha=0.05)
    ea.fit(y_true, y_pred1, y_pred2)

    # Test mse_significance
    z, p = ea.mse_significance(benchmark_mse=0.5)
    assert isinstance(z, float)
    assert isinstance(p, float)

    # Test diebold_mariano
    dm_stat, dm_pvalue = ea.diebold_mariano()
    assert isinstance(dm_stat, float)
    assert isinstance(dm_pvalue, float)

    # Test plot
    ea.plot()

# Test BayesianMethods with missing dependencies
# This test is wierd, nevertheless, wrote it for testing 
# the "ensure_pkg" from gofast.tools.depsutils conditionnaly import.
@pytest.mark.skipif(PYMCMC_AVAILABLE, reason="pymc3 is installed.")
def test_BayesianMethods_skip():
    with pytest.raises(ImportError):
        BayesianMethods()

# Test FeatureImportanceTests
def test_FeatureImportanceTests(permutation_importance_data):
    X, y = permutation_importance_data
    model = RandomForestClassifier()
    fit = FeatureImportanceTests(model=model, scoring='accuracy', 
                                 n_repeats=10, random_state=42)
    fit.fit(X, y)

    # Test permutation_importance_test
    importances_mean, importances_std = fit.permutation_importance_test()
    assert isinstance(importances_mean, np.ndarray)
    assert isinstance(importances_std, np.ndarray)
    assert len(importances_mean) == X.shape[1]

    # Test plot
    fit.plot()

# Test VarianceComparison
def test_VarianceComparison(variance_comparison_data):
    group1, group2, group3 = variance_comparison_data
    vc = VarianceComparison(alpha=0.05)
    vc.fit(group1, group2, group3)

    # Test f_test
    f_stat, p_value = vc.f_test(0, 1)
    assert isinstance(f_stat, float)
    assert isinstance(p_value, float)

    # Test levene_test
    stat, p = vc.levene_test(center='median')
    assert isinstance(stat, float)
    assert isinstance(p, float)

    # Test plot
    vc.plot()

# Test NormalityTests
def test_NormalityTests(normality_data):
    nt = NormalityTests(alpha=0.05)
    nt.fit(normality_data)

    # Test shapiro_wilk_test
    stat, p = nt.shapiro_wilk_test()
    assert isinstance(stat, float)
    assert isinstance(p, float)

    # Test anderson_darling_test
    stat_ad, crit_vals, sig_levels = nt.anderson_darling_test(dist='norm')
    assert isinstance(stat_ad, float)
    assert isinstance(crit_vals, np.ndarray)
    assert isinstance(sig_levels, np.ndarray)

    # Test plot
    nt.plot()

# Test GoodnessOfFit
def test_GoodnessOfFit(classification_data):
    X, y = classification_data
    model = LogisticRegression()
    gof = GoodnessOfFit(model=model, alpha=0.05, fit_model=True)
    gof.fit(X, y)

    # Test chi_square_test
    chi2, p_value = gof.chi_square_test()
    assert isinstance(chi2, float)
    assert isinstance(p_value, float)

    # Test kolmogorov_smirnov_test
    d_stat, p_ks = gof.kolmogorov_smirnov_test()
    assert isinstance(d_stat, float)
    assert isinstance(p_ks, float)

    # # Test hosmer_lemeshow_test
    # chi2_hl, p_hl = gof.hosmer_lemeshow_test(groups=10)
    # assert isinstance(chi2_hl, float)
    # assert isinstance(p_hl, float)

    # Test plot
    gof.plot()

# Test InformationCriteria
def test_InformationCriteria(regression_data):
    X, y = regression_data
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': clone(LinearRegression())
    }
    ic = InformationCriteria(models=models, fit_models=True)
    ic.fit(X, y)

    # Test aic method
    aic_values = ic.aic()
    assert isinstance(aic_values, dict)
    assert 'LinearRegression' in aic_values
    assert 'Ridge' in aic_values

    # Test bic method
    bic_values = ic.bic()
    assert isinstance(bic_values, dict)
    assert 'LinearRegression' in bic_values
    assert 'Ridge' in bic_values

    # Test plot method
    ic.plot(criterion='AIC')
    ic.plot(criterion='BIC')

    # Test invalid criterion
    with pytest.raises(ValueError):
        ic.plot(criterion='INVALID')

# Test ProbabilisticModels
def test_ProbabilisticModels(classification_data):
    X, y = classification_data
    model = LogisticRegression()
    pm_model = ProbabilisticModels(model=model, fit_model=True, n_bins=10)
    pm_model.fit(X, y)

    # Test brier_score
    bs = pm_model.brier_score()
    assert isinstance(bs, float)

    # Test log_likelihood_test
    ll = pm_model.log_likelihood_test()
    assert isinstance(ll, float)

    # Test calibration_error
    ece = pm_model.calibration_error()
    assert isinstance(ece, float)

    # Test plot
    pm_model.plot()

# Test ResidualAnalysis
def test_ResidualAnalysis(regression_data):
    X, y = regression_data
    model = LinearRegression()
    ra = ResidualAnalysis(model=model, fit_model=True)
    ra.fit(X, y)

    # Test durbin_watson_test
    dw_stat = ra.durbin_watson_test()
    assert isinstance(dw_stat, float)

    # Test breusch_pagan_test
    bp_stat, bp_pvalue = ra.breusch_pagan_test()
    assert isinstance(bp_stat, float)
    assert isinstance(bp_pvalue, float)

    # Test plot
    ra.plot()

# Test RegressionModelTests
def test_RegressionModelTests(regression_data):
    X_full, y = regression_data
    X_reduced = X_full[:, :1]
    import statsmodels.api as sm

    full_model = sm.OLS(y, sm.add_constant(X_full))
    reduced_model = sm.OLS(y, sm.add_constant(X_reduced))

    rmt = RegressionModel(full_model=full_model, reduced_model=reduced_model)
    rmt.fit(X_full, y, X_reduced=X_reduced)

    # Test t_test_coefficients
    t_vals, p_vals = rmt.t_test_coefficients()
    assert isinstance(t_vals, np.ndarray)
    assert isinstance(p_vals, np.ndarray )
    assert len(t_vals) == X_full.shape[1] + 1  # including intercept

    # Test partial_f_test
    f_stat, p_value = rmt.partial_f_test()
    assert isinstance(f_stat, float)
    assert isinstance(p_value, float)

    # Test plot
    rmt.plot()


# Test NonParametrics
def test_NonParametrics(nonparametric_data):
    group1, group2, group3 = nonparametric_data
    np_tests = NonParametrics(alpha=0.05)
    
    # Run with the groups
    np_tests.run(group1, group2, group3)
    
    # Test groups_ attribute
    assert hasattr(np_tests, 'groups_')
    assert np_tests.groups_ == (group1, group2, group3)
    
    # Test Mann-Whitney U test between group1 and group2
    u_stat, p_value = np_tests.mann_whitney_u_test(0, 1, alternative='two-sided')
    assert isinstance(u_stat, float)
    assert isinstance(p_value, float)
    
    # Edge case: identical groups
    u_stat_identical, p_value_identical = np_tests.mann_whitney_u_test(0, 0, alternative='two-sided')
    assert isinstance(u_stat_identical, float)
    assert isinstance(p_value_identical, float)
    
    # Test Kruskal-Wallis H-test across all groups
    h_stat, p_val = np_tests.kruskal_wallis_test()
    assert isinstance(h_stat, float)
    assert isinstance(p_val, float)
    
    # Test plot method (ensure it runs without error)
    np_tests.plot()
    
# Test ModelComparison
def test_ModelComparison(model_comparison_data):
    X, y = model_comparison_data
    models = {
        'LogisticRegression': LogisticRegression(),
        'RidgeClassifier': RidgeClassifier(),
        'RandomForest': RandomForestClassifier()
    }
    mc = ModelComparison(
        models=models, alpha=0.05, 
        fit_models=True,
        scoring=accuracy_score
        )
    
    # Fit models and compute metrics with cv=2
    mc.fit(X, y, cv=2)
    
    # Check if predictions and metrics are recorded
    assert hasattr(mc, 'predictions_')
    assert hasattr(mc, 'metrics_')
    assert set(mc.predictions_.keys()) == set(models.keys())
    assert set(mc.metrics_.keys()) == set(models.keys())
    
    # Test paired_t_test between LogisticRegression and RidgeClassifier
    # Note: Paired t-test expects paired samples; ensure models have same number of scores
    if len(mc.metrics_['LogisticRegression']) == len(mc.metrics_['RidgeClassifier']):
        t_stat, p_value = mc.paired_t_test('LogisticRegression', 'RidgeClassifier')
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
    
    # Test McNemar's test between LogisticRegression and RandomForest
    chi2, p_mcnemar = mc.mcnemar_test('LogisticRegression', 'RandomForest')
    assert isinstance(chi2, float)
    assert isinstance(p_mcnemar, float)
    
    # Test Wilcoxon signed-rank test between RidgeClassifier and RandomForest
    w_stat, p_wilcoxon = mc.wilcoxon_signed_rank_test('RidgeClassifier', 'RandomForest')
    assert isinstance(w_stat, float)
    assert isinstance(p_wilcoxon, float)
    
    # Test Friedman test across all models
    f_stat, p_friedman = mc.friedman_test()
    assert isinstance(f_stat, float)
    assert isinstance(p_friedman, float)
    
    # Test ANOVA test across all models
    f_anova, p_anova = mc.anova_test()
    assert isinstance(f_anova, float)
    assert isinstance(p_anova, float)
    
    # Test plot method
    mc.plot(metric_name='Accuracy')
    
    # Test invalid model names
    with pytest.raises(KeyError):
        mc.paired_t_test('ModelA', 'ModelB')  # Non-existent models
    
# Test ModelRobustness
def test_ModelRobustness(model_robustness_data):
    X, y = model_robustness_data
    model = RandomForestClassifier()
    sensitivity_params = {'feature_index': 0, 'delta': 0.1}
    mr = ModelRobustness(
        model=model,
        adversarial_method='gaussian_noise',
        sensitivity_params=sensitivity_params,
        uncertainty_method='bootstrap',
        n_iterations=10,
        noise_level=0.1
    )
    
    # Fit the model
    mr.fit(X, y)
    
    # Test adversarial method
    adversarial_accuracy = mr.adversarial()
    assert isinstance(adversarial_accuracy, float)
    assert 0.0 <= adversarial_accuracy <= 1.0
    
    # Test sensitivity analysis
    sensitivity = mr.sensitivity_analysis()
    assert isinstance(sensitivity, float)
    assert 0.0 <= sensitivity <= 1.0
    
    # Test uncertainty estimation (bootstrap)
    mean_acc, std_acc = mr.uncertainty_estimation()
    assert isinstance(mean_acc, float)
    assert isinstance(std_acc, float)
    assert 0.0 <= mean_acc <= 1.0
    assert 0.0 <= std_acc <= 1.0
    
    # Test run_all_tests
    results = mr.run_all_tests()
    assert isinstance(results, dict)
    assert 'adversarial_accuracy' in results
    assert 'sensitivity' in results
    assert 'uncertainty_mean' in results
    assert 'uncertainty_std' in results
    
    # Test visualize_robustness (ensure it runs without error)
    mr.visualize_robustness()
    
    # Test uncertainty_method 'bayesian'
    mr_bayesian = ModelRobustness(
        model=LogisticRegression(),
        adversarial_method='salt_pepper',
        sensitivity_params={'feature_index': 1, 'delta': 0.05},
        uncertainty_method='bayesian',
        n_iterations=10,
        noise_level=0.1
    )
    mr_bayesian.fit(X, y)
    
    # Mock predict_proba for testing purposes
    if hasattr(mr_bayesian.model_, 'predict_proba'):
        # Assuming the model is fitted
        try:
            mean_entropy, std_entropy = mr_bayesian.uncertainty_estimation()
            assert isinstance(mean_entropy, float)
            assert isinstance(std_entropy, float)
            assert mean_entropy >= 0.0
            assert std_entropy >= 0.0
        except Exception as e:
            pytest.fail(f"Bayesian uncertainty_estimation failed: {e}")
    else:
          mr_bayesian.uncertainty_estimation()
    
    # Test unsupported adversarial_method
    
    with pytest.raises(InvalidParameterError):
        mr_invalid_adv = ModelRobustness(
            model=model,
            adversarial_method='unsupported_method',
            sensitivity_params=sensitivity_params,
            uncertainty_method='bootstrap',
            n_iterations=10,
            noise_level=0.1
        )
        mr_invalid_adv.fit(X, y)
        
        mr_invalid_adv.adversarial()
    
    # Test unsupported uncertainty_method
    with pytest.raises(ValueError):
        mr_invalid_uncertainty = ModelRobustness(
            model=model,
            adversarial_method='gaussian_noise',
            sensitivity_params=sensitivity_params,
            uncertainty_method='unsupported_uncertainty',
            n_iterations=10,
            noise_level=0.1
        )
        mr_invalid_uncertainty.fit(X, y)
        mr_invalid_uncertainty.uncertainty_estimation()
    
    # Test visualize_robustness without running tests
    mr_no_tests = ModelRobustness(
        model=model,
        adversarial_method='gaussian_noise',
        sensitivity_params=sensitivity_params,
        uncertainty_method='bootstrap',
        n_iterations=10,
        noise_level=0.1
    )
    mr_no_tests.fit(X, y)
    with pytest.raises(RuntimeError):
        mr_no_tests.visualize_robustness()

# Test SequentialTesting class
def test_SequentialTesting_accept_h1(sequential_testing_data_accept_h1):
    """
    Test SequentialTesting with data leading to 'Accept H1'.
    """
    p0 = 0.3
    p1 = 0.7
    alpha = 0.05
    beta = 0.2
    st = SequentialTesting(p0=p0, p1=p1, alpha=alpha, beta=beta)
    
    # Run the sequential test
    st.run(sequential_testing_data_accept_h1)
    
    # Check that decision is 'Accept H1'
    assert st.decision_ == 'Accept H1', "Decision should be 'Accept H1'"
    
    # Check that log_likelihood_ratio_ is a list with correct length
    assert isinstance(
        st.log_likelihood_ratio_, list), "log_likelihood_ratio_ should be a list"
    
    # Check that the last log likelihood ratio is >= log_A
    log_A = np.log((1 - beta) / alpha)
    assert st.log_likelihood_ratio_[-1] >= log_A, \
        "Final log-likelihood ratio should exceed upper threshold"
    
    # Test plot method runs without error
    try:
        st.plot()
    except Exception as e:
        pytest.fail(f"SequentialTesting.plot() raised an exception: {e}")

def test_SequentialTesting_accept_h0(sequential_testing_data_accept_h0):
    """
    Test SequentialTesting with data leading to 'Accept H0'.
    """
    p0 = 0.3
    p1 = 0.7
    alpha = 0.05
    beta = 0.2
    st = SequentialTesting(p0=p0, p1=p1, alpha=alpha, beta=beta)
    
    # Run the sequential test
    st.run(sequential_testing_data_accept_h0)
    
    # Check that decision is 'Accept H0'
    assert st.decision_ == 'Accept H0', "Decision should be 'Accept H0'"
    
    # Check that log_likelihood_ratio_ is a list with correct length
    assert hasattr(st, 'log_likelihood_ratio_'), \
        "log_likelihood_ratio_ attribute missing"
    assert isinstance(st.log_likelihood_ratio_, list
                      ), "log_likelihood_ratio_ should be a list"
    
    # Check that the last log likelihood ratio is <= log_B
    log_B = np.log(beta / (1 - alpha))
    assert st.log_likelihood_ratio_[-1] <= log_B,\
        "Final log-likelihood ratio should be below lower threshold"
    
    # Test plot method runs without error
    try:
        st.plot()
    except Exception as e:
        pytest.fail(f"SequentialTesting.plot() raised an exception: {e}")

def test_SequentialTesting_inconclusive(sequential_testing_data_inconclusive):
    """
    Test SequentialTesting with data leading to 'Inconclusive'.
    """
    p0 = 0.3
    p1 = 0.7
    alpha = 0.05
    beta = 0.2
    st = SequentialTesting(p0=p0, p1=p1, alpha=alpha, beta=beta)
    
    # Run the sequential test
    st.run(sequential_testing_data_inconclusive)
    
    # Check that decision is 'Inconclusive'
    assert st.decision_ == 'Inconclusive', "Decision should be 'Inconclusive'"
    
    # Check that sample_size_ is correct
    assert st.sample_size_ == len(sequential_testing_data_inconclusive),\
        "Sample size mismatch"
    
    # Check that log_likelihood_ratio_ is a list with correct length
    assert hasattr(st, 'log_likelihood_ratio_'), \
        "log_likelihood_ratio_ attribute missing"
    assert len(st.log_likelihood_ratio_) == len(
        sequential_testing_data_inconclusive), "log_likelihood_ratio_ length mismatch"
    assert isinstance(st.log_likelihood_ratio_, list
                      ), "log_likelihood_ratio_ should be a list"
    
    # Check that the last log likelihood ratio is between log_B and log_A
    log_A = np.log((1 - beta) / alpha)
    log_B = np.log(beta / (1 - alpha))
    last_s = st.log_likelihood_ratio_[-1]
    assert log_B < last_s < log_A, \
        "Final log-likelihood ratio should be between lower and upper thresholds"
    
    # Test plot method runs without error
    try:
        st.plot()
    except Exception as e:
        pytest.fail(f"SequentialTesting.plot() raised an exception: {e}")

def test_SequentialTesting_invalid_params():
    """
    Test SequentialTesting initialization with invalid parameters.
    """
    # Test invalid p0 (<0)
    with pytest.raises(ValueError):
        SequentialTesting(p0=-0.1, p1=0.7)
    
    # Test invalid p1 (>1)
    with pytest.raises(ValueError):
        SequentialTesting(p0=0.3, p1=1.2)
    
    # Test invalid alpha (>1)
    with pytest.raises(ValueError):
        SequentialTesting(p0=0.3, p1=0.7, alpha=1.1)
    
    # Test invalid beta (<0)
    with pytest.raises(ValueError):
        SequentialTesting(p0=0.3, p1=0.7, beta=-0.2)

# Test DistributionComparison class
def test_DistributionComparison_identical(distribution_comparison_data_identical):
    """
    Test DistributionComparison with identical distributions.
    """
    base = np.e
    dc = DistributionComparison(base=base)
    
    P, Q = distribution_comparison_data_identical
    
    # Run the distribution comparison
    dc.run(P, Q)
    
    # Check that P_ and Q_ are set and normalized
    assert hasattr(dc, 'P_'), "P_ attribute missing"
    assert hasattr(dc, 'Q_'), "Q_ attribute missing"
    assert np.allclose(dc.P_, P), "P_ not correctly set"
    assert np.allclose(dc.Q_, Q), "Q_ not correctly set"
    assert np.isclose(np.sum(dc.P_), 1.0), "P_ is not normalized"
    assert np.isclose(np.sum(dc.Q_), 1.0), "Q_ is not normalized"
    
    # Test Jensen-Shannon divergence is 0
    js_div = dc.jensen_shannon_divergence()
    assert js_div == 0.0,\
        "Jensen-Shannon divergence should be 0 for identical distributions"
    
    # Test Kullback-Leibler divergence is 0
    kl_div = dc.kullback_leibler_divergence()
    assert kl_div == 0.0, \
        "Kullback-Leibler divergence should be 0 for identical distributions"
    
    # Test plot method runs without error
    try:
        dc.plot()
    except Exception as e:
        pytest.fail(f"DistributionComparison.plot() raised an exception: {e}")

def test_DistributionComparison_different(distribution_comparison_data_different):
    """
    Test DistributionComparison with different distributions.
    """
    base = 2  # Using log2
    dc = DistributionComparison(base=base)
    
    P, Q = distribution_comparison_data_different
    
    # Run the distribution comparison
    dc.run(P, Q)
    
    # Check that P_ and Q_ are set and normalized
    assert hasattr(dc, 'P_'), "P_ attribute missing"
    assert hasattr(dc, 'Q_'), "Q_ attribute missing"
    assert np.allclose(dc.P_, np.array(P) / np.sum(P)), "P_ not correctly normalized"
    assert np.allclose(dc.Q_, np.array(Q) / np.sum(Q)), "Q_ not correctly normalized"
    assert np.isclose(np.sum(dc.P_), 1.0), "P_ is not normalized"
    assert np.isclose(np.sum(dc.Q_), 1.0), "Q_ is not normalized"
    
    # Test Jensen-Shannon divergence is positive
    js_div = dc.jensen_shannon_divergence()
    assert js_div > 0.0, \
        "Jensen-Shannon divergence should be positive for different distributions"
    
    # Test Kullback-Leibler divergence is positive
    kl_div = dc.kullback_leibler_divergence()
    assert kl_div > 0.0, \
        "Kullback-Leibler divergence should be positive for different distributions"
    
    # Test plot method runs without error
    try:
        dc.plot()
    except Exception as e:
        pytest.fail(f"DistributionComparison.plot() raised an exception: {e}")

def test_DistributionComparison_non_normalized(
        distribution_comparison_data_non_normalized):
    """
    Test DistributionComparison with non-normalized distributions.
    """
    base = 10
    dc = DistributionComparison(base=base)
    
    P, Q = distribution_comparison_data_non_normalized
    
    # Run the distribution comparison
    with pytest.raises(ValueError): 
        dc.run(P, Q)
    
    # Check that P_ and Q_ are set and normalized
    assert not hasattr(dc, 'P_'), "P_ attribute should missing"
    assert not hasattr(dc, 'Q_'), "Q_ attribute should  missing"
    

def test_DistributionComparison_invalid_params():
    """
    Test DistributionComparison initialization with invalid parameters.
    """
    # Test invalid base (non-positive)
    with pytest.raises(InvalidParameterError):
        dc = DistributionComparison(base=0)
        P = [0.2, 0.3, 0.5]
        Q = [0.2, 0.3, 0.5]
        dc.run(P, Q)
    
    with pytest.raises(InvalidParameterError):
        dc = DistributionComparison(base=-1)
        P = [0.2, 0.3, 0.5]
        Q = [0.2, 0.3, 0.5]
        dc.run(P, Q)

def test_DistributionComparison_invalid_distributions():
    """
    Test DistributionComparison with invalid distributions.
    """
    dc = DistributionComparison()
    
    # Test with P and Q of different lengths
    P = [0.2, 0.3, 0.5]
    Q = [0.1, 0.4]
    with pytest.raises(ValueError):
        dc.run(P, Q)
    
    # Test with negative probabilities
    P = [0.2, -0.3, 1.1]
    Q = [0.1, 0.4, 0.5]
    with pytest.raises(ValueError):
        dc.run(P, Q)
    
    # Test with non-numeric inputs
    P = ['a', 'b', 'c']
    Q = [0.1, 0.4, 0.5]
    with pytest.raises(ValueError):
        dc.run(P, Q)

def test_DistributionComparison_jensen_shannon_divergence_identical(
        distribution_comparison_data_identical):
    """
    Test Jensen-Shannon divergence for identical distributions.
    """
    P, Q = distribution_comparison_data_identical
    dc = DistributionComparison()
    dc.run(P, Q)
    js_div = dc.jensen_shannon_divergence()
    assert js_div == 0.0, \
        "Jensen-Shannon divergence should be 0 for identical distributions"

def test_DistributionComparison_kullback_leibler_divergence_identical(
        distribution_comparison_data_identical):
    """
    Test Kullback-Leibler divergence for identical distributions.
    """
    P, Q = distribution_comparison_data_identical
    dc = DistributionComparison()
    dc.run(P, Q)
    kl_div = dc.kullback_leibler_divergence()
    assert kl_div == 0.0,\
        "Kullback-Leibler divergence should be 0 for identical distributions"


if __name__=='__main__': 
    pytest.main([__file__])

