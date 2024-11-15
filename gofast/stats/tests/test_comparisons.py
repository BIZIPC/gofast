# -*- coding: utf-8 -*-
# test_model_comparisons.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from gofast.stats.comparisons import (
    friedman_test,
    friedman_test_in,
    nemenyi_posthoc,
    wilcoxon_test,
    normalize_preference,
    plot_model_rankings,
    posthoc_tests,
    calculate_pairwise_mean_differences,
    posthoc_analysis,
    get_p_adj_for_groups,
    compute_model_ranks
)
from gofast.api.formatter import DataFrameFormatter
from gofast.tools.depsutils import ensure_module_installed 

SK_POSTHOC_AVAILABLE =False 
try : 
    import scikit_posthocs
except: 
    SK_POSTHOC_AVAILABLE= ensure_module_installed(
        "scikit_posthocs", dist_name ="scikit-posthocs", auto_install=True )
    
    if SK_POSTHOC_AVAILABLE: 
        import scikit_posthocs # noqa 

# ================================
# Fixtures for Test Data
# ================================
@pytest.fixture
def valid_performance_data():
    return pd.DataFrame({
        'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
        'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
        'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    })


@pytest.fixture
def empty_performance_data():
    return pd.DataFrame()


@pytest.fixture
def nan_performance_data():
    return pd.DataFrame({
        'Model_A': [0.8, np.nan, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, np.nan]
    })


@pytest.fixture
def mock_posthoc_result_nemenyi():
    return pd.DataFrame({
        'Model_A': [1.0, 0.05, 0.01],
        'Model_B': [0.05, 1.0, 0.20],
        'Model_C': [0.01, 0.20, 1.0]
    }, index=['Model_A', 'Model_B', 'Model_C'])


@pytest.fixture
def mock_posthoc_result_tukey():
    return pd.DataFrame({
        'group1': ['Model_A', 'Model_A', 'Model_B'],
        'group2': ['Model_B', 'Model_C', 'Model_C'],
        'meandiff': [0.01, 0.00, -0.02],
        'p-adj': [0.50, 1.00, 0.75],
        'reject': [False, False, False]
    })


# ================================
# Test Classes
# ================================

class TestPerformFriedmanTest:
    def test_valid_data(self, valid_performance_data):
        result = friedman_test(valid_performance_data)
        assert hasattr(result, 'dfs'), "Result should have 'df' attribute"
        friedman_result = result.friedman_result
        for column in ['Statistic', 'Degrees of Freedom', 'p-value', 'Significant Difference']:
            assert column in friedman_result.columns, f"'{column}' should be in Friedman result columns"

    def test_empty_dataframe(self, empty_performance_data):
        with pytest.raises(ValueError, match="model_performance_data DataFrame cannot be empty."):
            friedman_test(empty_performance_data)

    def test_dataframe_with_nans(self, nan_performance_data):
        with pytest.raises(ValueError, match="model_performance_data DataFrame contains NaN values. Please clean your data."):
            friedman_test(nan_performance_data)

    def test_invalid_alpha(self, valid_performance_data):
        with pytest.raises(ValueError, match='alpha must be between 0 and 1, got: 1.5'):
            friedman_test(valid_performance_data, alpha=1.5)

    def test_invalid_score_preference(self, valid_performance_data):
        with pytest.raises(ValueError, match="Invalid score_preference"):
            friedman_test(valid_performance_data, score_preference='invalid preference')

    def test_with_posthoc(self, valid_performance_data):

        result = friedman_test(valid_performance_data, perform_posthoc=True)
        assert 'Friedman Test Results' in result.titles,\
                "Result should contain 'Friedman Test Results' title"

class TestNormalizePreference:
    @pytest.mark.parametrize("input_pref,expected", [
        ("higher_is_better", "higher is better"),
        ("Higher Is Better", "higher is better"),
        ("LOWER_is_better", "lower is better"),
        ("lower is better", "lower is better"),
        ("HIGHer", "higher is better"),
        ("lowER", "lower is better"),
    ])
    def test_normalize_preference_valid(self, input_pref, expected):
        assert normalize_preference(input_pref) == expected

    @pytest.mark.parametrize("input_pref", [
        "better",
        "highest",
        "lowest",
        "",
        "up",
        "down"
    ])
    def test_normalize_preference_invalid(self, input_pref):
        with pytest.raises(ValueError, match="Invalid score_preference"):
            normalize_preference(input_pref)


class TestPerformNemenyiPosthocTest:
    def test_valid_data(self, valid_performance_data):
        result = nemenyi_posthoc(valid_performance_data)
        assert 'p_values' in result.keywords
        assert 'significant_differences' in result.keywords
        assert 'average_ranks' in result.keywords

    def test_invalid_alpha(self, valid_performance_data):
        with pytest.raises(ValueError, match="significance_level must be between 0 and 1, got: 1.2"):
            nemenyi_posthoc(valid_performance_data, significance_level=1.2)

class TestPerformWilcoxonTest:
    def test_valid_data(self, valid_performance_data):
        result = wilcoxon_test(valid_performance_data)
        assert hasattr(result, 'df'), "Result should have 'df' attribute"
        wilcoxon_results = result.df
        assert wilcoxon_results.shape == (3, 3), "Wilcoxon results should be a 3x3 matrix"
        np.testing.assert_array_equal(np.diag(wilcoxon_results), np.nan, "Diagonal should be NaN")

    def test_mask_non_significant_true(self, valid_performance_data):

        result = wilcoxon_test(valid_performance_data, mask_non_significant=True)
        wilcoxon_results = result.df
        assert not (wilcoxon_results == np.nan).all().all(), "No significant values detected less than 0.05"


class TestPerformFriedmanTest2:
    def test_valid_dataframe_input(self, valid_performance_data):
        result = friedman_test_in(valid_performance_data)
        assert isinstance(result, DataFrameFormatter), "Result should be a DataFrameFormatter instance"
        result_df = result.df
        expected_columns = ["Friedman Test Statistic", "p-value", "Significant Difference"]
        assert list(result_df.columns) == expected_columns, "Result columns mismatch"
        assert result_df.shape[0] == 1, "There should be exactly one row in the result"
        assert isinstance(result_df.at[0, "Friedman Test Statistic"], float), "Statistic should be float"
        assert isinstance(result_df.at[0, "p-value"], float), "p-value should be float"

    def test_valid_dict_input(self, valid_performance_data):
        result = friedman_test_in(valid_performance_data)
        assert isinstance(result, DataFrameFormatter), "Result should be a DataFrameFormatter instance"
        result_df = result.df
        expected_columns = ["Friedman Test Statistic", "p-value", "Significant Difference"]
        assert list(result_df.columns) == expected_columns, "Result columns mismatch"
        assert result_df.shape[0] == 1, "There should be exactly one row in the result"
        assert isinstance(result_df.at[0, "Friedman Test Statistic"], float), "Statistic should be float"
        assert isinstance(result_df.at[0, "p-value"], float), "p-value should be float"
  
    def test_empty_dataframe_input(self, empty_performance_data):
        with pytest.raises(ValueError, match='At least 3 sets of samples must be given for Friedman test, got 0.'):
            friedman_test_in(empty_performance_data)

    def test_dataframe_with_nans(self, nan_performance_data):
        with pytest.raises(ValueError, match="NaN values detected in the data. Set `nan_policy='omit'` to drop them."):
            friedman_test_in(nan_performance_data)

    def test_friedmanchisquare_called_correctly(self, valid_performance_data):
        result = friedman_test_in(valid_performance_data)
        result_df = result.df
        assert result_df.at[0, "Friedman Test Statistic"] == 0.4
        assert result_df.at[0, "p-value"] == 0.8187
        assert result_df.at[0, "Significant Difference"] == False

    def test_minimum_number_of_models(self):
        data = pd.DataFrame({
            'Model_A': [0.8, 0.82, 0.78],
            'Model_B': [0.79, 0.84, 0.76]
        })
        # Assuming the function works with 2 models
        with pytest.raises(ValueError, match="At least 3 sets of samples must be given for Friedman test, got 2."):
            friedman_test_in(data)
            
    def test_tied_ranks(self):
        data = pd.DataFrame({
            'Model_A': [0.8, 0.80, 0.80],
            'Model_B': [0.8, 0.80, 0.80],
            'Model_C': [0.8, 0.80, 0.80]
        })
        result = friedman_test_in(data)
        result_df = result.df
        assert result_df.at[0, "Significant Difference"]==False,( 
            "Should not be a significant difference, result_df.at[0,"
            " RuntimeWarning: invalid value encountered in scalar divide "
            )


class TestGetPAdjForGroups:
    def test_valid_input(self):
        data = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C'],
            'p-adj': [0.50, 1.00, 0.75]
        })
        result = get_p_adj_for_groups(data, 'Model_A', 'Model_B')
        assert result == 0.50, f"Expected 0.50, but got {result}"
        result = get_p_adj_for_groups(data, 'Model_B', 'Model_C')
        assert result == 0.75, f"Expected 0.75, but got {result}"

    def test_invalid_pair(self):
        data = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C'],
            'p-adj': [0.50, 1.00, 0.75]
        })
        with pytest.raises(ValueError, match="No matching pair found for Model_A and Model_D."):
            get_p_adj_for_groups(data, 'Model_A', 'Model_D')

    def test_reverse_order(self):
        data = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C'],
            'p-adj': [0.50, 1.00, 0.75]
        })
        result = get_p_adj_for_groups(data, 'Model_B', 'Model_A')
        assert result == 0.50, f"Expected 0.50, but got {result}"

    def test_multiple_entries(self):
        data = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C', 'Model_C', 'Model_A'],
            'p-adj': [0.50, 1.00, 0.75, 0.60, 0.55]
        })

        result = get_p_adj_for_groups(data, 'Model_A', 'Model_B')
        assert result == 0.50, f"Expected 0.50, but got {result}"

    def test_no_side_effects(self):
        data = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C'],
            'p-adj': [0.50, 1.00, 0.75]
        })
        initial_data = data.copy()
        result = get_p_adj_for_groups(data, 'Model_A', 'Model_B')
        assert result == 0.50, f"Expected 0.50, but got {result}"
        pd.testing.assert_frame_equal(data, initial_data), "Original data should remain unchanged"


class TestPlotModelRankings:
    def test_valid_dataframe(self, valid_performance_data):
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_model_rankings(valid_performance_data)
            mock_show.assert_called_once()

    def test_empty_dataframe(self, empty_performance_data):
        with pytest.raises(ValueError, match="The model_performance_data DataFrame is empty."):
            plot_model_rankings(empty_performance_data)

    def test_dataframe_with_nans(self, nan_performance_data):
        plot_model_rankings(nan_performance_data)

    def test_fig_size(self, valid_performance_data):
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_model_rankings(valid_performance_data, fig_size=(12, 6))
            mock_show.assert_called_once()

    def test_score_preference_lower_is_better(self, valid_performance_data):
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_model_rankings(valid_performance_data, score_preference="lower is better")
            mock_show.assert_called_once()

    def test_invalid_score_preference(self, valid_performance_data):
        with pytest.raises(ValueError, match="Invalid score_preference. Choose 'higher is better' or 'lower is better'."):
            plot_model_rankings(valid_performance_data, score_preference="invalid")


@pytest.mark.skipif(not SK_POSTHOC_AVAILABLE, "scikit-posthocs not installed.")
class TestPerformPosthocTest:
    @pytest.mark.skip( "is an instance of local TurkeyTest;"
                      " gofast.stats.model_comparisons.posthoc_tests.<locals>.TurkeyTest ")
    def test_tukey_valid(self, valid_performance_data, mock_posthoc_result_tukey):
            result = posthoc_tests(valid_performance_data, test_method='tukey')
            assert isinstance(result, type), "is an instance of local TurkeyTest"


    def test_nemenyi_valid(self, valid_performance_data):
        result = posthoc_tests(valid_performance_data, test_method='nemenyi')
     
        assert isinstance(result.df, pd.DataFrame), "Result should be a DataFrame"

    def test_invalid_method(self, valid_performance_data):
        with pytest.raises(ValueError, match="Invalid test_method. Supported methods are 'tukey' and 'nemenyi'."):
            posthoc_tests(valid_performance_data, test_method='invalid_method')


    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match='2 or more groups required for multiple comparisons'):
            posthoc_tests([1, 2, 3])

    def test_friedman_test_needed(self, valid_performance_data):
            result = posthoc_tests(valid_performance_data, test_method='nemenyi')
            assert isinstance(result.df, pd.DataFrame), "Result should be a DataFrame"

class TestCalculatePairwiseMeanDifferences:
    def test_valid_input(self, mock_data):
        result = calculate_pairwise_mean_differences(valid_performance_data)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert set(result.columns) == set(valid_performance_data.columns), "Columns should match input models"

    def test_return_group_true(self, mock_data):
        result = calculate_pairwise_mean_differences(valid_performance_data, return_group=True)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert 'group1' in result.columns, "Result should contain 'group1' column"

@pytest.mark.skipif(not SK_POSTHOC_AVAILABLE, "scikit-posthocs not installed.")
class TestPerformPosthocAnalysis:
    def test_posthoc_analysis_tukey(self, valid_performance_data, mock_posthoc_result_tukey):
        with patch('statsmodels.stats.multicomp.pairwise_tukeyhsd') as mock_tukey:
            mock_tukey.return_value = mock_posthoc_result_tukey
            result = posthoc_analysis(valid_performance_data, test_method='tukey')
            mock_tukey.assert_called_once()
            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    def test_posthoc_analysis_nemenyi(self, valid_performance_data, mock_posthoc_result_nemenyi):
        with patch('scikit_posthocs.posthoc_nemenyi_friedman') as mock_nemenyi:
            mock_nemenyi.return_value = mock_posthoc_result_nemenyi
            result = posthoc_analysis(valid_performance_data, test_method='nemenyi')
            mock_nemenyi.assert_called_once_with(valid_performance_data)
            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    def test_posthoc_analysis_invalid_method(self, mock_data):
        with pytest.raises(ValueError, match="Invalid test_method. Supported methods are 'tukey' and 'nemenyi'."):
            posthoc_analysis(mock_data, test_method='invalid_method')


class TestComputeModelRanks:
    def test_valid_input(self, valid_performance_data):
        result = compute_model_ranks(valid_performance_data)
        assert isinstance(result, DataFrameFormatter), "Result should be a DataFrameFormatter instance"
        ranks_df = result.df
        assert isinstance(ranks_df, pd.DataFrame), "Ranks should be a DataFrame"
        assert set(ranks_df.columns) == set(valid_performance_data.columns), "Columns should match input models"
        assert ranks_df.shape == valid_performance_data.shape, "Ranks shape should match input data"

    def test_invalid_input_type(self):
        with pytest.raises(ValueError, match='Unable to convert to DataFrame: DataFrame constructor not properly called!'):
            compute_model_ranks("invalid_input")

    def test_dataframe_with_nans(self, nan_performance_data):
        with pytest.raises(ValueError, match="NaN values detected in the data. Set `nan_policy='omit'` to drop them."):
            compute_model_ranks(nan_performance_data)


# ================================
# Additional Test Functions
# ================================
@pytest.mark.skipif(not SK_POSTHOC_AVAILABLE, "scikit-posthocs not installed.")
def test_calculate_pairwise_mean_differences_return_group(mock_data):
    """Test calculate_pairwise_mean_differences with return_group=True."""
    with patch('gofast.stats.model_comparisons.calculate_pairwise_mean_differences') as mock_calc:
        mock_calc.return_value = pd.DataFrame({
            'group1': ['Model_A', 'Model_A', 'Model_B'],
            'group2': ['Model_B', 'Model_C', 'Model_C'],
            'meandiff': [0.01, 0.00, -0.02]
        })
        result = calculate_pairwise_mean_differences(mock_data, return_group=True)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert 'group1' in result.columns, "Result should contain 'group1' column"
        
@pytest.mark.skipif(not SK_POSTHOC_AVAILABLE, "scikit-posthocs not installed.")
def test_posthoc_analysis_tukey(mock_data, mock_posthoc_result_tukey):
    """Test posthoc_analysis function with Tukey."""
    with patch('statsmodels.stats.multicomp.pairwise_tukeyhsd') as mock_tukey:
        mock_tukey.return_value = mock_posthoc_result_tukey
        result = posthoc_analysis(mock_data, test_method='tukey')
        mock_tukey.assert_called_once()
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        
@pytest.mark.skipif(not SK_POSTHOC_AVAILABLE, "scikit-posthocs not installed.")
def test_posthoc_analysis_nemenyi(mock_data, mock_posthoc_result_nemenyi):
    """Test posthoc_analysis function with Nemenyi."""
    with patch('scikit_posthocs.posthoc_nemenyi_friedman') as mock_nemenyi:
        mock_nemenyi.return_value = mock_posthoc_result_nemenyi
        result = posthoc_analysis(mock_data, test_method='nemenyi')
        mock_nemenyi.assert_called_once_with(mock_data)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"


# ================================
# Main Execution
# ================================

if __name__ == "__main__":
    pytest.main([__file__])
