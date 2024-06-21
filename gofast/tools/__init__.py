"""
The Tools sub-package offers a variety of utilities for data handling, 
parameter computation, model estimation, and evaluation. It extends
mathematical concepts through the module :mod:`~gofast.tools.mathex`. 
Additionally, machine learning utilities and supplementary functionalities 
are facilitated by :mod:`~gofast.tools.mlutils` and 
:mod:`~gofast.tools.coreutils`, respectively.
 
"""

import importlib

MODULE_MAPPING = {
    'baseutils': [
         'array2hdf5',
         'binning_statistic',
         'categorize_target',
         'category_count',
         'denormalizer',
         'extract_target',
         'fancier_downloader',
         'fillNaN',
         'get_target',
         'interpolate_data',
         'interpolate_grid',
         'labels_validator',
         'normalizer',
         'remove_outliers',
         'remove_target_from_array',
         'rename_labels_in',
         'save_or_load',
         'scale_y',
         'select_features',
         'select_features',
         'smooth1d',
         'smoothing',
         'soft_bin_stat',
         'speed_rowwise_process'
    ],
    'coreutils': [
        'cleaner', 
        'extract_coordinates',
        'features_in', 
        'find_features_in',
        'parallelize_jobs',
        'pair_data',
        'projection_validator',
        'random_sampling', 
        'random_selector',
        'replace_data', 
        'resample_data', 
        'save_job',
        'smart_label_classifier', 
        'split_train_test',
        'split_train_test_by_id',
        'store_or_write_hdf5',
    ],
    'mathex': [
        'adaptive_moving_average', 
        'adjust_for_control_vars',
        'calculate_adjusted_lr', 
        'calculate_binary_iv', 
        'calculate_optimal_bins', 
        'calculate_residuals', 
        'compute_balance_accuracy',
        'compute_effort_yield',
        'compute_errors', 
        'compute_p_values',
        'compute_sunburst_data',
        'compute_cost_based_threshold', 
        'compute_sensitivity_specificity', 
        'compute_youdens_index',
        'cubic_regression', 
        'exponential_regression', 
        'get_bearing', 
        'get_distance',
        'infer_sankey_columns', 
        'label_importance',
        'linear_regression',
        'linkage_matrix', 
        'logarithmic_regression',
        'make_mxs', 
        'minmax_scaler',
        'moving_average',
        'normalize',
        'optimized_spearmanr', 
        'quadratic_regression', 
        'rank_data', 
        'savgol_filter', 
        'sinusoidal_regression',
        'standard_scaler', 
        'step_regression',
        'weighted_spearman_rank', 
    ],
    'mlutils': [
        'bi_selector', 
        'bin_counting', 
        'build_data_preprocessor',
        'deserialize_data',
        'display_feature_contributions', 
        'discretize_categories', 
        'evaluate_model',
        'fetch_model', 
        'fetch_tgz', 
        'get_correlated_features',
        'get_feature_contributions',
        'get_global_score',
        'handle_imbalance',
        'laplace_smoothing', 
        'laplace_smoothing_categorical',
        'laplace_smoothing_word', 
        'load_csv', 
        'load_model',
        'make_pipe', 
        'one_click_preprocess', 
        'resampling',
        'save_dataframes', 
        'select_feature_importances', 
        'serialize_data', 
        'smart_split',
        'soft_encoder', 
        'soft_data_split', 
        'soft_imputer', 
        'soft_scaler', 
        'stats_from_prediction',
        'stratify_categories'
    ]
}

# Lazy loader function
def __getattr__(name):
    # Loop through the module mapping to find the function
    for submodule, functions in MODULE_MAPPING.items():
        if name in functions:
            # Import the submodule if the function is found
            module = importlib.import_module('.' + submodule, 'gofast.tools')
            func = getattr(module, name)
            # Cache the imported function back into globals to speed up future imports
            globals()[name] = func
            return func
    raise AttributeError(f"No function named {name} found in gofast.tools.")

# Dynamically populate __all__ to make imports 
# like 'from gofast.tools import *' work correctly
__all__ = sum(MODULE_MAPPING.values(), [])

# Adjusted __dir__ to assist with autocomplete and dir() calls
def __dir__():
    return __all__






