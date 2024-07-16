# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
GOFast: Accelerate Your Machine Learning Workflow
"""

import os
import sys
import logging
import warnings

# Only modify sys.path if necessary, avoid inserting unnecessary paths
package_dir = os.path.dirname(__file__)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Configure logging with lazy loading
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Environment setup for compatibility
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# Suppress FutureWarnings globally, consider doing it locally if possible
warnings.simplefilter(action='ignore', category=FutureWarning)

if not __package__:
    __package__ = 'gofast'


# Dynamic import to reduce initial load time
def lazy_import(module_name, global_name=None):
    if global_name is None:
        global_name = module_name
    import importlib
    globals()[global_name] = importlib.import_module(module_name)

# Generate version
try:
    from ._version import version
    __version__ = version.split('.dev')[0]
except ImportError:
    __version__ = "0.1.0"

# Check and import main dependencies lazily
_main_dependencies = {
    "numpy": None,
    "scipy": None,
    "scikit-learn": "sklearn",
    "matplotlib": None,
    "pandas": None,
    "seaborn": None,
    "tqdm":None,
    "statsmodels": None, 
}

_missing_dependencies = []

for module, import_name in _main_dependencies.items():
    try:
        lazy_import(module if not import_name else import_name, module)
    except ImportError as e:
        _missing_dependencies.append(f"{module}: {e}")

if _missing_dependencies:
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(
        _missing_dependencies))


# Set a default LOG_PATH if it's not already set
os.environ.setdefault('LOG_PATH', os.path.join(package_dir, 'gflogs'))

# Import the logging setup function from _gofastlog.py
from ._gofastlog import gofastlog

# Define the path to the _gflog.yml file
config_file_path = os.path.join(package_dir, '_gflog.yml')

# Set up logging with the path to the configuration file
gofastlog.load_configuration(config_file_path)


# Public API
# __all__ = ['show_versions']

# Reset warnings to default
warnings.simplefilter(action='default', category=FutureWarning)

__doc__= """\
Accelerate Your Machine Learning Workflow
==========================================

:code:`gofast` is a comprehensive machine learning toolbox designed to 
streamline and accelerate every step of your data science workflow. 
Its objectives are: 
    
* `Enhance Productivity`: Reduce the time spent on routine data tasks.
* `User-Friendly`: Whether you're a beginner or an expert, gofast is designed 
  to be intuitive and accessible for all users in the machine learning community.
* `Community-Driven`: welcoming contributions and suggestions from the community
  to continuously improve and evolve.

`GoFast`_ focused on delivering high-speed tools and utilities that 
assist users in swiftly navigating through the critical stages of data 
analysis, processing, and modeling.

.. _GoFast: https://github.com/WEgeophysics/gofast

"""
# from .baseutils import ( 
#     categorize_target,
#     select_features, 
#     extract_target,
#     array2hdf5,
#     fancier_downloader,
#     labels_validator,
#     rename_labels_in,
#     save_or_load,
#     speed_rowwise_process, 
#  )
# from .coreutils import (
#     cleaner, 
#     denormalize,
#     extract_coordinates,
#     features_in,
#     find_features_in,
#     interpolate_grid,
#     normalizer,
#     parallelize_jobs,
#     pair_data,
#     projection_validator,
#     random_sampling,
#     random_selector,
#     remove_outliers,
#     replace_data,
#     resample_data,
#     save_job,
#     smart_label_classifier,
#     split_train_test,
#     split_train_test_by_id,
#     store_or_write_hdf5,
#     to_numeric_dtypes,
# )
# from .dataops import (
#     analyze_data_corr, 
#     apply_bow_vectorization,
#     apply_tfidf_vectorization,
#     apply_word_embeddings,
#     assess_outlier_impact,
#     augment_data,
#     audit_data,
#     base_transform, 
#     boxcox_transformation,
#     check_missing_data,
#     convert_date_features,
#     correlation_ops, 
#     data_assistant, 
#     drop_correlated_features, 
#     enrich_data_spectrum,
#     fetch_remote_data,
#     format_long_column_names,
#     handle_categorical_features,
#     handle_datasets_with_hdfstore,
#     handle_duplicates, 
#     handle_unique_identifiers, 
#     handle_missing_data,
#     handle_outliers_in,
#     handle_skew, 
#     inspect_data,
#     prepare_data, 
#     read_data,
#     request_data,
#     sanitize,
#     scale_data,
#     simple_extractive_summary,
#     store_or_retrieve_data,
#     summarize_text_columns,
#     transform_dates,
#     verify_data_integrity,
# )

# from .mathex import (
#     adaptive_moving_average,
#     adjust_for_control_vars, 
#     binning_statistic,
#     calculate_binary_iv, 
#     calculate_optimal_bins, 
#     calculate_residuals,
#     category_count,
#     compute_effort_yield,
#     compute_sunburst_data,
#     cubic_regression,
#     exponential_regression,
#     get_bearing,
#     get_distance,
#     infer_sankey_columns, 
#     interpolate1d,
#     interpolate2d,
#     label_importance,
#     linear_regression,
#     linkage_matrix,
#     logarithmic_regression,
#     make_mxs,
#     minmax_scaler,
#     moving_average,
#     normalize,
#     optimized_spearmanr, 
#     quality_control,
#     quadratic_regression,
#     rank_data, 
#     savgol_filter,
#     scale_y,
#     sinusoidal_regression,
#     smooth1d,
#     smoothing,
#     soft_bin_stat,
#     standard_scaler,
#     step_regression,
#     weighted_spearman_rank, 
# )

# from .mlutils import (
#     bi_selector,
#     bin_counting,
#     build_data_preprocessor,
#     codify_variables,
#     deserialize_data,
#     discretize_categories,
#     evaluate_model,
#     fetch_model,
#     fetch_tgz,
#     get_correlated_features,
#     get_global_score,
#     get_target,
#     handle_imbalance,
#     laplace_smoothing,
#     laplace_smoothing_categorical,
#     laplace_smoothing_word,
#     load_csv,
#     load_model,
#     make_pipe,
#     one_click_preprocess, 
#     resampling,
#     save_dataframes,
#     select_feature_importances,
#     serialize_data,
#     smart_split,
#     soft_data_split,
#     soft_imputer,
#     soft_scaler,
#     stats_from_prediction,
#     stratify_categories,
# )

# __all__=[
#      'adaptive_moving_average',
#      'adjust_for_control_vars',
#      'analyze_data_corr',
#      'apply_bow_vectorization',
#      'apply_tfidf_vectorization',
#      'apply_word_embeddings',
#      'array2hdf5',
#      'assess_outlier_impact',
#      'audit_data',
#      'augment_data',
#      'base_transform', 
#      'bi_selector',
#      'bin_counting',
#      'bin_counting',
#      'binning_statistic',
#      'boxcox_transformation',
#      'build_data_preprocessor',
#      'butterworth_filter',
#      'calculate_binary_iv', 
#      'calculate_optimal_bins', 
#      'calculate_residuals',
#      'categorize_target',
#      'category_count',
#      'check_missing_data',
#      'cleaner',
#      'codify_variables',
#      'compute_effort_yield',
#      'compute_sunburst_data',
#      'convert_date_features',
#      'correlation_ops', 
#      'cubic_regression',
#      'data_assistant',
#      'denormalize',
#      'deserialize_data',
#      'discretize_categories',
#      'drop_correlated_features', 
#      'enrich_data_spectrum',
#      'evaluate_model',
#      'evaluate_model',
#      'exponential_regression',
#      'extract_coordinates',
#      'extract_target',
#      'fancier_downloader',
#      'features_in',
#      'fetch_model',
#      'fetch_remote_data',
#      'fetch_tgz',
#      'find_features_in',
#      'format_long_column_names',
#      'get_bearing',
#      'get_correlated_features',
#      'get_distance',
#      'get_global_score',
#      'get_global_score',
#      'get_target',
#      'handle_categorical_features',
#      'handle_datasets_with_hdfstore',
#      'handle_duplicates', 
#      'handle_imbalance',
#      'handle_missing_data',
#      'handle_outliers_in',
#      'handle_unique_identifiers', 
#      'handle_skew', 
#      'infer_sankey_columns',
#      'inspect_data',
#      'interpolate1d',
#      'interpolate2d',
#      'interpolate_grid',
#      'label_importance',
#      'labels_validator',
#      'laplace_smoothing',
#      'laplace_smoothing_categorical',
#      'laplace_smoothing_word',
#      'linear_regression',
#      'linkage_matrix',
#      'load_csv',
#      'load_model',
#      'logarithmic_regression',
#      'make_mxs',
#      'make_pipe',
#      'make_pipe',
#      'minmax_scaler',
#      'moving_average',
#      'normalize',
#      'normalizer',
#      'one_click_preprocess',
#      'optimized_spearmanr', 
#      'pair_data',
#      'parallelize_jobs',
#      'prepare_data', 
#      'projection_validator',
#      'quadratic_regression',
#      'quality_control',
#      'random_sampling',
#      'random_selector',
#      'rank_data', 
#      'read_data',
#      'remove_outliers',
#      'rename_labels_in',
#      'replace_data',
#      'request_data',
#      'resample_data',
#      'resampling',
#      'resampling',
#      'reshape',
#      'sanitize',
#      'save_dataframes',
#      'save_job',
#      'save_or_load',
#      'savgol_filter',
#      'scale_data',
#      'scale_y',
#      'select_feature_importances',
#      'select_features',
#      'select_features',
#      'serialize_data',
#      'simple_extractive_summary',
#      'sinusoidal_regression',
#      'smart_label_classifier',
#      'smart_split',
#      'smooth1d',
#      'smoothing',
#      'soft_bin_stat',
#      'soft_data_split',
#      'soft_imputer',
#      'soft_imputer',
#      'soft_scaler',
#      'soft_scaler',
#      'speed_rowwise_process',
#      'split_train_test',
#      'split_train_test',
#      'split_train_test',
#      'split_train_test_by_id',
#      'standard_scaler',
#      'stats_from_prediction',
#      'step_regression',
#      'store_or_retrieve_data',
#      'store_or_write_hdf5',
#      'stratify_categories',
#      'summarize_text_columns',
#      'to_numeric_dtypes',
#      'transform_dates',
#      'verify_data_integrity', 
#      'weighted_spearman_rank', 
#  ]