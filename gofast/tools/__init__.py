"""
Utils sub-package offers several tools for data handling, parameters computation 
models estimation and evalution, and graphs visualization. The extension of the 
mathematical concepts, and the core of program are performed via the modules 
:mod:`~gofast.utils.exmath`. Whereas the machine learning utilities and 
additional functionalities are performed with :mod:`~gofast.utils.mlutils` and 
:mod:`~gofast.utils.funcutils` respectively. 
"""

from .baseutils import (
    audit_data, 
    read_data,
    get_remote_data, 
    array2hdf5, 
    save_or_load, 
    request_data, 
    fancier_downloader,
    speed_rowwise_process, 
    )
from .mathex import ( 
    interpolate1d, 
    interpolate2d,
    scale_y, 
    get_bearing, 
    moving_average, 
    linkage_matrix, 
    get_distance,
    smooth1d, 
    smoothing, 
    quality_control, 
    adaptive_moving_average, 
    savgol_filter, 
    )
from .funcutils import ( 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier, 
    remove_outliers,
    normalizer, 
    cleaner, 
    save_job, 
    random_selector, 
    interpolate_grid, 
    pair_data, 
    random_sampling, 
    replace_data, 
    store_or_write_hdf5, 
    inspect_data, 
    )

from .mlutils import ( 
    evaluate_model,
    select_features, 
    get_global_score,  
    get_correlated_features, 
    find_features_in, 
    codify_variables, 
    categorize_target, 
    resampling, 
    bin_counting, 
    labels_validator, 
    projection_validator, 
    rename_labels_in , 
    soft_imputer, 
    soft_scaler, 
    select_feature_importances, 
    make_pipe, 
    bi_selector, 
    get_target, 
    export_target,  
    stats_from_prediction, 
    fetch_tgz,  
    fetch_model, 
    load_csv, 
    split_train_test_by_id, 
    split_train_test, 
    discretize_categories, 
    stratify_categories, 
    serialize_data, 
    load_dumped_data, 
    naive_data_split, 
    laplace_smoothing, 
    features_in, 
    
    ) 
__all__=[
    'audit_data', 
    'inspect_data', 
    'read_data',
    'array2hdf5', 
    'save_or_load', 
    'request_data', 
    'get_remote_data', 
    'fancier_downloader',
    'savgol_filter', 
    'interpolate1d', 
    'interpolate2d',
    'scale_y', 
    'select_features', 
    'get_global_score',  
    'split_train_test', 
    'speed_rowwise_process', 
    'get_correlated_features', 
    'find_features_in',
    'codify_variables', 
    'evaluate_model',
    'moving_average', 
    'linkage_matrix',
    'reshape', 
    'to_numeric_dtypes' , 
    'smart_label_classifier', 
    'evaluate_model',
    'select_features', 
    'get_global_score', 
    'split_train_test', 
    'find_features_in', 
    'categorize_target', 
    'resampling', 
    'bin_counting', 
    'labels_validator', 
    'projection_validator', 
    'rename_labels_in' , 
    'soft_imputer', 
    'soft_scaler', 
    'select_feature_importances', 
    'make_pipe', 
    'bi_selector', 
    'get_target', 
    'export_target',  
    'stats_from_prediction', 
    'fetch_tgz', 
    'fetch_model', 
    'load_csv', 
    'split_train_test_by_id', 
    'split_train_test', 
    'discretize_categories', 
    'stratify_categories', 
    'serialize_data', 
    'load_dumped_data', 
    'naive_data_split', 
    'soft_imputer', 
    'soft_scaler', 
    'make_pipe',
    'classify_k',
    'label_importance', 
    'remove_outliers', 
    'normalizer',
    'get_distance',
    'get_bearing', 
    'quality_control', 
    'cleaner', 
    'save_job', 
    'random_selector', 
    'interpolate_grid',
    'smooth1d', 
    'smoothing', 
    'pair_data', 
    'random_sampling', 
    'plot_voronoi', 
    'plot_roc_curves', 
    'replace_data', 
    'store_or_write_hdf5', 
    "resampling", 
    "bin_counting",
    "adaptive_moving_average", 
    "butterworth_filter",
    "plot_l_curve", 
    "laplace_smoothing", 
    "features_in"
    
    ]



