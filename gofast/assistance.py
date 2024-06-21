# -*- coding: utf-8 -*-

import importlib
import pkgutil
import warnings
import gofast.tools 
import gofast.dataops
from gofast.api.summary import ReportFactory, assemble_reports
from gofast.api.util import remove_extra_spaces, get_table_size 
TW = get_table_size() 

__all__=["assist_me", "gofast_explorer"]

TASK_MAPPING = {
    'data_preprocessing': [
        'cleaner', 
        'denormalize', 
        'normalizer', 
        'binning_statistic', 
        'category_count', 
        'handle_missing_data',
        'handle_outliers_in', 
        'handle_skew',
        'handle_duplicates', 
        'scale_data',
        'remove_outliers', 
        'replace_data',
        'handle_categorical_features',
        'correlation_ops', 
        'drop_correlated_features',
        'interpolate_grid',
        'resample_data'
    ],
    'data_transformation': [
        'boxcox_transformation', 
        'apply_bow_vectorization', 
        'apply_tfidf_vectorization', 
        'apply_word_embeddings',
        'convert_date_features', 
        'transform_dates'
    ],
    'data_management': [
        'array2hdf5',
        'store_or_retrieve_data',
        'store_or_write_hdf5',
        'handle_datasets_with_hdfstore',
        'save_or_load', 
        'fetch_remote_data',
        'read_data'
    ],
    'feature_engineering': [
        'select_features',
        'extract_target', 
        'find_features_in', 
        'features_in', 
        'get_target', 
        'pair_data',
        'select_feature_importances', 
        'bi_selector',
        'bin_counting'
    ],
    'model_preparation': [
        'split_train_test', 
        'split_train_test_by_id',
        'build_data_preprocessor',
        'make_pipe', 
        'one_click_preprocess',
        'soft_data_split',
        'stratify_categories',
        'smart_split',
        'soft_encoder', 
        'soft_scaler', 
        'soft_imputer', 
        'handle_imbalance',
        'resampling',
    ],
    'data_analysis': [
        'analyze_data_corr', 
        'check_missing_data',
        'audit_data', 
        'inspect_data', 
        'quality_control',
        'verify_data_integrity',
        'correlation_ops'
    ],
    'statistics_and_math': [
        'adaptive_moving_average',
        'calculate_adjusted_lr',
        'calculate_optimal_bins',
        'calculate_binary_iv',
        'compute_effort_yield', 
        'compute_sensitivity_specificity', 
        'compute_cost_based_threshold',
        'compute_youdens_index', 
        'compute_errors',
        'linear_regression',
        'quadratic_regression',
        'cubic_regression', 
        'logarithmic_regression',
        'exponential_regression',
        'interpolate1d', 
        'interpolate2d', 
        'rank_data',
        'optimized_spearmanr', 
    ],
    'model_evaluation_and_analysis': [
        'evaluate_model', 
        'stats_from_prediction', 
        'display_feature_contributions',
        'get_feature_contributions',
        'get_correlated_features', 
        'get_global_score'
    ]
}

def assist_me(*tasks: str, on_error='warn'):
    """
    Provides some tool recommendations for specified tasks using the 
    :mod:`gofast.tools` or :mod:`gofast.dataops` library.

    Function dynamically fetches some tools related to user-specified tasks and
    organizes them into categories. It returns detailed descriptions and 
    recommendations for each tool. If an invalid task is provided, the function
    can warn the user or silently ignore the error based on the 'on_error' 
    parameter.

    Parameters
    ----------
    *tasks : str
        One or more task descriptions for which the user seeks tool recommendations.
    on_error : str, optional
        Error handling strategy when an invalid task is provided. Options are:
        - 'warn': Warn the user about the invalid input (default).
        - 'ignore': Silently ignore any invalid tasks.

    Returns
    -------
    str or dict
        If valid tasks are provided, returns a dictionary with tasks as keys 
        and tool descriptions as values. If no valid tasks are provided, 
        returns an error message.

    Examples
    --------
    >>> from gofast.assistance import assist_me
    >>> assist_me('data_preprocessing', 'model_evaluation_and_analysis')
  
    >>> assist_me('invalid_task', on_error='warn')
    Warning: Invalid task(s) provided: invalid_task. Please select from:\
        data_preprocessing, model_evaluation_and_analysis.
    """
    valid_tasks = list(TASK_MAPPING.keys())
    invalid_tasks = [task for task in tasks if task not in valid_tasks]
    if invalid_tasks:
        if on_error == 'warn':
            warnings.warn(f"Invalid task(s) provided: {', '.join(invalid_tasks)}."
                          f" Please select from: {', '.join(valid_tasks)}")

    tasks = [task for task in tasks if task in valid_tasks]

    if not tasks:
        return ("No valid tasks provided. Unable to proceed. Please provide at"
                " least one valid task. Available tasks are: {', '.join(valid_tasks)}.")

    task_tool_mapping = {}
    for task in tasks:
        tools = TASK_MAPPING.get(task, [])
        module_dict = {}
        for tool in tools: 
            module = 'tools' if hasattr(gofast.tools, tool) else 'dataops'
            value = getattr(gofast.tools, tool) if hasattr(
                gofast.tools, tool) else getattr(gofast.dataops, tool)
            module_dict[f'gofast.{module}.{tool}'] = remove_extra_spaces( 
                value.__doc__.split(".")[0].strip().replace("\n", '')
                )
        task_tool_mapping[task] = module_dict

    if not task_tool_mapping:
        return "Task not found. Please provide a more specific task description."

    tables = []
    for category, tools in task_tool_mapping.items():
        tool_table = ReportFactory(title=category.replace('_', ' ').title())
        tool_table.add_mixed_types(tools, table_width=TW)
        tables.append(tool_table)

    assemble_reports(*tables, display=True)

def gofast_explorer(package_path, /,  exclude_names=None):
    """
    Provides a guided exploration of the 'gofast' package, returning descriptions
    for modules or functions, excluding any specified by the user or those 
    beginning with an underscore.

    Parameters
    ----------
    package_path : str
        The subpackage or module path within the 'gofast' package. This can be 
        a broad subpackage name like 'stats' or a specific module like 
        'stats.descriptive'.
    exclude_names : list, optional
        A list of names to exclude from the results. This could include names 
        of submodules, modules, or functions that should not be returned in 
        the output dictionary.

    Returns
    -------
    dict
        A dictionary where keys are fully qualified names and values are the 
        first sentence of their documentation. If documentation is missing, a
        placeholder text is provided.

    Examples
    --------
    >>> from gofast.assistance import gofast_explorer 
    >>> gofast_explorer('stats')
    {'stats.model_comparisons': 'Provides functions for comparing statistical models.',
     'stats.inferential': 'Supports inferential statistical testing.'}
    >>> gofast_explorer('stats')
    ================================================================================
                                     Package stats                                  
    --------------------------------------------------------------------------------
    gofast.stats.descriptive          : See More in :ref:`User Guide`.
    gofast.stats.inferential          : See More in :ref:`User Guide`.
    gofast.stats.model_comparisons    : See More in :ref:`User Guide`.
    gofast.stats.probs                : These functions offer a range of
                                        probability utilities suitable for large
                                        datasets, leveraging the power of NumPy
                                        and SciPy for efficient computation
    gofast.stats.relationships        : See More in :ref:`User Guide`.
    gofast.stats.survival_reliability : See More in :ref:`User Guide`.
    gofast.stats.utils                : See More in :ref:`User Guide`.
    ================================================================================

    >>> gofast_explorer('stats.descriptive', exclude_names=['describe'])
    {'stats.descriptive.get_range': 'Returns the range of data in the dataset.',
     'stats.descriptive.hmean': 'Calculates the harmonic mean of the data.'}
    >>> gofast_explorer( 'dataops.inspection')
    ================================================================================
                               Module dataops.inspection                            
    --------------------------------------------------------------------------------
    dataops.inspection.verify_data_integrity : Verifies the integrity of data
                                               within a DataFrame
    dataops.inspection.inspect_data          : Performs an exhaustive inspection
                                               of a DataFrame
    ================================================================================
    Notes
    -----
    The function uses dynamic importing and introspection to fetch module or 
    function level documentation. As it imports modules dynamically, ensure 
    that the 'gofast' package is correctly installed and accessible.

    If `exclude_names` is provided, any module, submodule, or function whose 
    name appears in this list will not be included in the results. Additionally,
    names starting with an underscore ('_') are automatically excluded to 
    avoid returning private or protected modules or functions unless they are
    explicitly requested.
    """

    default_exclusion = [
        'setup', 'seed_control', 'coreutils', 'funcutils', "test_res_api", 
        'tests', 'thread', 'version', 'validator', 'config']
    
    exclude_names = exclude_names or default_exclusion
    exclude_names = [exclude_names] if isinstance (exclude_names, str) else exclude_names
    
    # Exclude 'gofast.' in package path if exists
    package_path= str(package_path).lower().replace ('gofast.', '')
    base_package = "gofast"
    full_path = f"{base_package}.{package_path}"
    exclude_names = set(exclude_names if exclude_names else [])
    description_dict = {}

    try:
        module = importlib.import_module(full_path)
    except ImportError:
        return {package_path: "Module not found. Please check the package path."}

    if hasattr(module, '__path__'):  # It's a package
        for importer, modname, ispkg in pkgutil.iter_modules(module.__path__, module.__name__ + '.'):
            if modname.split('.')[-1][0] == '_' or modname.split('.')[-1] in exclude_names:
                continue
            mod = importlib.import_module(modname)
            doc = mod.__doc__.split('.')[0] if mod.__doc__ else "See More in :ref:`User Guide`."
            description_dict[modname] = remove_extra_spaces(doc)
    else:  # It's a module
        if hasattr(module, '__all__'):
            for func_name in module.__all__:
                if func_name[0] == '_' or func_name in exclude_names:
                    continue
                func = getattr(module, func_name, None)
                doc = func.__doc__.split('.')[0] if func.__doc__ else "Documentation not available."
                description_dict[f"{package_path}.{func_name}"] = remove_extra_spaces(doc)
        else:
            doc = module.__doc__.split('.')[0] if module.__doc__ else "See More in :ref:`User Guide`."
            description_dict[package_path] = remove_extra_spaces(doc)
    
    # Generate and display a report
    TW = get_table_size()
    title = f"{'Package' if hasattr(module, '__path__') else 'Module'} {package_path}"
    description_report = ReportFactory(title=title)
    description_report.add_mixed_types(description_dict, table_width=TW)

    print(description_report)


