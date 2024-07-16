# -*- coding: utf-8 -*-

"""Provides tools for assisting users with navigating and utilizing the features
 of the `gofast` library, including direct support and exploration functions."""

import importlib
import textwrap
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
        'denormalizer', 
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
        'handle_unique_identifiers',
        'save_or_load', 
        'fetch_remote_data',
        'read_data', 
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
        'one_click_prep',
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
        'correlation_ops', 
        'drop_correlated_features',
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
        'interpolate_data', 
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

TASK_DESCRIPTIONS = {
    'data_preprocessing': (
        "Tasks related to preparing and cleaning data before analysis or "
        "modeling..."
    ),
    'data_transformation': (
        "Tasks involving the transformation of data to make it suitable for "
        "analysis or modeling..."
    ),
    'data_management': (
        "Tasks for managing data storage, retrieval, and handling datasets "
        "efficiently..."
    ),
    'feature_engineering': (
        "Tasks for creating, selecting, and transforming features to improve "
        "model performance..."
    ),
    'model_preparation': (
        "Tasks related to preparing data for model training and evaluation, "
        "including building data preprocessors, and handling data imbalances..."
    ),
    'data_analysis': (
        "Tasks for analyzing data to understand patterns, quality, and "
        "integrity including performing data, audits, and verifying data integrity..."
    ),
    'statistics_and_math': (
        "Tasks involving statistical and mathematical operations for data "
        "analysis and modeling..."
    ),
    'model_evaluation_and_analysis': (
        "Tasks for evaluating and analyzing model performance and feature "
        "contributions..."
    )
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
    if tasks and ( str(tasks[0]).lower() =='help' or tasks[0]==help): 
        return _assist_me(tasks[0])
  
    tasks = _manage_task(tasks)
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
        task_tool_mapping[task] = dict(sorted ( module_dict.items()))

    if not task_tool_mapping:
        return "Task not found. Please provide a more specific task description."
    
    tables = []
    for category, tools in task_tool_mapping.items():
        tool_table = ReportFactory(title=category.replace('_', ' ').title())
        tool_table.add_mixed_types(tools, table_width=TW)
        tables.append(tool_table)

    assemble_reports(*tables, display=True)
    
def _manage_task (tasks): 
    import random 
    if str(tasks[0]).lower() in ("*", "all") or tasks[0]==all :
        return sorted (TASK_MAPPING.keys()) 
    elif str(tasks[0]).lower() =='any' or tasks[0] ==any: 
        # random select one task 
        return [random.choice(sorted (TASK_MAPPING.keys()))]
    return tasks 
        
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
    
    if str(package_path)=="gofast": 
        return _get_gofast_package_descriptions ()
    
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

def _get_gofast_package_descriptions(include_private=False):
    descriptions = {
        "_build": "Contains build scripts and configurations for the package.",
        "analysis": "Includes modules for data analysis and statistical computations.",
        "api": "Provides API interfaces and methods for external integration.",
        "backends": "Houses different backend implementations for various operations.",
        "cli": "Command-line interface tools and scripts.",
        "compat": "Ensures compatibility with different versions and dependencies.",
        "dataops": "Data operations and management utilities.",
        "datasets": "Contains datasets and data loading utilities.",
        "estimators": "Machine learning estimators and related utilities.",
        "experimental": "Experimental features and modules under development.",
        "externals": "External dependencies and third-party integrations.",
        "geo": "Geospatial data processing and analysis tools.",
        "gflogs": "Logging utilities specific to the gofast framework.",
        "models": "Defines various machine learning models.",
        "nn": "Neural network models, data processing, training, and hyperparameter tuning tools.",
        "plot": "Plotting and visualization tools.",
        "pyx": "Python extension modules for performance enhancement.",
        "stats": "Statistical functions and analysis tools.",
        "tools": "Miscellaneous tools and utilities for the package.",
        "transformers": "Transformers and preprocessing modules for data transformation.",
        "__init__[m]": "Initialization file for the package.",
        "_dep_config[m]": "Dependency configuration settings.",
        "_distributor_init[m]": "Initialization for distribution setup.",
        "_gofastlog[m]": "Logging configurations and settings for gofast.",
        "_public[m]": "Public API definitions and exports.",
        "assistance[m]": "Helper functions and assistance utilities.",
        "base[m]": "Base classes and core functionalities.",
        "config[m]": "Configuration settings and utilities.",
        "decorators[m]": "Decorators for various functionalities within the package.",
        "exceptions[m]": "Custom exceptions and error handling.",
        "metrics[m]": "Performance metrics and evaluation tools.",
        "model_selection[m]": "Tools for model selection and validation.",
        "query[m]": "Query utilities for data retrieval and manipulation.",
        "util[m]": "Base package initialization utility functions."
    }
    
    if not include_private:
        descriptions = {k: v for k, v in descriptions.items() if not k.startswith('_')}
    
    # Generate and display a report
    TW = get_table_size()
    title = "Subpackages & Modules[m]"
    description_report = ReportFactory(title=title)
    description_report.add_mixed_types(descriptions, table_width=TW)
    print(description_report)

def _assist_me(help_task):
    """
    This function provides a detailed description and assistance for various data 
    processing and analysis tasks. When called with 'help' or a specific task name,
    it generates a detailed guidance report on what each task entails and how to
    perform it using the gofast toolkit.

    Parameters
    ----------
    help_task : str
        The help keyword or task name.

    Returns
    -------
    None
        This function prints the assistance directly.
    """
    border = '=' * TW
    sub_border ='-' * TW 
    print(border)
    print("GOFast Assistance - How can I help?".center(TW))
    print(sub_border)
    
    message = (
        "I'm designed to give you a foundational understanding of data processing tools.\n"
        "To call me, use:\n\n"
        "    >>> assist_me('my_task').\n\n"
        "I provide basic tools. For a deeper dive, use the explorer tools with the "
        "following commands:\n\n"
        "    >>> import gofast as gf\n"
        "    >>> gf.config.PUBLIC= True  # make sure to set as True\n"
        "    >>> gf.explore('gofast.package.module_name')\n\n"
        "See the table below for tasks I can quickly perform or help you handle with your data."
    )
    
    # Applying textwrap to ensure each paragraph is formatted correctly within TW
    wrapped_message = "\n".join([textwrap.fill(paragraph, TW) for paragraph in message.split("\n")])
    print(wrapped_message)
    print()
    # Assuming ReportFactory and TASK_DESCRIPTIONS are defined and properly set up elsewhere
    description_report = ReportFactory(title="Available Tasks - Detailed Descriptions")
    description_report.add_mixed_types(dict(sorted(TASK_DESCRIPTIONS.items())), table_width=TW)
    print(description_report)
    
    print()
    print("How can I assist you further?".center(TW))
    print()
    print()

