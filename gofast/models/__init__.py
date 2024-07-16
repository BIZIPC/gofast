"""
Validation and optimization sub-package. 
"""

from .utils import ( 
    find_best_C, 
    get_scorers,
    get_cv_mean_std_scores, 
    get_split_best_scores, 
    display_cv_tables, 
    display_fine_tuned_results, 
    display_model_max_details, 
    shrink_covariance_cv_score, 
    calculate_aggregate_scores, 
    analyze_score_distribution, 
    estimate_confidence_interval, 
    rank_cv_scores, 
    filter_scores, 
    visualize_score_distribution, 
    performance_over_time, 
    calculate_custom_metric, 
    handle_missing_in_scores, 
    export_cv_results, 
    comparative_analysis,
    plot_parameter_importance, 
    plot_hyperparameter_heatmap, 
    visualize_learning_curve, 
    plot_validation_curve, 
    plot_feature_importance,
    plot_roc_curve_per_fold, 
    plot_confidence_intervals, 
    plot_pairwise_model_comparison,
    plot_feature_correlation, 
    base_evaluation, 
    get_best_kPCA_params
  ) 

__all__=[
    "get_best_kPCA_params", 
    "get_scorers",
    "get_cv_mean_std_scores", 
    "get_split_best_scores", 
    "display_cv_tables", 
    "display_fine_tuned_results", 
    "display_model_max_details", 
    "dummy_evaluation",
    "CrossValidator", 
    "shrink_covariance_cv_score", 
    "find_best_C", 
    "calculate_aggregate_scores", 
    "analyze_score_distribution", 
    "estimate_confidence_interval", 
    "rank_cv_scores", 
    "filter_scores", 
    "visualize_score_distribution", 
    "performance_over_time", 
    "calculate_custom_metric", 
    "handle_missing_in_scores", 
    "export_cv_results", 
    "comparative_analysis", 
    "plot_parameter_importance", 
    "plot_hyperparameter_heatmap", 
    "visualize_learning_curve", 
    "plot_validation_curve", 
    "plot_feature_importance",
    "plot_roc_curve_per_fold", 
    "plot_confidence_intervals", 
    "plot_pairwise_model_comparison",
    "plot_feature_correlation", 
    "base_evaluation"
    ]


import typing  
if typing.TYPE_CHECKING:
    from ._deep_selection import  HyperbandSearchCV # noqa
    
def __getattr__(name):
    if name =="HyperbandSearchCV":
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "`enable_hyperband_selection`:\n"
            "`from gofast.experimental import enable_hyperband_selection`"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
