"""
Validation and optimization sub-package. 

The  `gofast.models` subpackage organizes and exposes various model
training, evaluation, and hyperparameter optimization functionalities. 
It offers a comprehensive suite of tools for machine learning practitioners 
looking to build, evaluate, and optimize models efficiently. 
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

try:
    import tensorflow as tf # noqa
except :
    pass
else: 
    from .deep_search import (
        plot_history,
        base_tuning,
        robust_tuning,
        build_mlp_model,
        fair_neural_tuning,
        deep_cv_tuning,
        train_and_evaluate2,
        train_and_evaluate,
        Hyperband,
        PBTTrainer,
        custom_loss,
        train_epoch,
        calculate_validation_loss,
        data_generator,
        evaluate_model,
        train_model,
        create_lstm_model,
        create_cnn_model,
        create_autoencoder_model,
        create_attention_model,
        plot_errors,
        plot_predictions, 
        find_best_lr, 
        create_sequences, 
        build_lstm_model, 
        make_future_predictions, 
        lstm_ts_tuner, 
        cross_validate_lstm,
    )
    __all__+=[
        "plot_history",
        "base_tuning",
        "robust_tuning",
        "build_mlp_model",
        "fair_neural_tuning",
        "deep_cv_tuning",
        "train_and_evaluate2",
        "train_and_evaluate",
        "Hyperband",
        'PBTTrainer',
        "custom_loss",
        "train_epoch",
        "calculate_validation_loss",
        "data_generator",
        "evaluate_model",
        "train_model",
        "create_lstm_model",
        "create_cnn_model",
        "create_autoencoder_model",
        "create_attention_model",
        "plot_errors",
        "plot_predictions", 
        "find_best_lr", 
        "create_sequences", 
        "make_future_predictions", 
        "build_lstm_model", 
        "lstm_ts_tuner", 
        "cross_validate_lstm", 
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
