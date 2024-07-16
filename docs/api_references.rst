
.. _api_ref:

===============
API Reference
===============

Overview
--------
The GoFast API is a comprehensive machine learning and data analysis toolbox 
designed to accelerate the development of data-driven applications. It offers a 
wide range of subpackages, each targeting specific aspects of data handling, 
analysis, and model development.

Subpackages
-----------
- **analysis**: Tools for in-depth data analysis, including exploratory data analysis and feature selection.
- **base**: Core classes and functions that form the foundation of the GoFast toolkit.
- **datasets**: A collection of datasets for experimentation, benchmarking, and practice.
- **estimators**: Advanced machine learning estimators for both regression and classification tasks.
- **geo**: Geospatial data handling and analysis functions, useful for geographic data processing.
- **metrics**: Evaluation metrics and performance measures for assessing models.
- **models**: Optimize, pre-built models and architectures for a variety of machine learning applications.
- **plot**: Visualization tools to create informative and interactive plots and charts.
- **stats**: Statistical functions and tests to analyze data and derive meaningful insights.
- **tools**: Utility functions and helpers that streamline common tasks and data manipulations.
- **transformers**: Data transformation and preprocessing tools for preparing datasets for analysis.
- **query**:


Usage
-------
To utilize the functionalities offered by the GoFast API, import the required subpackage:

.. code-block:: python

   import gofast as gf  # or
   from gofast.tools import speed_rowwise_process #or
   from gofast.estimators import HammersteinWienerRegressor #or 
   from gofast.models.optimize import parallelize_estimators

Each sub-package is designed with a user-friendly interface, ensuring easy access and integration 
into data science workflows. Whether you are conducting exploratory data analysis, building complex 
machine learning models, or creating insightful visualizations, the GoFast API provides the necessary 
tools and functions to streamline your work.

The :code:`GoFast` library provides a range of utilities designed to accelerate 
machine learning workflows. This API reference provides detailed documentation for all the modules, 
classes, and functions within :code:`GoFast`.

.. toctree::
   :maxdepth: 2

   gofast.analysis
   gofast.bases
   gofast.datasets
   gofast.geo
   gofast.models
   gofast.metrics
   gofast.query
   gofast.tools
   gofast.transformers 
   gofast.plot 
   gofast.stats 

.. _analysis_ref:

:mod:`gofast.analysis`: Analyses 
====================================

The `gofast.analysis` is a comprehensive subpackage dedicated to various analytical 
techniques and methodologies. It includes modules for factor analysis (:mod:`gofast.analysis.factors`), 
dimensionality reduction (:mod:`gofast.analysis.dimensionality`), and decomposition (:mod:`gofast.analysis.decomposition`),
offering a wide array of tools for advanced data analysis, feature extraction, and 
pattern recognition.
 
.. automodule:: gofast.analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`analysis <analysis>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
	analysis.compare_pca_fa_scores
	analysis.get_eigen_components
	analysis.get_pca_fa_scores
	analysis.get_most_variance_component
	analysis.get_total_variance_ratio
	analysis.iPCA
	analysis.kPCA
	analysis.ledoit_wolf_score
	analysis.linear_discriminant_analysis
	analysis.LLE
	analysis.make_scedastic_data
	analysis.nPCA
    analysis.oblimin_rotation 
	analysis.principal_axis_factoring
	analysis.plot_decision_regions
	analysis.project_ndim_vs_explained_variance
	analysis.promax_rotation
	analysis.rotated_factor
	analysis.samples_hotellings_t_square
	analysis.spectral_fa	
	analysis.transform_to_principal_components
	analysis.varimax_rotation  

.. _base_ref:

:mod:`gofast.base`: Base Data Operations 
==========================================

The `gofast.base` module includes data classes for loading and manipulations single  
or many dataframes at the same time. :mod:`~gofast.base` provides core data operations 
and utilities, forming the backbone of the toolbox. It encompasses essential functionalities for 
data manipulation and transformation, serving as a foundation for a wide range of 
data-centric tasks and workflows.
 
.. automodule:: gofast.base
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`base <base>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   base.Data
   base.FeatureProcessor
   base.FrameOperations
   base.MergeableSeries
   base.MergeableFrames
   base.MissingHandler
   base.TargetProcessor

.. _datasets_ref:

:mod:`gofast.datasets`: Datasets 
==================================

"The :code:`Gofast` :mod:`~gofast.datasets` module is a versatile collection of curated 
datasets, encompassing a wide range of domains and data types. It serves as a valuable 
resource for data exploration, experimentation, and model training across various 
machine learning and data science projects, offering ready-to-use datasets that 
expedite research and analysis.
 
.. automodule:: gofast.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets <datasets>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   datasets.fetch_data
   datasets.load_bagoue
   datasets.load_dyspnea
   datasets.load_forensic 
   datasets.load_iris
   datasets.load_hlogs
   datasets.load_hydro_metrics
   datasets.load_nlogs 
   datasets.load_mxs 
   datasets.load_jrs_bet 
   datasets.load_statlog
   datasets.make_sounding 
   datasets.make_african_demo_info 
   datasets.make_agronomy_feedback
   datasets.make_cc_factors
   datasets.make_classification
   datasets.make_elogging 
   datasets.make_erp 
   datasets.make_ert 
   datasets.make_gadget_sales 
   datasets.make_medical_diagnosis
   datasets.make_mining_ops
   datasets.make_regression
   datasets.make_retail_store 
   datasets.make_social_media_comments
   datasets.make_tem 
   datasets.make_water_demand
   datasets.make_well_logging 


.. _estimators_ref:

:mod:`gofast.estimators`: Estimators 
======================================

The :mod:`gofast.estimators` module is a comprehensive collection of 
advanced machine learning estimators designed for both regression and 
classification tasks. It extends beyond traditional models to include 
innovative and hybrid approaches, blending various machine learning techniques 
to address complex data patterns and diverse problem sets.

.. automodule:: gofast.estimators
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`estimators <estimators>` section for further details.


Classes
~~~~~~~~~~~~

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst

   estimators.AdalineClassifier
   estimators.AdalineMixte
   estimators.AdalineRegressor
   estimators.AdalineStochasticClassifier
   estimators.AdalineStochasticRegressor
   estimators.BasePerceptron
   estimators.BenchmarkRegressor 
   estimators.BenchmarkClassifier 
   estimators.BoostedRegressionTree
   estimators.BoostedClassifierTree
   estimators.DecisionStumpRegressor
   estimators.DecisionTreeBasedRegressor
   estimators.DecisionTreeBasedClassifier
   estimators.GradientDescentClassifier
   estimators.GradientDescentRegressor
   estimators.KMFClassifier 
   estimators.KMFRegressor
   estimators.HammersteinWienerClassifier
   estimators.HammersteinWienerRegressor
   estimators.HBTEnsembleClassifier
   estimators.HBTEnsembleRegressor
   estimators.HWEnsembleClassifier
   estimators.HWEnsembleRegressor
   estimators.HybridBoostedTreeClassifier
   estimators.HybridBoostedTreeRegressor
   estimators.MajorityVoteClassifier
   estimators.NeuroFuzzyEnsemble
   estimators.SimpleAverageClassifier
   estimators.SimpleAverageRegressor
   estimators.WeightedAverageClassifier
   estimators.WeightedAverageRegressor
   
:mod:`gofast.geo`: Geosciences
================================

The :mod:`gofast.geo` module is tailored for geosciences, offering a specialized toolkit of tools and functions
to streamline the processing, analysis, and visualization of geospatial data. It empowers users to work effectively
with geological and geographical data. 

This module includes utilities for managing geology, drilling, and borehole data, which can be
utilized in machine learning tasks. To utilize the :ref:`geosciences <geo>` module, you will need to have the
``pyproj`` library or ``osgeo`` installed. If you prefer to use ``osgeo``, you must also install and configure
`GDAL <https://opensourceoptions.com/how-to-install-gdal/>`_.

In some cases, effective manipulation of geosciences datasets may require a solid foundation in the field. 
:gofast: provides specific feature engineering techniques tailored for handling complex boreholes, land subsidence data,
and more. For more comprehensive geoscience capabilities, it is recommended to explore specialized libraries such as
`watex <https://watex.readthedocs.io>`_ or others.

.. automodule:: gofast.geo
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`geosciences <geo>` section for further details.


Classes
~~~~~~~~~~~~

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst

   geo.Profile
   geo.Location
   
Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   geo.build_random_thickness 
   geo.classify_k
   geo.find_aquifer_groups
   geo.find_similar_labels
   geo.get_azimuth
   geo.get_bearing
   geo.get_stratum_thickness
   geo.get_aquifer_section 
   geo.get_aquifer_sections
   geo.get_unique_section
   geo.get_compressed_vector 
   geo.get_sections_from_depth
   geo.label_importance
   geo.make_coords 
   geo.make_mxs_labels
   geo.partition_holes
   geo.plot_stratalog
   geo.predict_nga_labels
   geo.reduce_samples
   geo.refine_locations 
   geo.select_base_stratum 
   geo.smart_thickness_ranker 

.. _models_ref:

gofast.models
===============

The `gofast.models` module contains various machine learning models 
optimization tricks for speed and efficiency. The subpackages include a suite of 
specialized tools for various aspects of machine learning and model 
optimization. It encompasses the :mod:`gofast.models.search` module 
for model validation, the :mod:`gofast.models.optimize` module for
model optimization, :mod:gofast.models.utils` module for versatile tasks, 
and the :mod:`gofast.models.depp_search` module for fine-tuning 
hyperparameters in machine learning and neural networks.
 

Classes
~~~~~~~~~~~~

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.search.BaseEvaluation
   models.search.CrossValidator
   models.search.BaseSearch
   models.search.SearchMultiple
   models.search.MultipleSearch
   models.selection.SequentialSearchCV
   models.selection.SwarmSearchCV
   models.selection.AnnealingSearchCV
   models.selection.EvolutionarySearchCV
   models.selection.GradientBasedSearchCV
   models.selection.GeneticSearchCV
   
   
Functions
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: function.rst

   models.get_best_kPCA_params
   models.get_scorers
   models.get_cv_mean_std_scores
   models.get_split_best_scores 
   models.display_cv_tables 
   models.display_fine_tuned_results
   models.display_model_max_details
   models.dummy_evaluation
   models.parallelize_estimators
   models.optimize_hyperparameters 
   models.optimize_search 
   models.optimize_search2
   models.shrink_covariance_cv_score
   models.base_tuning 
   models.robust_tuning 
   models.neural_tuning
   models.deep_tuning
   models.find_best_C
   models.calculate_aggregate_scores 
   models.analyze_score_distribution 
   models.estimate_confidence_interval 
   models.rank_cv_scores
   models.filter_scores
   models.visualize_score_distribution 
   models.performance_over_time 
   models.calculate_custom_metric 
   models.handle_missing_in_scores 
   models.export_cv_results 
   models.comparative_analysis 
   models.plot_parameter_importance 
   models.plot_hyperparameter_heatmap 
   models.plot_learning_curve 
   models.plot_validation_curve 
   models.plot_feature_importance
   models.plot_roc_curve_per_fold 
   models.plot_confidence_intervals 
   models.plot_pairwise_model_comparison
   models.plot_feature_correlation 
   models.base_evaluation 

.. _metrics_ref:

gofast.metrics
================

`gofast.metrics` includes performance metrics for evaluating models.
The module offers a comprehensive collection of performance metrics 
and evaluation tools, aiding users in quantifying and assessing the 
effectiveness and accuracy of models and algorithms in data analysis 
and machine learning tasks.

.. automodule:: gofast.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.precision_recall_tradeoff
   metrics.roc_curve_
   metrics.confusion_matrix_ 
   metrics.get_eval_scores
   metrics.mean_squared_log_error
   metrics.balanced_accuracy
   metrics.information_value
   metrics.mean_absolute_error
   metrics.mean_squared_error 
   metrics.root_mean_squared_error
   metrics.r_squared
   metrics.mean_absolute_percentage_error 
   metrics.explained_variance_score
   metrics.median_absolute_error
   metrics.max_error
   metrics.mean_squared_log_error
   metrics.mean_poisson_deviance
   metrics.mean_gamma_deviance
   metrics.mean_absolute_deviation
   metrics.dice_similarity_coeff
   metrics.gini_coeff
   metrics.hamming_loss
   metrics.fowlkes_mallows_index
   metrics.rmse_log_error
   metrics.mean_percentage_error
   metrics.percentage_bias
   metrics.spearmans_rank_correlation
   metrics.precision_at_k 
   metrics.ndcg_at_k 
   metrics.mean_reciprocal_rank
   metrics.average_precision
   metrics.jaccard_similarity_coeff

.. _tools_ref:

gofast.tools
=============

Utility functions supporting various operations within :code:`GoFast`.
:mod:`gofast.tools` module provides a set of common and new utilities for fast preprocessing 
data and handle a large dataset. The module encompasses a range of powerful and efficient tools 
designed to enhance various aspects of data processing, analysis, and visualization, 
streamlining workflows and improving productivity. The list of the tools are not exhaustive. 

.. automodule:: gofast.tools
   :no-members:
   :no-inherited-members:

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tools.adaptive_moving_average
   tools.apply_bow_vectorization 
   tools.apply_tfidf_vectorization 
   tools.apply_word_embeddings 
   tools.array2hdf5
   tools.audit_data
   tools.augment_data
   tools.assess_outlier_impact
   tools.bi_selector
   tools.bin_counting
   tools.binning_statistic
   tools.boxcox_transformation
   tools.build_data_preprocessor
   tools.butterworth_filter
   tools.calculate_binary_iv, 
   tools.calculate_optimal_bins, 
   tools.categorize_target
   tools.category_count
   tools.check_missing_data
   tools.cleaner
   tools.codify_variables
   tools.compute_effort_yield
   tools.compute_sunburst_data 
   tools.convert_date_features
   tools.cubic_regression
   tools.denormalize
   tools.discretize_categories
   tools.enrich_data_spectrum
   tools.evaluate_model
   tools.evaluate_model
   tools.exponential_regression
   tools.export_target
   tools.fancier_downloader
   tools.features_in
   tools.fetch_model
   tools.fetch_tgz
   tools.find_features_in
   tools.get_bearing
   tools.get_correlated_features
   tools.get_distance
   tools.get_global_score
   tools.get_remote_data
   tools.get_target
   tools.handle_categorical_features
   tools.handle_missing_data
   tools.handle_outliers_in_data
   tools.infer_sankey_columns 
   tools.inspect_data
   tools.interpolate1d
   tools.interpolate2d
   tools.interpolate_grid
   tools.label_importance
   tools.labels_validator
   tools.laplace_smoothing
   tools.linear_regression
   tools.linkage_matrix
   tools.load_csv
   tools.load_dumped_data
   tools.load_model
   tools.logarithmic_regression
   tools.make_mxs
   tools.make_pipe
   tools.minmax_scaler
   tools.moving_average
   tools.normalize
   tools.normalizer
   tools.pair_data
   tools.projection_validator
   tools.quadratic_regression
   tools.quality_control
   tools.random_sampling
   tools.random_selector
   tools.read_data
   tools.remove_outliers
   tools.rename_labels_in
   tools.replace_data
   tools.request_data
   tools.resampling
   tools.reshape
   tools.sanitize
   tools.save_dataframes
   tools.save_or_load
   tools.save_job
   tools.savgol_filter
   tools.scale_data
   tools.scale_y
   tools.select_feature_importances
   tools.select_features
   tools.serialize_data
   tools.simple_extractive_summary
   tools.sinusoidal_regression
   tools.smart_label_classifier
   tools.smooth1d
   tools.smoothing
   tools.soft_bin_stats
   tools.soft_data_split
   tools.soft_imputer
   tools.soft_scaler
   tools.summarize_text_columns
   tools.speed_rowwise_process
   tools.split_train_test
   tools.split_train_test_by_id
   tools.standard_scaler
   tools.stats_from_prediction
   tools.step_regression
   tools.store_data 
   tools.store_or_write_hdf5
   tools.stratify_categories
   tools.to_numeric_dtypes
   tools.transform_dates
   tools.verify_data_integrity
      
.. _plot_ref:

:mod:`gofast.plot`: Visualization 
==================================

mod:`gofast.plot` sub-package combines 'Exploratory Plots' and 
'Evaluation Plots' along with versatile 'Plot Utilities.' This 
module empowers users to explore, analyze, and evaluate data 
efficiently through insightful visual representations to enhance 
data analysis and decision-making.

.. automodule:: gofast.plot
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization <plot>` section for further details.

Classes
~~~~~~~~~

.. currentmodule:: gofast


.. autosummary::
   :toctree: generated/
   :template: class.rst

   plot.EasyPlotter
   plot.EvalPlotter
   plot.MetricPlotter
   plot.QuestPlotter
   plot.TimeSeriesPlotter

Methods
~~~~~~~~~~~~

:mod:`~gofast.plot.explore`: Exploratory Plots  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"Exploratory plots" are visualizations that help users explore and 
understand their data by providing insights and patterns, making 
data analysis more accessible and insightful.

.. automodule:: gofast.plot.explore 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization  <plot>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot.EasyPlotter.plotHistCatDistribution
	plot.EasyPlotter.ExPlot.plotBarCatDistribution
	plot.EasyPlotter.plotMultiCatDistribution
	plot.EasyPlotter.plotCorrMatrix
	plot.EasyPlotter.plotNumFeatures
	plot.EasyPlotter.plotJoint2Features
	plot.EasyPlotter.plotSemanticScatter
	plot.EasyPlotter.plotDiscussingFeatures
	plot.EasyPlotter.PlotCoordDistributions
	plot.QuestPlotter.plotClusterParallelCoords
    plot.QuestPlotter.plotRadViz
	plot.QuestPlotter.plotPairwiseFeatures
	plot.QuestPlotter.plotCutQuantiles
	plot.QuestPlotter.plotDistributions
	plot.QuestPlotter.plotPairGrid
	plot.QuestPlotter.plotFancierJoin
	plot.QuestPlotter.plotScatter
	plot.QuestPlotter.plotHistBinTarget
	plot.QuestPlotter.plotHist
	plot.QuestPlotter.plotMissingPatterns

:mod:`~gofast.plot.evaluate`: Evaluation Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"Evaluation plots" refer to a set of visual representations 
commonly used in the context of model evaluation. These plots 
provide insights into the performance, accuracy, and behavior of 
models or algorithms. Plot typically include metrics such as 
accuracy, precision, recall, F1-score, ROC curves, and more. 
They enable users to visualize the trade-offs and characteristics 
of different models or algorithms, facilitating the evaluation and 
optimization process.

.. automodule:: gofast.plot.evaluate 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization <plot>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot.EvalPlotter.plotPCA
	plot.EvalPlotter.plotPR
	plot.EvalPlotter.plotROC
	plot.EvalPlotter.plotRobustPCA
	plot.EvalPlotter.plotConfusionMatrix
	plot.EvalPlotter.plotFeatureImportance
	plot.EvalPlotter.plotHeatmap
	plot.EvalPlotter.plotBox
	plot.EvalPlotter.plotHistogram
	plot.EvalPlotter.plot2d
	plot.MetricPlotter.plotConfusionMatrix
	plot.MetricPlotter.plotRocCurve
	plot.MetricPlotter.plotPrecisionRecallCurve
	plot.MetricPlotter.plotLearningCurve
	plot.MetricPlotter.plotSilhouette
	plot.MetricPlotter.plotLiftCurve
	plot.MetricPlotter.plotCumulativeGain
	plot.MetricPlotter.plotPrecisionRecallPerClass
	plot.MetricPlotter.plotActualVSPredicted

:mod:`~gofast.plot.ts`: TimeSeries Plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"TimeSeries plots" are specialized visualizations for analyzing time-series data. 
These plots provide valuable insights into the trends, patterns, and characteristics 
of data that changes over time. They are essential tools in various fields such as 
finance, meteorology, and signal processing, helping users to understand temporal 
dynamics and make predictions based on historical data trends.

.. automodule:: gofast.plot.ts
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization <plot>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

    plot.TimeSeriesPlotter.heatmapCorrelation
    plot.TimeSeriesPlotter.pieChart
    plot.TimeSeriesPlotter.plotArea
    plot.TimeSeriesPlotter.plotAutocorrelation
    plot.TimeSeriesPlotter.plotBar
    plot.TimeSeriesPlotter.plotBox
    plot.TimeSeriesPlotter.plotBubble
    plot.TimeSeriesPlotter.plotCumulativeDistribution
    plot.TimeSeriesPlotter.plotCumulativeLine
    plot.TimeSeriesPlotter.plotDecomposition
    plot.TimeSeriesPlotter.plotDensity
    plot.TimeSeriesPlotter.plotErrorbar
    plot.TimeSeriesPlotter.plotHexbin
    plot.TimeSeriesPlotter.plotHistogram
    plot.TimeSeriesPlotter.plotkde
    plot.TimeSeriesPlotter.plotLag
    plot.TimeSeriesPlotter.plotLine
    plot.TimeSeriesPlotter.plotpacf
    plot.TimeSeriesPlotter.plotRadViz
    plot.TimeSeriesPlotter.plotRollingMean
    plot.TimeSeriesPlotter.plotScatter
    plot.TimeSeriesPlotter.plotScatterWithTrendline
    plot.TimeSeriesPlotter.plotStackedArea
    plot.TimeSeriesPlotter.plotStackedBar
    plot.TimeSeriesPlotter.plotStackedLine
    plot.TimeSeriesPlotter.plotStep
    plot.TimeSeriesPlotter.plotSunburst
    plot.TimeSeriesPlotter.plotWaterfall
    plot.TimeSeriesPlotter.radarChart

Functions
~~~~~~~~~~~~

:mod:`~gofast.plot.utils`: Plot Utilities 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plot utilities are a set of functions and tools designed to 
simplify and enhance the process of data visualization in the 
Gofast API. These utilities provide developers with a convenient 
way to create, customize, and display various types of plots and 
charts, making it easier to communicate and analyze data visually. 
Whether you need to create scatterplots, histograms, line charts, or
more complex visualizations, plot utilities in the Gofast API offer 
a seamless and user-friendly way to generate compelling graphics 
that aid in data exploration, analysis, and presentation.

.. automodule:: gofast.plot.utils 
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`visualization  <plot>` section for further details.

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.utils.plot_unified_pca
   plot.utils.plot_learning_inspection
   plot.utils.plot_learning_inspections
   plot.utils.plot_silhouette
   plot.utils.plot_dendrogram
   plot.utils.plot_dendroheat
   plot.utils.plot_loc_projection
   plot.utils.plot_model
   plot.utils.plot_reg_scoring 
   plot.utils.plot_matshow
   plot.utils.plot_model_scores
   plot.utils.plot_mlxtend_heatmap
   plot.utils.plot_mlxtend_matrix
   plot.utils.plot_cost_vs_epochs 
   plot.utils.plot_elbow
   plot.utils.plot_clusters
   plot.utils.plot_pca_components
   plot.utils.plot_base_dendrogram
   plot.utils.plot_learning_curves 
   plot.utils.plot_confusion_matrices 
   plot.utils.plot_yb_confusion_matrix 
   plot.utils.plot_sbs_feature_selection
   plot.utils.plot_regularization_path
   plot.utils.plot_rf_feature_importances
   plot.utils.plot_base_silhouette
   plot.utils.plot_voronoi
   plot.utils.plot_roc_curves 
   plot.utils.plot_l_curve
   plot.utils.plot_taylor_diagram 
   plot.utils.plot_cv
   plot.utils.plot_confidence
   plot.utils.plot_confidence_ellipse
   plot.utils.plot_text
   plot.utils.plot_cumulative_variance 
   plot.utils.plot_shap_summary 
   plot.utils.plot_custom_boxplot
   plot.utils.plot_abc_curve
   plot.utils.plot_permutation_importance
   plot.utils.create_radar_chart
   plot.utils.plot_r_squared
   plot.utils.plot_cluster_comparison
   plot.utils.plot_sunburst 
   plot.utils.plot_sankey
   plot.utils.plot_euler_diagram 
   plot.utils.create_upset_plot 
   plot.utils.plot_venn_diagram 
   plot.utils.create_matrix_representation 
   plot.utils.plot_feature_interactions 
   plot.utils.plot_regression_diagnostics 
   plot.utils.plot_residuals_vs_leverage 
   plot.utils.plot_residuals_vs_fitted
   plot.utils.plot_variables 
   plot.uils.plot_correlation_with_target

.. _stats_ref:

gofast.stats
====================

The :mod:`gofast.stats` sub-package offers a comprehensive suite of 
statistical functions and tools tailored for high-performance data analysis.
This module is designed to facilitate efficient and accurate statistical 
computations, providing users with a broad range of functionalities from 
basic descriptive statistics to advanced probabilistic modeling. For more-in-depth, 
user may visit other powerful libraries such as ``scipy.stats``, ``statsmodels``, etc.

.. automodule:: gofast.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: function.rst

   stats.mean
   stats.median
   stats.mode 
   stats.var
   stats.std
   stats.get_range
   stats.quartiles
   stats.correlation
   stats.iqr 
   stats.z_scores
   stats.describe
   stats.skew
   stats.kurtosis
   stats.t_test_independent
   stats.perform_linear_regression
   stats.chi2_test
   stats.anova_test
   stats.perform_kmeans
   stats.normal_pdf
   stats.normal_cdf
   stats.binomial_pmf
   stats.poisson_logpmf
   stats.uniform_sampling
   stats.stochastic_volatility_model
   stats.hierarchical_linear_model
   stats.harmonic_mean
   stats.weighted_median
   stats.bootstrap 
   stats.kaplan_meier_analysis 
   stats.get_gini_coeffs
   stats.multidim_scaling
   stats.dca_analysis
   stats.spectral_clustering
   stats.levene_test
   stats.kolmogorov_smirnov_test
   stats.cronbach_alpha
   stats.friedman_test
   stats.statistical_tests
  
.. _transformers_ref:

gofast.transformers
====================

The :mod:`gofast.transformers` module is a versatile collection of data transformers 
designed to streamline data preprocessing and transformation tasks. These transformers 
offer essential tools for encoding, scaling, and modifying data, making it easier for 
users to prepare their datasets for machine learning and analysis.

.. automodule:: gofast.transformers
   :no-members:
   :no-inherited-members:

.. currentmodule:: gofast

.. autosummary::
   :toctree: generated/
   :template: class.rst

   transformers.AttributesCombinator
   transformers.SequentialBackwardSelection
   transformers.FloatCategoricalToIntTransformer
   transformers.KMeansFeaturizer
   transformers.StratifyFromBaseFeature
   transformers.BaseColumnSelector 
   transformers.BaseCategoricalEncoder 
   transformers.BaseFeatureScaler 
   transformers.CategoryBaseStratifier 
   transformers.CategorizeFeatures
   transformers.FrameUnion
   transformers.DataFrameSelector
   transformers.CombinedAttributesAdder
   transformers.FeaturizeX
   transformers.TextFeatureExtractor 
   transformers.DateFeatureExtractor
   transformers.FeatureSelectorByModel
   transformers.PolynomialFeatureCombiner 
   transformers.DimensionalityReducer
   transformers.CategoricalEncoder
   transformers.FeatureScaler
   transformers.MissingValueImputer
   transformers.ColumnSelector
   transformers.LogTransformer
   transformers.TimeSeriesFeatureExtractor
   transformers.CategoryFrequencyEncoder
   transformers.DateTimeCyclicalEncoder
   transformers.LagFeatureGenerator
   transformers.DifferencingTransformer
   transformers.MovingAverageTransformer
   transformers.CumulativeSumTransformer
   transformers.SeasonalDecomposeTransformer
   transformers.FourierFeaturesTransformer
   transformers.TrendFeatureExtractor 
   transformers.ImageResizer 
   transformers.ImageNormalizer
   transformers.ImageToGrayscale
   transformers.ImageAugmenter
   transformers.ImageChannelSelector
   transformers.ImageFeatureExtractor
   transformers.ImageEdgeDetector
   transformers.ImageHistogramEqualizer
   transformers.ImagePCAColorAugmenter
   transformers.ImageBatchLoader