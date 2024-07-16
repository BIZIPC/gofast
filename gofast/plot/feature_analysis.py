# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `feature_analysis` module provides visualization tools for feature 
analysis in machine learning models. It includes functions for 
plotting feature importances, interactions, correlations with targets, 
dependence plots, feature selection processes, permutation importance, 
and regularization paths.
"""

from __future__ import annotations 
import warnings
import scipy.stats
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.axes
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier 
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression 

from ..api.types import Optional, Tuple,  List, Union 
from ..api.types import ArrayLike, DataFrame, Series
from ..api.summary import ReportFactory 
from ..tools.coreutils import is_iterable
from ..tools.coreutils import to_numeric_dtypes, fill_nan_in
from ..tools.validator import get_estimator_name
from ..tools.validator import check_X_y, check_consistent_length
from .utils import _set_sns_style, make_mpl_properties 

__all__=[
  'plot_importances_ranking',
  'plot_rf_feature_importances',
  'plot_feature_interactions',
  'plot_variables',
  'plot_correlation_with_target',
  'plot_dependences',
  'plot_sbs_feature_selection',
  'plot_permutation_importance',
  'plot_regularization_path',   
]

def plot_regularization_path ( 
        X, y , c_range=(-4., 6. ), fig_size=(8, 5), sns_style =False, 
        savefig = None, **kws 
        ): 
    r""" Plot the regularisation path from Logit / LogisticRegression 
    
    Varying the  different regularization strengths and plot the  weight 
    coefficient of the different features for different regularization 
    strength. 
    
    Note that, it is recommended to standardize the data first. 
    
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features. X is expected to be 
        standardized. 

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    c_range: list or tuple [start, stop] 
        Regularization strength list. It is a range from the strong  
        strong ( start) to lower (stop) regularization. Note that 'C' is 
        the inverse of the Logistic Regression regularization parameter 
        :math:`\lambda`. 
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
        
    kws: dict, 
        Additional keywords arguments passed to 
        :class:`sklearn.linear_model.LogisticRegression`
    
    Examples
    --------
    >>> from gofast.plot.feature_analysis import plot_regularization_path 
    >>> from gofast.datasets import fetch_data
    >>> X, y = fetch_data ('bagoue analysed' ) # data aleardy standardized
    >>> plot_regularization_path (X, y ) 

    """
    X, y = check_X_y( X, y,  to_frame= True)
    
    if not is_iterable(c_range): 
        raise TypeError ("'C' regularization strength is a range of C " 
                         " Logit parameter: (start, stop).")
    c_range = sorted (c_range )
    
    if len(c_range) < 2: 
        raise ValueError ("'C' range expects two values [start, stop]")
        
    if len(c_range) >2 : 
        warnings.warn ("'C' range expects two values [start, stop]. Values"
                       f" are shrunk to the first two values: {c_range[:2]} "
                       )
    weights, params = [], []    
    for c in np.arange (*c_range): 
        lr = LogisticRegression(penalty='l1', C= 10.**c, solver ='liblinear', 
                                multi_class='ovr', **kws)
        lr.fit(X,y )
        weights.append (lr.coef_[1])
        params.append(10**c)
        
    weights = np.array(weights ) 
    colors = make_mpl_properties(weights.shape[1])
    if not hasattr (X, 'columns'): 
        flabels =[f'{i:>7}' for i in range (X.shape[1])] 
    else: flabels = X.columns   
    
    # plot
    fig, ax = plt.subplots(figsize = fig_size )
    if sns_style: 
        _set_sns_style (sns_style)

    for column , color in zip( range (weights.shape [1]), colors ): 
        plt.plot (params , weights[:, column], 
                  label =flabels[column], 
                  color = color 
                  )

    plt.axhline ( 0 , color ='black', ls='--', lw= 3 )
    plt.xlim ( [ 10 ** int(c_range[0] -1), 10 ** int(c_range[1]-1) ])
    plt.ylabel ("Weight coefficient")
    plt.xlabel ('C')
    plt.xscale( 'log')
    plt.legend (loc ='upper left',)
    ax.legend(
            loc ='upper right', 
            bbox_to_anchor =(1.38, 1.03 ), 
            ncol = 1 , fancybox =True 
    )
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_permutation_importance(
    importances: ArrayLike, 
    feature_names: List[str], 
    title: str = "Permutation feature importance",
    xlabel: str = "RF importance",
    ylabel: str = "Features", 
    figsize: Tuple[int, int] = (10, 8), 
    color: str = "skyblue", 
    edgecolor: str = "black", 
    savefig: Optional[str] = None
) -> None:
    """
    Plot permutation feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    importances : array-like
        The feature importances, typically obtained from a model 
        or permutation test.
    feature_names : list of str
        The names of the features corresponding to the importances.
    title : str, optional
        Title of the plot. Defaults to "Permutation feature importance".
    xlabel : str, optional
        Label for the x-axis. Defaults to "RF importance".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Features".
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Defaults to (10, 8).
    color : str, optional
        Bar color. Defaults to "skyblue".
    edgecolor : str, optional
        Bar edge color. Defaults to "black".
    savefig : str, optional
        Path to save the figure. If None, the figure is not saved. 
        Defaults to None.
    
    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the plot.
    ax : Axes
        The matplotlib Axes object for the plot.
    
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.feature_analysis import plot_permutation_importance
    >>> importances = np.random.rand(30)
    >>> feature_names = ['Feature {}'.format(i) for i in range(30)]
    >>> plot_permutation_importance(
        importances, feature_names, title="My Plot", xlabel="Importance",
        ylabel="Features", figsize=(8, 10), color="lightblue",
        edgecolor="gray", savefig="importance_plot.png")
    """

    # Sort the feature importances in ascending order for plotting
    sorted_indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(importances)), importances[sorted_indices],
            color=color, edgecolor=edgecolor)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(np.array(feature_names)[sorted_indices])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Optionally save the figure to a file
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()
    return fig, ax

def plot_dependences(
    model: BaseEstimator, 
    X: Union[ArrayLike, DataFrame], 
    features: Union[List[int], List[str]], 
    kind: str = 'average', 
    grid_resolution: int = 100, 
    feature_names: Optional[List[str]] = None, 
    percentiles: Tuple[float, float] = (0.05, 0.95), 
    n_jobs: Optional[int] = None, 
    verbose: int = 0, 
    ax: Optional[matplotlib.axes.Axes] = None
) -> matplotlib.axes.Axes:
    """
    Generates Partial Dependence Plots (PDP) or Individual Conditional 
    Expectation (ICE) plots for specified features using a fitted model.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn-compatible estimator that implements `predict` 
        or `predict_proba`.
    X : Union[np.ndarray, pd.DataFrame]
        The input samples. Pass directly as a Fortran-contiguous NumPy array 
        to avoid unnecessary memory duplication. For pandas DataFrame, 
        ensure binary columns are used.
    features : Union[List[int], List[str]]
        The target features for which to create the PDPs or ICE plots. For 
        `feature_names` provided, `features` can be a list of feature names.
    kind : str, optional
        The kind of plot to generate. 'average' generates the PDP, and 
        'individual' generates the ICE plots. Defaults to 'average'.
    grid_resolution : int, optional
        The number of evenly spaced points where the partial dependence 
        is evaluated. Defaults to 100.
    feature_names : Optional[List[str]], optional
        List of feature names if `X` is a NumPy array. `feature_names` is 
        used for axis labels. Defaults to None.
    percentiles : Tuple[float, float], optional
        The lower and upper percentile used to create the extreme values 
        for the PDP axes. Must be in [0, 1]. Defaults to (0.05, 0.95).
    n_jobs : Optional[int], optional
        The number of jobs to run in parallel for `plot_partial_dependence`. 
        `None` means 1. Defaults to None.
    verbose : int, optional
        Verbosity level. Defaults to 0.
    ax : Optional[matplotlib.axes.Axes], optional
        Matplotlib axes object to plot on. If None, creates a new figure. 
        Defaults to None.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib axes object with the plot.

    Examples
    --------
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from gofast.plot.feature_analysis import plot_dependences
    >>> X, y = make_friedman1()
    >>> model = GradientBoostingRegressor().fit(X, y)
    >>> plot_dependences(model, X, features=[0, 1], kind='average')

    See Also 
    ---------
    sklearn.inspection.PartialDependenceDisplay: 
        Class simplifies generating PDP and ICE plots. 
        
    Note
    ----
    PDP and ICE plots are valuable  tools for understanding the effect of 
    features on the prediction of a model, providing insights into the model's 
    behavior over a range of feature values.
    """
    if not hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
       raise TypeError("The model must implement predict or predict_proba method.")
   
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, len(features) * 4))
    
    disp = PartialDependenceDisplay.from_estimator(
        model, X, features, kind=kind, feature_names=feature_names, 
        percentiles=percentiles, grid_resolution=grid_resolution, 
        n_jobs=n_jobs, verbose=verbose, ax=ax,
        line_kw={'color': 'gray', 'alpha': 0.5} if kind == 'individual' else {},
        pd_line_kw={'color': 'red', 'linestyle': '--'} if kind == 'average' else {}
    )
    
    plot_title = ("Partial Dependence Plots" if kind == 'average' 
                  else "Individual Conditional Expectation Plots"
                  )
    disp.figure_.suptitle(plot_title)
    plt.subplots_adjust(top=0.9)  # Adjust the title to not overlap with plots
    plt.show()
    
    return ax

def plot_sbs_feature_selection(
        sbs_estimator,/, X=None, y=None, fig_size=(8, 5), 
        sns_style=False, savefig=None, verbose=0, **sbs_kws
    ):
    """
    Plot the feature selection process using Sequential Backward Selection (SBS).

    This function visualizes the selection of the best feature subset at each stage 
    in the SBS algorithm. It requires either a fitted SBS estimator or the training
    data (`X` and `y`) to fit the estimator during the plot generation.

    Parameters
    ----------
    sbs_estimator : :class:`~.gofast.transformers.SequentialBackwardSelection`
        The SBS estimator. Can be pre-fitted; if not, `X` and `y` must be provided
        for fitting during the plot generation.

    X : array-like of shape (n_samples, n_features), optional
        Training data, with `n_samples` as the number of samples and `n_features`
        as the number of features. Required if `sbs_estimator` is not pre-fitted.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        Target values corresponding to `X`. Required if `sbs_estimator` is not
        pre-fitted.

    fig_size : tuple of (width, height), default=(8, 5)
        Size of the matplotlib figure, specified as a width and height tuple.

    sns_style : bool, default=False
        If True, apply seaborn styling to the plot.

    savefig : str, optional
        File path where the figure is saved. If provided, the plot is saved
        to this path.

    verbose : int, default=0
        If set to a positive number, print feature labels and their importance
        rates.

    sbs_kws : dict, optional
        Additional keyword arguments passed to the
        :class:`~.gofast.base.SequentialBackwardSelection` class.

    Examples
    --------
    # Example 1: Plotting a pre-fitted SBS
    >>> from sklearn.neighbors import KNeighborsClassifier, train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.transformers import SequentialBackwardSelection
    >>> from gofast.tools.utils import plot_sbs_feature_selection
    >>> X, y = fetch_data('bagoue analysed')  # Data already standardized
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs = SequentialBackwardSelection(knn)
    >>> sbs.fit(X_train, y_train)
    >>> plot_sbs_feature_selection(sbs, sns_style=True)

    # Example 2: Plotting an SBS estimator without pre-fitting
    >>> plot_sbs_feature_selection(knn, X_train, y_train)  # Same result as above
    """

    from ..transformers import SequentialBackwardSelection as SBS 
    if ( 
        not hasattr (sbs_estimator, 'scores_') 
        and not hasattr (sbs_estimator, 'k_score_')
            ): 
        if ( X is None or y is None ) : 
            clfn = get_estimator_name( sbs_estimator)
            raise TypeError (f"When {clfn} is not a fitted "
                             "estimator, X and y are needed."
                             )
        sbs_estimator = SBS(estimator = sbs_estimator, **sbs_kws)
        sbs_estimator.fit(X, y )
        
    k_feat = [len(k) for k in sbs_estimator.subsets_]
    
    if verbose: 
        flabels =None 
        if  ( not hasattr (X, 'columns') and X is not None ): 
            warnings.warn("None columns name is detected."
                          " Created using index ")
            flabels =[f'{i:>7}' for i in range (X.shape[1])]
            
        elif hasattr (X, 'columns'):
            flabels = list(X.columns)  
        elif hasattr ( sbs_estimator , 'feature_names_in'): 
            flabels = sbs_estimator.feature_names_in 
            
        if flabels is not None: 
            k3 = list (sbs_estimator.subsets_[X.shape[1]])
            print("Smallest feature for subset (k=3) ")
            print(flabels [k3])
            
        else : print("No column labels detected. Can't print the "
                     "smallest feature subset.")
        
    if sns_style: 
        _set_sns_style (sns_style)
        
    plt.figure(figsize = fig_size)
    plt.plot (k_feat , sbs_estimator.scores_, marker='o' ) 
    plt.ylim ([min(sbs_estimator.scores_) -.25 ,
               max(sbs_estimator.scores_) +.2 ])
    plt.ylabel (sbs_estimator.scorer_name_ )
    plt.xlabel ('Number of features')
    plt.tight_layout() 
    
    if savefig is not None:
        plt.savefig(savefig )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_variables(
    data: DataFrame, 
    target: Union[Optional[str], ArrayLike] = None,
    kind: str = "cat", 
    colors: Optional[List[str]] = None, 
    target_labels: Optional[Union[List[str], ArrayLike]] = None,
    fontsize: int = 12, 
    ylabel: Optional[str] = None, 
    figsize: Tuple[int, int] = (20, 16)
    ) -> None:
    """
    Plot variables in the dataframe based on their type (categorical or numerical)
    against a target variable.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the variables to plot.
    target : Optional[str], optional
        The name of the target variable. If provided, it must exist in `data`.
        If array is passed, it must be consistent with the data. 
    kind : str, optional
        The kind of variables to plot ('cat' for categorical, 'num' for numerical). 
        Default is 'cat'.
    colors : Optional[List[str]], optional
        A list of colors to use for the plot. If not provided, defaults will be used.
    target_labels : Optional[Union[List[str], np.ndarray]], optional
        Labels for the different values of the target variable. If not provided,
        values will be used as labels.
    fontsize : int, optional
        Font size for labels in the plot. Default is 12.
    ylabel : Optional[str], optional
        Label for the y-axis. If not provided, no label is set.
    figsize : Tuple[int, int], optional
        Figure size for the plot. Default is (20, 16).

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
    ValueError
        If `target` is provided but does not exist in `data`.

    Examples
    --------
    >>> import seaborn as sns
    >>> from gofast.plot.feature_analysis import plot_variables
    >>> df = sns.load_dataset('titanic')
    >>> plot_variables(df, kind='cat', target='survived',
                       colors=['blue', 'red'], target_labels=['Died', 'Survived'], 
                       fontsize=10, ylabel='Count')
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected data to be a pandas DataFrame."
                        f" Got {type(data).__name__!r}")
    if target is None:
        raise ValueError("Target must be provided.")
        
    if isinstance ( target, str): 
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' does not exist in"
                             " the dataframe.")
        target_data = data[target]
        data = data.drop(columns=[target])
    elif isinstance ( target, ( np.ndarray, pd.Series)): 
        target_data = np.array (target).copy() 
    
    check_consistent_length(data, target_data)
    # Determine categorical and numerical variables
    data, num_vars, cat_vars= to_numeric_dtypes(data, return_feature_types=True)

    if colors is None:
        colors = ['royalblue', 'green'] +  list(make_mpl_properties (
            len(np.unique (target_data))))
    
    if kind.lower() == 'cat':
        _plot_categorical_variables(data, cat_vars, target_data, colors,
                                    target_labels, fontsize, ylabel, figsize)
    elif kind.lower() == 'num':
        _plot_numerical_variables(data, num_vars, target_data, colors, 
                                  target_labels, fontsize, ylabel, figsize)
    else:
        raise ValueError(f"Unsupported plot kind '{kind}'. Choose 'cat' for "
                         "categorical or 'num' for numerical variables.")

def _plot_categorical_variables(
    data: DataFrame, cat_vars: List[str], target: Series, 
    colors: Optional[List[str]], target_labels: Optional[Union[List[str], ArrayLike]],
    fontsize: int, ylabel: Optional[str], figsize: Tuple[int, int]) -> None:
    """
    Helper function to plot categorical variables against a target variable.
    """
    if target_labels is None or len(target_labels) != len(np.unique(target)):
        target_labels = [str(val) for val in np.unique(target)]

    plt.figure(figsize=figsize)
    for i, var in enumerate(cat_vars, 1):
        plt.subplot(len(cat_vars)//3+1, 3, i)
        for j, val in enumerate (np.unique(target)): 
            data[target== val][var].hist(
                bins=10, color= colors[j],  
                label=target_labels [j], 
                alpha=0.8 if j%2==0 else 0.5 )
        plt.title(var)
        plt.ylabel(ylabel if ylabel else 'Count')
        plt.xticks(rotation=45)
        plt.legend(labels=target_labels, fontsize=fontsize)
    plt.tight_layout()

def _plot_numerical_variables(
    data: DataFrame, num_vars: List[str], target: Series, 
    colors: Optional[List[str]], target_labels: Optional[Union[List[str], ArrayLike]],
    fontsize: int, ylabel: Optional[str], figsize: Tuple[int, int]) -> None:
    """
    Helper function to plot numerical variables against a target variable.
    """
    if target_labels is None or len(target_labels) != len(np.unique(target)):
        target_labels = [str(val) for val in np.unique(target)]

    plt.figure(figsize=figsize)
    for i, var in enumerate(num_vars, 1):
        plt.subplot(len(num_vars)//2+1, 2, i)
        for j, val in enumerate(np.unique(target)):
            subset = data[target == val]
            plt.hist(subset[var], bins=15, color=colors[j % len(colors)],
                     alpha=0.75, label=str(target_labels[j]))
        plt.title(var)
        plt.xlabel(var)
        plt.ylabel(ylabel if ylabel else 'Frequency')
        plt.legend(fontsize=fontsize)
    plt.tight_layout()


def plot_correlation_with_target(
    data: DataFrame, 
    target: Union[str, Series], 
    kind: str = 'bar', 
    show_grid: bool = True, 
    fig_size: Tuple[int, int] = (20, 8), 
    title: Optional[str] = None, 
    color: Optional[str] = None, 
    sns_style: Optional[str] = None, 
    **kwargs
    ) -> plt.Axes:
    """
    Plot the correlation of each feature in the dataframe with a
    specified target.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the features to be correlated with the target.
    target : Union[str, pd.Series]
        The target feature name (as a string) or a pandas Series object. 
        If a string is provided, it should be a column name in `data`.
    kind : str, optional
        The kind of plot to generate. Default is 'bar'.
    show_grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    fig_size : Tuple[int, int], optional
        The figure size in inches (width, height). Default is (20, 8).
    title : Optional[str], optional
        The title of the plot. If None, defaults to "Correlation with target". 
        Default is None.
    color : Optional[str], optional
        The color for the bars or lines in the plot. If None, defaults to 
        "royalblue". Default is None.
    sns_style : Optional[str], optional
        The seaborn style to apply to the plot. If None, the default seaborn 
        style is used. Default is None.
    **kwargs : dict
        Additional keyword arguments to be passed to the plot function.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object with the plot.

    Examples
    --------
    >>> import seaborn as sns
    >>> from gofast.plot.feature_analysis import plot_correlation_with_target
    >>> df = sns.load_dataset('iris')
    >>> plot_correlation_with_target(df, 'petal_length', kind='bar', 
                                     sns_style='whitegrid', color='green')
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected data to be a pandas DataFrame."
                        f" Got {type(data).__name__!r}")

    if isinstance(target, str):
        if target not in data.columns:
            raise ValueError(
                f"Target column '{target}' does not exist in the dataframe.")
        target_data = data[target]
        data = data.drop(columns=target)
    else:
        target_data = target

    if sns_style:
        sns.set_style(sns_style)

    correlation = data.corrwith(target_data)
    ax = correlation.plot(kind=kind, grid=show_grid, figsize=fig_size,
                          title=title or "Correlation with target",
                          color=color or "royalblue", **kwargs)

    return ax
 
    
def plot_feature_interactions(
    data: DataFrame, /, 
    features: Optional[List[str]] = None, 
    histogram_bins: int = 15, 
    scatter_alpha: float = 0.7,
    corr_round: int = 2,
    plot_color: str = 'skyblue',
    edge_color: str = 'black',
    savefig: Optional[str] = None
) -> plt.Figure:
    """
    Visualizes the interactions (distributions and relationships) among 
    various features in a dataset. 
    
    The visualization includes histograms for distribution of features, 
    scatter plots for pairwise relationships, and Pearson correlation 
    coefficients.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the dataset.
    features : Optional[List[str]]
        A list of feature names to be visualized. If None, all features are used.
    histogram_bins : int, optional
        The number of bins for the histograms. Default is 15.
    scatter_alpha : float, optional
        Alpha blending value for scatter plot, between 0 (transparent) and 
        1 (opaque). Default is 0.7.
    corr_round : int, optional
        The number of decimal places for rounding the correlation coefficient.
        Default is 2.
    plot_color : str, optional
        The color for the plots. Default is 'skyblue'.
    edge_color : str, optional
        The edge color for the histogram bins. Default is 'black'.
    savefig : Optional[str], optional
        The file path to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Example
    -------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.plot.feature_analysis import plot_feature_interactions
    >>> df = pd.DataFrame({
        'Feature1': np.random.randn(100),
        'Feature2': np.random.rand(100),
        'Feature3': np.random.gamma(2., 2., 100)
    })
    >>> fig = plot_feature_interactions(df, histogram_bins=20, scatter_alpha=0.5)
    
    This will create a customized plot with histograms, scatter plots, 
    and correlation coefficients for all features in the DataFrame.
    """
    data = to_numeric_dtypes(data, pop_cat_features=True )
    if features is None:
        features= list( data.columns )
        
    # fill Nan, if exist in data
    data = fill_nan_in(data )
    num_features = len(features)
    
    fig, axs = plt.subplots(num_features, num_features, figsize=(15, 15))

    for i in range(num_features):
        for j in range(num_features):
            if i == j:  # Diagonal - Histogram
                axs[i, j].hist(data[features[i]], bins=histogram_bins,
                               color=plot_color, edgecolor=edge_color)
                axs[i, j].set_title(f'Distribution of {features[i]}')
            elif i < j:  # Upper Triangle - Scatter plot
                axs[i, j].scatter(data[features[j]], data[features[i]],
                                  alpha=scatter_alpha)
                axs[i, j].set_xlabel(features[j])
                axs[i, j].set_ylabel(features[i])
                # Calculate and display Pearson correlation
                corr, _ = scipy.stats.pearsonr(data[features[i]], data[features[j]])
                axs[i, j].annotate(f'ρ = {corr:.{corr_round}f}', xy=(0.1, 0.9),
                                   xycoords='axes fraction')
            else:  # Lower Triangle - Empty
                axs[i, j].axis('off')

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return fig


def plot_rf_feature_importances(
    clf=None, 
    X=None, 
    y=None, 
    importances=None,
    fig_size=(8, 4), 
    savefig=None, 
    n_estimators=500,
    verbose=0, 
    sns_style=None, 
    **kwargs):

    """
    Plot feature importances using either a provided RandomForest classifier,
    another classifier with feature importances, or directly from provided 
    feature importances array.
    
    Parameters
    ----------
    clf : classifier, default=None
        A fitted classifier object that has an attribute `feature_importances_`.
        If None, and `importances` is also None, a RandomForestClassifier will
        be instantiated and fitted.
    X : array-like of shape (n_samples, n_features), default=None
        The training input samples. Required if `clf` is not fitted or if
        `importances` is None.
    y : array-like of shape (n_samples,), default=None
        The target values (class labels) as integers or strings.
    importances : array-like of shape (n_features,), default=None
        Precomputed feature importances. If provided, `clf` will be ignored.
    fig_size : tuple, default=(8, 4)
        Width, height in inches of the figure.
    savefig : str, default=None
        If provided, the plot will be saved to the given path instead of shown.
    n_estimators : int, default=500
        Number of trees in the forest to train if a new RandomForestClassifier
        is created. Ignored if `clf` is provided and fitted.
    verbose : int, default=0
        If greater than 0, the feature importances will be printed to stdout.
    sns_style : str, default=None
        The style of seaborn to apply. See seaborn documentation for valid styles.
    **kwargs : dict
        Additional keyword arguments to pass to the RandomForestClassifier
        constructor, if needed.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.datasets import fetch_data
    >>> from gofast.plot.feature_analysis import plot_rf_feature_importances 
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> clf = RandomForestClassifier(n_estimators=100)
    >>> clf.fit(X, y)
    >>> plot_rf_feature_importances(clf=clf, X=X, verbose=1)
    
    >>> X, y = fetch_data ('bagoue analysed' , return_X_y =True) 
    >>> plot_rf_feature_importances (
        RandomForestClassifier(), X=X, y=y , sns_style=True)

    Notes
    -----
    If both `clf` and `importances` are None, the function will raise an error.
    Ensure that at least one is provided. If `clf` is provided but not fitted,
    and `X` and `y` are provided, it will fit the classifier automatically.
    This function is designed to be flexible in handling both pre-fitted models
    and models that require fitting. Importances are extracted directly from the
    classifier if available; otherwise, it assumes they are provided directly
    through the `importances` parameter.
    """

    # Validate input
    if clf is None and importances is None and X is None  :
        raise ValueError("Either a classifier or precomputed importances"
                         " must be provided.")

    # If importances are not provided, attempt to use the classifier
    if importances is None:
        if clf is None:
            clf = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
        if hasattr(clf, 'feature_importances_') and X is not None:
            importances = clf.feature_importances_
        elif X is not None:
            # Check if y is None and create a dummy y if necessary
            if y is None:
                y = np.zeros(X.shape[0])
            clf.fit(X, y)
            importances = clf.feature_importances_
        else:
            raise ValueError("X and y are needed when classifier is not"
                             " pre-fitted or importances are not provided.")

    # Verify that importances are correctly provided or calculated
    if importances is None:
        raise ValueError(
            "Feature importances could not be computed or were not provided.")

    # Prepare feature labels
    feature_labels = X.columns if hasattr(X, 'columns') else [
        f'Feature {i}' for i in range(X.shape[1])]
    indices = np.argsort(importances)[::-1]

    if verbose:
        ranking_dict ={}
        for i, idx in enumerate(indices):
            ranking_dict[f'{i + 1}) {feature_labels[idx]}'] = f"{importances[idx]:.4f}"
        summary = ReportFactory(title = "Feature ranking").add_recommendations(
            ranking_dict,)
        print(summary )

    # Seaborn style setting
    if sns_style:
        _set_sns_style (sns_style)

    # Plotting
    plt.figure(figsize=fig_size)
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_labels[i] for i in indices],
               rotation=90)
    plt.xlim([-1, len(importances)])
    plt.ylabel('Importance Rate')
    plt.xlabel('Feature Labels')
    plt.tight_layout()

    # Save or show the figure
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
        
def plot_importances_ranking(
    data, /, 
    column=None, 
    kind='barh', 
    color='skyblue',
    fig_size=(8, 4), 
    sns_style=None, 
    savefig=None
    ):
    """
    Plot the ranking of feature importances from various input formats such as
    pandas Series, DataFrame, or a one-dimensional numpy array.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, np.ndarray
        The data containing the feature importances. Can be a pandas Series,
        a DataFrame with one or multiple columns, or a one-dimensional numpy array.
    column : str, int, optional
        Specific column name or index to be used from DataFrame. If not provided,
        the first column is used by default if the DataFrame only has one column.
    kind : str, default='barh'
        The kind of plot to generate. Options include 'bar', 'barh', etc.
    color : str, default='skyblue'
        Color of the plot elements.
    fig_size : tuple, default=(8, 4)
        The figure size in inches (width, height).
    sns_style : str, optional
        The style of seaborn to apply to the plot. See seaborn documentation for
        available styles.
    savefig : str, optional
        Path to save the figure to. If provided, the plot is saved to this path
        and not shown.

    Raises
    ------
    ValueError
        If `data` is not a pandas Series, DataFrame with suitable dimensions, or
        a one-dimensional numpy array. Also raised if the specified `column`
        does not exist within the DataFrame.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.feature_analysis import plot_importances_ranking
    >>> importances = pd.Series([0.1, 0.2, 0.7], index=['Feature 1', 'Feature 2', 'Feature 3'])
    >>> plot_importances_ranking(importances)

    >>> df = pd.DataFrame({
    ...     'A': [0.1, 0.2, 0.3],
    ...     'B': [0.3, 0.2, 0.1]
    ... })
    >>> plot_importances_ranking(df, column='A', sns_style='whitegrid',
                                 savefig='importances.png')

    Notes
    -----
    This function is designed to handle different data input formats by checking the
    type of `data` and processing it appropriately to extract feature importances.
    If `data` is a DataFrame and `column` is not specified, and the DataFrame contains
    multiple columns, a ValueError is raised. This function assumes that feature
    importance data is numeric and will raise an error if non-numeric data is passed.
    """

    if isinstance(data, pd.DataFrame):
        if column is not None:
            if column in data.columns:
                importances = data[column]
            else:
                raise ValueError(f"Column '{column}' not found in DataFrame.")
        elif data.shape[1] == 1:  # One column
            importances = data.squeeze()
        elif data.shape[0] == 1:  # One row
            importances = pd.Series(data.iloc[0].values, index=data.columns)
        elif isinstance ( column, int) and column < len(data.columns): 
            importances = data.iloc[:, column]
        else:
            raise ValueError("DataFrame must have exactly one column"
                             " or a 'column' parameter must be specified.")
            
    elif isinstance(data, pd.Series):
        importances = data
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            importances = pd.Series(data, index=[f'feature_{i+1}' for i in range(data.size)])
        else:
            raise ValueError("Numpy array must be one-dimensional")
    else:
        raise ValueError(
            "Input must be a pandas Series, DataFrame, or one-dimensional numpy array")

    # Ensure the data is numeric
    if not np.issubdtype(importances.dtype, np.number):
        raise ValueError("Importance data must be numeric")

    # Create and display the plot
    plt.figure(figsize=fig_size)
    importances.sort_values(ascending=True).plot(kind=kind, color=color)
    plt.title('Feature Importance Ranking')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    # Annotate bars with their numeric values
    for idx, value in enumerate(importances.sort_values(ascending=True)):
        plt.text(value, idx, f"{value:.4f}")
    
    if sns_style:
        _set_sns_style (sns_style)

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
 























