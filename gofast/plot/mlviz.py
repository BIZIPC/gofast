# -*- coding: utf-8 -*-

from __future__ import annotations 
import warnings 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 
from matplotlib.collections import EllipseCollection

from sklearn.metrics import confusion_matrix, roc_curve,mean_absolute_error 
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error 
from sklearn.model_selection import learning_curve, KFold 
from sklearn.utils import resample
try: 
    from keras.models import Model
except : 
    pass 
from ..api.types import Optional, Tuple, Any, List, Union, Callable, NDArray 
from ..api.types import Dict, ArrayLike, DataFrame, Series
from ..tools.coreutils import is_iterable, make_obj_consistent_if
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import _is_cross_validated, validate_yy, validate_keras_model
from ..tools.validator import assert_xy_in, get_estimator_name, check_is_fitted
from .utils import _set_sns_style, _make_axe_multiple
from .utils import make_plot_colors  
from ._config import PlotConfig 

__all__= [ 
    'plot_confusion_matrices',
    'plot_yb_confusion_matrix', 
    'plot_confusion_matrix', 
    'plot_roc_curves',
    'plot_taylor_diagram',
    'plot_cv',
    'plot_confidence',
    'plot_confidence_ellipse',
    'plot_shap_summary',
    'plot_abc_curve',
    'plot_learning_curves',
    'plot_cost_vs_epochs', 
    'plot_regression_diagnostics', 
    'plot_residuals_vs_leverage', 
    'plot_residuals_vs_fitted', 
    'plot_r2'    
    ]

def plot_taylor_diagram(
    *y_preds: List[ArrayLike], 
    reference: ArrayLike, 
    names: Optional[List[str]] = None, 
    kind: str = "default", 
    fig_size: Optional[tuple] = None
    ) -> None:
    """
    Plots a Taylor Diagram, which is used to graphically summarize 
    how closely a set of predictions match observations. The diagram 
    displays the correlation between each prediction and the 
    observations (reference) as the angular coordinate and the 
    standard deviation as the radial coordinate.

    Parameters
    ----------
    y_preds : variable number of np.ndarray
        Each argument is a one-dimensional array containing the predictions 
        from different models. Each prediction array should be the same 
        length as the reference data array.

    reference : np.ndarray
        A one-dimensional array containing the reference data against 
        which the predictions are compared. This should have the same 
        length as each prediction array.

    names : list of str, optional
        A list of names for each set of predictions. If provided, this list 
        should be the same length as the number of 'y_preds'. If not provided, 
        predictions will be labeled as Prediction 1, Prediction 2, etc.

    kind : str, optional
        Determines the angular coverage of the plot. If "default", the plot 
        spans 180 degrees. If "half_circle", the plot spans 90 degrees.

    fig_size : tuple, optional
        The size of the figure in inches. If not provided, defaults to (10, 8).

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.mlviz import plot_taylor_diagram
    >>> y_preds = [np.random.normal(loc=0, scale=1, size=100),
    ...            np.random.normal(loc=0, scale=1.5, size=100)]
    >>> reference = np.random.normal(loc=0, scale=1, size=100)
    >>> plot_taylor_diagram(*y_preds, reference=reference, names=['Model A', 'Model B'], 
    ...                        kind='half_circle')

    Notes
    -----
    Taylor diagrams provide a visual way of assessing multiple 
    aspects of prediction performance in terms of their ability to 
    reproduce observational data. It's particularly useful in the 
    field of meteorology but can be applied broadly to any 
    predictive models that need comparison to a reference.
    """
    # Ensure all prediction arrays are numpy arrays and one-dimensional
    y_preds = [np.asarray(pred).flatten() for pred in y_preds]
    
    reference = np.asarray ( reference).flatten() 
    # Verify that all predictions and the reference are of consistent length
    assert all(pred.size == reference.size for pred in y_preds), \
        "All predictions and the reference must be of the same length"

    # Calculate statistics for each prediction
    correlations = [np.corrcoef(pred, reference)[0, 1] for pred in y_preds]
    standard_deviations = [np.std(pred) for pred in y_preds]
    reference_std = np.std(reference)

    # Create polar plot
    fig = plt.figure(figsize=fig_size or (10, 8))
    ax1 = fig.add_subplot(111, polar=True)

    # Convert correlations to angular coordinates in radians
    angles = np.arccos(correlations)
    radii = standard_deviations

    # Plot each prediction
    for i, angle in enumerate(angles):
        ax1.plot([angle, angle], [0, radii[i]], 
                 label=f'{names[i]}' if names and i < len(names) else f'Prediction {i+1}')
        ax1.plot(angle, radii[i], 'o')

    # Add reference standard deviation
    ax1.plot([0, 0], [0, reference_std], label='Reference')
    ax1.plot(0, reference_std, 'o')

    # Configure the plot according to 'kind'
    if kind == "half_circle":
        ax1.set_thetamax(90)
    else:  # default kind
        ax1.set_thetamax(180)

    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('W')
    ax1.set_rlabel_position(90)
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Taylor Diagram')

    plt.legend()
    plt.show()
    

def plot_cost_vs_epochs(
    regs: Union[Callable, List[Callable]],
    *,
    X: Optional[np.ndarray] = None, 
    y: Optional[np.ndarray] = None,
    fig_size: Tuple[int, int] = (10, 4),
    marker: str = 'o',
    savefig: Optional[str] = None,
    **kws: Dict[str, Any]
) -> List[plt.Axes]:
    """
    Plots the logarithm of loss or cost against the number of epochs for
    different regression estimators. 
    
    Function checks for precomputed 'cost_', 'loss_', or 'weights_' attributes 
    in the regressors.  If not found, it requires training data (X, y) to 
    calculate the loss.

    Parameters
    ----------
    regs : Callable or list of Callables
        Single or list of regression estimators. Estimators should be already fitted.
    X : np.ndarray, optional
        Feature matrix used for training the models, required if no 'cost_',
        'loss_', or 'weights_' attributes are found.
    y : np.ndarray, optional
        Target vector used for training the models, required if no 'cost_',
        'loss_', or 'weights_' attributes are found.
    fig_size : tuple of int, default (10, 4)
        The size of the figure to be plotted.
    marker : str, default 'o'
        Marker style for the plot points.
    savefig : str, optional
        Path to save the figure to. If None, the figure is shown.
    kws : dict
        Additional keyword arguments passed to `matplotlib.pyplot.plot`.

    Returns
    -------
    List of matplotlib.axes.Axes
        List of Axes objects with the plots.

    Examples
    --------
    >>> import numpy as np 
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> from gofast.plot.mlviz import plot_cost_vs_epochs
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.cost_ = [np.mean((y - reg.predict(X)) ** 2) for _ in range(10)]  # Simulated cost
    >>> plot_cost_vs_epochs([reg])
    
    >>> from gofast.plot.mlviz import plot_cost_vs_epochs
    >>> from gofast.estimators.adaline import AdalineClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> ada1 = AdalineClassifier(n_iter=10, eta=0.01).fit(X, y)
    >>> ada2 = AdalineClassifier(n_iter=10, eta=0.0001).fit(X, y)
    >>> plot_cost_vs_epochs(regs=[ada1, ada2])

    Notes
    -----
    This function assumes that the provided estimators are compatible with
    the scikit-learn estimator interface and have been fitted prior to calling
    this function. If 'cost_', 'loss_', or 'weights_' attributes are not found,
    and no training data (X, y) are provided, a ValueError will be raised.
    The function logs the cost or loss to better handle values spanning several
    orders of magnitude, and adds 1 before taking the logarithm to avoid
    mathematical issues with log(0).
    """
    if not isinstance(regs, list):
        regs = [regs]
    
    # Check if each regressor has a 'cost_' or 'loss_' attribute
    have_cost_loss_or_weights = [
        hasattr(reg, 'cost_') or hasattr(reg, 'weights_') 
        or hasattr (reg, 'loss_')for reg in regs]
    
    # If any regressor lacks 'cost_' or 'loss_', and X, y are not provided, raise error
    if not all(have_cost_loss_or_weights) and (X is None or y is None):
        raise ValueError(
            "All regression models must have a 'cost_', 'loss_' or weights_"
            " attribute or 'X' and 'y' must be provided for fitting."
        )
        
    fig, axs = plt.subplots(nrows=1, ncols=len(regs), figsize=fig_size)
    if len(regs)==1: 
        axs = [axs]
    for ax, reg in zip(axs, regs):
        if hasattr(reg, 'cost_'):
            loss = reg.cost_
        elif hasattr(reg, 'weights_'):
            # Assuming weights are the model parameters and we plot their L1 norm
            loss = [np.sum(np.abs(w)) for w in reg.weights_]
        elif hasattr(reg, 'loss_'):
            loss = reg.loss_
        else:
            # Calculate mean squared error if 'cost_', 'loss_', or 'weights_' is not available
            predictions = reg.predict(X)
            loss = [np.mean((y_val - pred) ** 2) for y_val, pred in zip(y, predictions)]
        
        epochs = range(1, len(loss) + 1)
        ax.plot(epochs, np.log10(np.array(loss) + 1), marker=marker, **kws)  # Safe logging
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Log10 of sum-squared-error plus one")
        eta_value = getattr(reg, 'eta', 'N/A')
        eta_str = f"{float(eta_value):.4f}" if isinstance(eta_value, (float, int)) else 'N/A'
        ax.set_title(f"{reg.__class__.__name__} - Learning rate {eta_str}")
  
    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show()

    return axs

def plot_learning_curves(
    models, 
    X ,
    y, 
    *, 
    cv =None, 
    train_sizes= None, 
    baseline_score =0.4,
    scoring=None, 
    convergence_line =True, 
    fig_size=(20, 6),
    sns_style =None, 
    savefig=None, 
    set_legend=True, 
    subplot_kws=None,
    **kws
    ): 
    """ 
    Horizontally visualization of multiple models learning curves. 
    
    Determines cross-validated training and test scores for different training
    set sizes.
    
    Parameters 
    ----------
    models: list or estimators  
        An estimator instance or not that implements `fit` and `predict` 
        methods which will be cloned for each validation. 
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
   
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        ``cv`` default value if None changed from 3-fold to 4-fold.
        
     train_sizes : array-like of shape (n_ticks,), \
             default=np.linspace(0.1, 1, 50)
         Relative or absolute numbers of training examples that will be used to
         generate the learning curve. If the dtype is float, it is regarded as a
         fraction of the maximum size of the training set (that is determined
         by the selected validation method), i.e. it has to be within (0, 1].
         Otherwise it is interpreted as absolute sizes of the training sets.
         Note that for classification the number of samples usually have to
         be big enough to contain at least one sample from each class.
         
    baseline_score: floatm default=.4 
        base score to start counting in score y-axis  (score)
        
    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        
    convergence_line: bool, default=True 
        display the convergence line or not that indicate the level of bias 
        between the training and validation curve. 
        
    fig_size : tuple (width, height), default =(14, 6)
        the matplotlib figure size given as a tuple of width and height
        
    sns_style: str, optional, 
        the seaborn style . 
        
    set_legend: bool, default=True 
        display legend in each figure. Note the default location of the 
        legend is 'best' from :func:`~matplotlib.Axes.legend`
        
    subplot_kws: dict, default is \
        dict(left=0.0625, right = 0.95, wspace = 0.1) 
        the subplot keywords arguments passed to 
        :func:`matplotlib.subplots_adjust` 
    kws: dict, 
        keyword arguments passed to :func:`sklearn.model_selection.learning_curve`
        
    Examples 
    ---------
    (1) -> plot via a metaestimator already cross-validated. 
    
    >>> import watex # must install watex to get the pretrained model ( pip install watex )
    >>> from watex.models.premodels import p 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.mlviz import plot_learning_curves
    >>> X, y = fetch_data ('bagoue prepared') # yields a sparse matrix 
    >>> # let collect 04 estimators already cross-validated from SVMs
    >>> models = [ p.SVM.linear , p.SVM.rbf , p.SVM.sigmoid , p.SVM.poly ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    (2) -> plot with  multiples models not crossvalidated yet.
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.svm import SVC 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> models =[LogisticRegression(), RandomForestClassifier(), SVC() ,
                 KNeighborsClassifier() ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    """
    if not is_iterable(models): 
        models =[models]
    
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1) 
    train_sizes = train_sizes or np.linspace(0.1, 1, 50)
    cv = cv or 4 
    if ( 
        baseline_score >=1 
        and baseline_score < 0 
        ): 
        raise ValueError ("Score for the base line must be less 1 and "
                          f"greater than 0; got {baseline_score}")
    
    if sns_style: 
        _set_sns_style (sns_style)
        
    mnames = [get_estimator_name(n) for n in models]

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize =fig_size)
    # for consistency, put axes on list when 
    # a single model is provided 
    if not is_iterable(axes): 
        axes =[axes] 
    fig.subplots_adjust(**subplot_kws)

    for k, (model, name) in enumerate(zip(models,  mnames)):
        cmodel = model.best_estimator_  if _is_cross_validated(
            model ) else model 
        ax = list(axes)[k]

        N, train_lc , val_lc = learning_curve(
            cmodel , 
            X, 
            y, 
            train_sizes = np.linspace(0.1, 1, 50),
            cv=cv, 
            scoring=scoring, 
            **kws
            )
        ax.plot(N, np.mean(train_lc, 1), 
                   color ="blue", 
                   label ="train score"
                   )
        ax.plot(N, np.mean(val_lc, 1), 
                   color ="r", 
                   label ="validation score"
                   )
        if convergence_line : 
            ax.hlines(np.mean([train_lc[-1], 
                                  val_lc[-1]]), 
                                 N[0], N[-1], 
                                 color="k", 
                                 linestyle ="--"
                         )
        ax.set_ylim(baseline_score, 1)
        #ax[k].set_xlim (N[0], N[1])
        ax.set_xlabel("training size")
        ax.set_title(name, size=14)
        if set_legend: 
            ax.legend(loc='best')
    # for consistency
    ax = list(axes)[0]
    ax.set_ylabel("score")
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_abc_curve(
    effort: List[float],
    yield_: List[float], 
    title: str = "ABC plot", 
    xlabel: str = "Effort", 
    ylabel: str = "Yield", 
    figsize: Tuple[int, int] = (8, 6),
    abc_line_color: str = "blue", 
    identity_line_color: str = "magenta", 
    uniform_line_color: str = "green",
    abc_linestyle: str = "-", 
    identity_linestyle: str = "--", 
    uniform_linestyle: str = ":",
    linewidth: int = 2,
    legend: bool = True,
    set_annotations: bool = True,
    set_a: Tuple[int, int] = (0, 2),
    set_b: Tuple[int, int] = (0, 0),
    set_c: Tuple[int, str] = (5, '+51 dropped'),
    savefig: Optional[str] = None
) -> None:
    """
    Plot an ABC curve comparing effort vs yield with reference lines for 
    identity and uniform distribution. This visualization helps to 
    assess the effectiveness of different efforts in achieving yields.

    Parameters
    ----------
    effort : List[float]
        The effort values (x-axis).
    yield_ : List[float]
        The yield values (y-axis).
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    figsize : Tuple[int, int], optional
        The figure size in inches.
    abc_line_color : str, optional
        The color of the ABC line.
    identity_line_color : str, optional
        The color of the identity line.
    uniform_line_color : str, optional
        The color of the uniform distribution line.
    abc_linestyle : str, optional
        The linestyle for the ABC line.
    identity_linestyle : str, optional
        The linestyle for the identity line.
    uniform_linestyle : str, optional
        The linestyle for the uniform line.
    linewidth : int, optional
        The width of the lines.
    legend : bool, optional
        Whether to show the legend.
    set_annotations : bool, optional
        Whether to annotate set A, B, C.
    set_a : Tuple[int, int], optional
        The info for set A annotation.
    set_b : Tuple[int, int], optional
        The info for set B annotation.
    set_c : Tuple[int, str], optional
        The info for set C annotation including number and label.
    savefig : Optional[str], optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object for the plot.
 
    
    See Also
    --------
    gofast.tools.mathex.compute_effort_yield: 
        Compute effort and yield values from importance data. 
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_abc_curve
    >>> effort = np.linspace(0, 1, 100)
    >>> yield_ = np.sqrt(effort)  # Non-linear yield
    >>> plot_abc_curve(effort, yield_, title='Effort-Yield Analysis')

    Notes
    -----
    The ABC curve is useful for evaluating the balance between effort
    and yield. Identity line shows perfect balance, while uniform line 
    shows consistent yield regardless of effort.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the curves
    ax.plot(effort, yield_, label='ABC', color=abc_line_color,
            linestyle=abc_linestyle, linewidth=linewidth)
    ax.plot([0, 1], [0, 1], label='Identity', color=identity_line_color,
            linestyle=identity_linestyle, linewidth=linewidth)
    ax.hlines(y=yield_[-1], xmin=0, xmax=1, label='Uniform',
              color=uniform_line_color, linestyle=uniform_linestyle,
              linewidth=linewidth)

    # Annotations for sets
    if set_annotations:
        ax.annotate(f'Set A: n = {set_a[1]}', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=9)
        ax.annotate(f'Set B: n = {set_b[1]}', xy=(0.05, 0.90),
                    xycoords='axes fraction', fontsize=9)
        ax.annotate(f'Set C: n = {set_c[0]} {set_c[1]}', xy=(0.05, 0.85), 
                    xycoords='axes fraction', fontsize=9)

    # Additional plot settings
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if legend:
        ax.legend()

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()
    
    return ax 
    
def plot_confidence(
    y: Optional[Union[str, ArrayLike]] = None, 
    x: Optional[Union[str, ArrayLike]] = None,  
    data: Optional[DataFrame] = None,  
    ci: float = .95,  
    kind: str = 'line', 
    b_samples: int = 1000, 
    **sns_kws: Dict
) -> plt.Axes:
    """
    Plot confidence interval data using a line plot, regression plot, 
    or the bootstrap method.
    
    A Confidence Interval (CI) is an estimate derived from observed data statistics, 
    indicating a range where a population parameter is likely to be found at a 
    specified confidence level. Introduced by Jerzy Neyman in 1937, CI is a crucial 
    concept in statistical inference. Common types include CI for mean, median, 
    the difference between means, a proportion, and the difference in proportions.

    Parameters 
    ----------
    y : Union[np.ndarray, str], optional
        Dependent variable values. If a string, `y` should be a column name in 
        `data`. `data` cannot be None in this case.
    x : Union[np.ndarray, str], optional
        Independent variable values. If a string, `x` should be a column name in 
        `data`. `data` cannot be None in this case.
    data : pd.DataFrame, optional
        Input data structure. Can be a long-form collection of vectors that can be 
        assigned to named variables or a wide-form dataset that will be reshaped.
    ci : float, default=0.95
        The confidence level for the interval.
    kind : str, default='line'
        The type of plot. Options include 'line', 'reg', or 'bootstrap'.
    b_samples : int, default=1000
        The number of bootstrap samples to use for the 'bootstrap' method.
    sns_kws : Dict
        Additional keyword arguments passed to the seaborn plot function.

    Returns 
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.mlviz import plot_confidence 
    >>> df = pd.DataFrame({'x': range(10), 'y': np.random.rand(10)})
    >>> ax = plot_confidence(x='x', y='y', data=df, kind='line', ci=0.95)
    >>> plt.show()
    
    >>> ax = plot_confidence(y='y', data=df, kind='bootstrap', ci=0.95, b_samples=500)
    >>> plt.show()
    """

    plot_functions = {
        'line': lambda: sns.lineplot(data=data, x=x, y=y, errorbar=('ci', ci), **sns_kws),
        'reg': lambda: sns.regplot(data=data, x=x, y=y, ci=ci, **sns_kws)
    }
    if isinstance ( data, dict): 
        data = pd.DataFrame ( data )
        
    x, y = assert_xy_in(x, y, data=data, ignore ="x")
    if kind in plot_functions:
        ax = plot_functions[kind]()
    elif kind.lower().startswith('boot'):
        if y is None:
            raise ValueError("y must be provided for bootstrap method.")
        if not isinstance(b_samples, int):
            raise ValueError("`b_samples` must be an integer.")

        medians = [np.median(resample(y, n_samples=len(y))) for _ in range(b_samples)]
        plt.hist(medians)
        plt.show()
        
        p = ((1.0 - ci) / 2.0) * 100
        lower, upper = np.percentile(medians, [p, 100 - p])
        print(f"{ci*100}% confidence interval between {lower} and {upper}")

        ax = plt.gca()
    else:
        raise ValueError(f"Unrecognized plot kind: {kind}")

    return ax

def plot_confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    data: Optional[DataFrame] = None,
    figsize: Tuple[int, int] = (6, 6),
    scatter_s: int = 0.5,
    line_colors: Tuple[str, str, str] = ('firebrick', 'fuchsia', 'blue'),
    line_styles: Tuple[str, str, str] = ('-', '--', ':'),
    title: str = 'Different Standard Deviations',
    show_legend: bool = True
) -> plt.Axes:
    """
    Plots the confidence ellipse of a two-dimensional dataset.

    This function visualizes the confidence ellipse representing the covariance 
    of the provided 'x' and 'y' variables. The ellipses plotted represent 1, 2, 
    and 3 standard deviations from the mean.

    Parameters
    ----------
    x : Union[str, np.ndarray, pd.Series]
        The x-coordinates of the data points or column name in DataFrame.
    y : Union[str, np.ndarray, pd.Series]
        The y-coordinates of the data points or column name in DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (6, 6).
    scatter_s : int, optional
        The size of the scatter plot markers. Default is 0.5.
    line_colors : Tuple[str, str, str], optional
        The colors of the lines for the 1, 2, and 3 std deviation ellipses.
    line_styles : Tuple[str, str, str], optional
        The line styles for the 1, 2, and 3 std deviation ellipses.
    title : str, optional
        The title of the plot. Default is 'Different Standard Deviations'.
    show_legend : bool, optional
        If True, shows the legend. Default is True.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    Note 
    -----
    The approach that is used to obtain the correct geometry 
    is explained and proved here:
      https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
      
    The method avoids the use of an iterative eigen decomposition 
    algorithm and makes use of the fact that a normalized covariance 
    matrix (composed of pearson correlation coefficients and ones) is 
    particularly easy to handle.
    
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> ax = plot_confidence_ellipse(x, y)
    >>> plt.show()
    """
    x, y= assert_xy_in(x, y, data = data )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.scatter(x, y, s=scatter_s)

    for n_std, color, style in zip([1, 2, 3], line_colors, line_styles):
        confidence_ellipse(x, y, ax, n_std=n_std, label=f'${n_std}\sigma$',
                           edgecolor=color, linestyle=style)

    ax.set_title(title)
    if show_legend:
        ax.legend()
    return ax

def confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    ax: plt.Axes, 
    n_std: float = 3.0, 
    facecolor: str = 'none', 
    data: Optional[DataFrame] = None,
    **kwargs
) -> Ellipse:
    """
    Creates a covariance confidence ellipse of x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays with the same size.
    ax : plt.Axes
        The axes object where the ellipse will be plotted.
    n_std : float, optional
        The number of standard deviations to determine the ellipse's radius. 
        Default is 3.
    facecolor : str, optional
        The color of the ellipse's face. Default is 'none' (no fill).
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.

    **kwargs
        Additional arguments passed to the Ellipse patch.

    Returns
    -------
    ellipse : Ellipse
        The Ellipse object added to the axes.

    Raises
    ------
    ValueError
        If 'x' and 'y' are not of the same size.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> fig, ax = plt.subplots()
    >>> confidence_ellipse(x, y, ax, n_std=2, edgecolor='red')
    >>> ax.scatter(x, y, s=3)
    >>> plt.show()
    """
    x, y = assert_xy_in(x, y, data = data )

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(
        scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_roc_curves (
   clfs, /, 
   X, y, 
   names =..., 
   colors =..., 
   ncols = 3, 
   score=False, 
   kind="inone",
   ax = None,  
   fig_size=( 7, 7), 
   **roc_kws ): 
    """ Quick plot of Receiving Operating Characterisctic (ROC) of fitted models 
    
    Parameters 
    ------------
    clfs: list, 
       list of models for ROC evaluation. Model should be a scikit-learn 
       or  XGBoost estimators 
       
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
    
    y : ndarray or Series of length (n_samples, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
        
    names: list, 
       List of model names. If not given, a raw name of the model is passed 
       instead.
     
    kind: str, default='inone' 
       If ``['individual'|'2'|'single']``, plot each ROC model separately. 
       Any other value, group of ROC curves into a single plot. 
       
       .. versionchanged:: 0.2.5 
          Parameter `all` is deprecated and replaced by `kind`. It henceforth 
          accepts arguments ``allinone|1|grouped`` or ``individual|2|single``
          for plotting mutliple ROC curves in one or separate each ROC curves 
          respecively. 
          
    colors : str, list 
       Colors to specify each model plot. 
       
    ncols: int, default=3 
       Number of plot to be placed inline before skipping to the next column. 
       This is feasible if `many` is set to ``True``. 
       
    score: bool,default=False
      Append the Area Under the curve score to the legend.  
      
    kws: dict,
        keyword argument of :func:`sklearn.metrics.roc_curve 
        
    Return
    -------
    ax: Axes.Subplot. 
    
    Examples 
    --------
    >>> from gofast.tools.utils import plot_roc_curves 
    >>> from sklearn.datasets import make_moons 
    >>> from gofast.exlib import ( train_test_split, KNeighborsClassifier, SVC ,
    XGBClassifier, LogisticRegression ) 
    >>> X, y = make_moons (n_samples=2000, noise=0.2)
    >>> X, Xt, y, yt = train_test_split (X, y, test_size=0.2) 
    >>> clfs = [ m().fit(X, y) for m in ( KNeighborsClassifier, SVC , 
                                         XGBClassifier, LogisticRegression)]
    >>> plot_roc_curves(clfs, Xt, yt)
    Out[66]: <AxesSubplot:xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (FPR)'>
    >>> plot_roc_curves(clfs, Xt, yt,kind='2', ncols = 4 , fig_size = (10, 4))
    """

    kind = '2' if str(kind).lower() in 'individual2single' else '1'

    def plot_roc(model, data, labels, score =False ):
        if hasattr(model, "decision_function"):
            predictions = model.decision_function(data)
        else:
            predictions = model.predict_proba(data)[:,1]
            
        fpr, tpr, _ = roc_curve(labels, predictions, **roc_kws )
        auc_score = None 
        if score: 
            auc_score = roc_auc_score ( labels, predictions,)
            
        return fpr, tpr , auc_score
    
    if not is_iterable ( clfs): 
       clfs = is_iterable ( clfs, exclude_string =True , transform =True ) 
       
    # make default_colors 
    colors = make_plot_colors(clfs, colors = colors )
    # save the name of models 
    names = make_obj_consistent_if (
        names , [ get_estimator_name(m) for m in clfs ]) 

    # check whether the model is fitted 
    if kind=='2': 
        fig, ax = _make_axe_multiple ( 
            clfs, ncols = ncols , ax = ax, fig_size = fig_size 
                                  ) 
    else: 
        if ax is None: 
            fig, ax = plt.subplots (1, 1, figsize = fig_size )  
    
    for k, ( model, name)  in enumerate (zip (clfs, names )): 
        check_is_fitted(model )
        fpr, tpr, auc_score = plot_roc(model, X, y, score)

        if hasattr (ax, '__len__'): 
            if len(ax.shape)>1: 
                i, j  =  k // ncols , k % ncols 
                axe = ax [i, j]
            else: axe = ax[k]
        else: axe = ax 

        axe.plot(fpr, tpr, label=name + ('' if auc_score is None 
                                         else f"AUC={round(auc_score, 3) }") , 
                 color = colors[k]  )
        
        if kind=='2': 
            axe.plot([0, 1], [0, 1], 'k--') 
            axe.legend ()
            axe.set_xlabel ("False Positive Rate (FPR)")
            axe.set_ylabel ("True Positive Rate (FPR)")
        # else: 
        #     ax.plot(fpr, tpr, label=name, color = colors[k])
            
    if kind!='2': 
        ax.plot([0, 1], [0, 1], 'k--') # AUC =.5 
        ax.set_xlabel ("False Positive Rate (FPR)")
        ax.set_ylabel ("True Positive Rate (FPR)")
        ax.legend() 
        
    return ax 

@ensure_pkg (
    "shap",
    extra = ( "The 'shap' package is required for plotting SHAP"
             " summary. Please install it to proceed."), 
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
   )
def plot_shap_summary(
    model: Any, 
    X: Union[ArrayLike, DataFrame], 
    feature_names: Optional[List[str]] = None, 
    plot_type: str = 'dot', 
    color_bar_label: str = 'Feature value', 
    max_display: int = 10, 
    show: bool = True, 
    plot_size: Tuple[int, int] = (15, 10), 
    cmap: str = 'coolwarm'
) -> Optional[Figure]:
    """
    Generate a SHAP (SHapley Additive exPlanations) summary plot for a 
    given model and dataset.

    Parameters
    ----------
    model : model object
        A trained model object that is compatible with SHAP explainer.

    X : array-like or DataFrame
        Input data for which the SHAP values are to be computed. If a DataFrame is
        provided, the feature names are taken from the DataFrame columns.

    feature_names : list, optional
        List of feature names if `X` is an array-like object 
        without feature names.

    plot_type : str, optional
        Type of the plot. Either 'dot' or 'bar'. The default is 'dot'.

    color_bar_label : str, optional
        Label for the color bar. The default is 'Feature value'.

    max_display : int, optional
        Maximum number of features to display on the summary plot. 
        The default is 10.

    show : bool, optional
        Whether to show the plot. The default is True. If False, 
        the function returns the figure object.

    plot_size : tuple, optional
        Size of the plot specified as (width, height). The default is (15, 10).

    cmap : str, optional
        Colormap to use for plotting. The default is 'coolwarm'.

    Returns
    -------
 
    figure : matplotlib Figure object or None
        The figure object if `show` is False, otherwise None.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.datasets import make_classification
    >>> from gofast.plot.mlviz import plot_shap_summary
    >>> X, y = make_classification(n_features=5, random_state=42, return_X_y=True)
    >>> model = RandomForestClassifier().fit(X, y)
    >>> plot_shap_summary(model, X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

    """
    import shap

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Create a summary plot
    plt.figure(figsize=plot_size)
    shap.summary_plot(
        shap_values, X, 
        feature_names=feature_names, 
        plot_type=plot_type,
        color_bar_label=color_bar_label, 
        max_display=max_display, 
        show=False
    )

    # Customize color bar label
    color_bar = plt.gcf().get_axes()[-1]
    color_bar.set_title(color_bar_label)

    # Set colormap if specified
    if cmap:
        plt.set_cmap(cmap)

    # Show or return the figure
    if show:
        plt.show()
    else:
        return plt.gcf()
    
@ensure_pkg ( "yellowbrick", extra = (
    "The 'yellowbrick' package is required to plot the confusion matrix. "
    "You can use the alternative function `~.plot_confusion_matrix` "
    "or install 'yellowbrick' manually if automatic installation is not enabled."
    ),
    auto_install=PlotConfig.install_dependencies ,
    use_conda=PlotConfig.use_conda 
   )
def plot_yb_confusion_matrix (
        clf, Xt, yt, labels = None , encoder = None, savefig =None, 
        fig_size =(6, 6), **kws
        ): 
    """ Confusion matrix plot using the 'yellowbrick' package.  
    
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix().
    A confusion matrix shows each combination of the true and predicted
    classes for a test data set.

    The default color map uses a yellow/orange/red color scale. The user can
    choose between displaying values as the percent of true (cell value
    divided by sum of row) or as direct counts. If percent of true mode is
    selected, 100% accurate predictions are highlighted in green.

    Requires a classification model.
    
    Be sure 'yellowbrick' is installed before using the function, otherwise an 
    ImportError will raise. 
    
    Parameters 
    -----------
    clf : classifier estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.
        
    Xt : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : ndarray or Series of length n
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
    labels : list of str, default: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
          
    Returns 
    --------
    cmo: :class:`yellowbrick.classifier.confusion_matrix.ConfusionMatrix`
        return a yellowbrick confusion matrix object instance. 
    
    Examples 
    --------
    >>> #Import the required models and fetch a an extreme gradient boosting 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.tools.utils import plot_yb_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='xgboost' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ ) # pmo.XGB.best_estimator_
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    
    >>> # now plot the score 
    >>> plot_yb_confusion_matrix (pmo.XGB.best_estimator_, Xt, yt  )
    """

    from yellowbrick.classifier import ConfusionMatrix 
    fig, ax = plt.subplots(figsize = fig_size )
    cmo= ConfusionMatrix (clf, classes=labels, 
                         label_encoder = encoder, **kws
                         )
    cmo.score(Xt, yt)
    cmo.show()

    if savefig is not None: 
        fig.savefig(savefig, dpi =300)

    plt.close () if savefig is not None else plt.show() 
    
    return cmo 

def plot_r2(
    y_true: ArrayLike, 
    y_pred: ArrayLike,
    *, 
    title: Optional[str]=None,  
    xlabel: Optional[str]=None, 
    ylabel: Optional[str]=None,  
    fig_size: Tuple[int, int]=(8, 8),
    scatter_color: str='blue', 
    line_color: str='red', 
    line_style: str='--', 
    annotate: bool=True, 
    ax: Optional[plt.Axes]=None, 
    **r2_score_kws
    ):
    """
    Plot a scatter plot of actual vs. predicted values and annotate 
    the R-squared value to visualize the model's performance.

    This function uses the actual and predicted values to plot a scatter 
    diagram, illustrating how close the predictions are to the actual values.
    It can also plot a line representing perfect predictions for reference 
    and annotate the plot with the R-squared value, providing a visual metric 
    of the model's accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The true target values.
    
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The predicted target values.
        
    title : str, optional
        The title of the plot. If None, defaults to 'Model
        Performance: Actual vs Predicted'.
        
    xlabel : str, optional
        The label for the x-axis. If None, defaults to 'Actual Values'.
        
    ylabel : str, optional
        The label for the y-axis. If None, defaults to 'Predicted Values'.
        
    fig_size : tuple, optional
        The size of the figure in inches. Defaults to (8, 8).
        
    scatter_color : str, optional
        The color of the scatter plot points. Defaults to 'blue'.
        
    line_color : str, optional
        The color of the line representing perfect predictions.
        Defaults to 'red'.
        
    line_style : str, optional
        The style of the line representing perfect predictions.
        Defaults to '--'.
        
    annotate : bool, optional
        If True, annotates the plot with the R-squared value. 
        Defaults to True.
        
    ax : matplotlib.axes.Axes, optional
        The axes upon which to draw the plot. If None, a new figure
        and axes are created.
        
    **r2_score_kws : dict, optional
        Additional keyword arguments to be passed to `sklearn.metrics.r2_score`.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> import matplotlib.pyplot as plt

    # Generating synthetic data
    >>> X, y = make_regression(n_samples=100, n_features=1, 
                               noise=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Fitting a linear model
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)

    # Plotting R-squared performance
    >>> plot_r2(y_test, y_pred, title='Linear Regression Performance',
                xlabel='Actual', ylabel='Predicted', scatter_color='green', 
                line_color='orange', line_style='-.')

    # Integrating with existing matplotlib figure and axes
    >>> fig, ax = plt.subplots()
    >>> plot_r2(y_test, y_pred, ax=ax)
    >>> plt.show()

    This function is versatile and can be used directly within data science 
    and machine learning workflows to visually assess model performance.
    """
    y_true, y_pred= validate_yy(y_true, y_pred, "continuous")
    if ax is None: 
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate R-squared value
    r_squared = r2_score(y_true, y_pred, **r2_score_kws)
    
    # Plot actual vs predicted values
    ax.scatter(y_true, y_pred, color=scatter_color, label='Predictions vs Actual data')
    
    # Plot a line representing perfect predictions
    perfect_preds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(perfect_preds, perfect_preds, color=line_color,
            linestyle=line_style, label='Perfect fit')
    
    # Annotate the R-squared value on the plot if requested
    if annotate:
        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.2f}$', 
                fontsize=12, ha='left', va='top', transform=ax.transAxes)
    
    # Enhancing the plot
    ax.set_xlabel(xlabel or 'Actual Values')
    ax.set_ylabel(ylabel or 'Predicted Values')
    ax.set_title(title or 'Model Performance: Actual vs Predicted')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Show the plot only if `ax` was not provided
    if ax is None:
        plt.show()
        
    return ax 

def plot_confusion_matrices (
    clfs, X: NDArray, y: ArrayLike, *,  
    annot: bool =True, 
    pkg: Optional[str]=None, 
    normalize: str='true', 
    sample_weight: Optional[ArrayLike]=None,
    encoder: Optional[Callable[[], ...]]=None, 
    fig_size: Tuple [int, int] = (22, 6),
    savefig:Optional[str] =None, 
    subplot_kws: Optional[Dict[Any, Any]]=None,
    **scorer_kws
    ):
    """ 
    Plot inline multiple model confusion matrices using either the sckitlearn 
    or 'yellowbrick'
    
    Parameters 
    -----------
    clfs : list of classifier estimators
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. Note that the classifier 
        must be fitted beforehand.
        
    X : ndarray or DataFrame of shape (M X N)
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    y : ndarray of shape (M, ) or Series oF length (M, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
    
    pkg: str, optional , default ='sklearn'
        the library to handle the plot. It could be 'yellowbrick'. The basic 
        confusion matrix is handled by the scikit-learn package. 

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
        
    annot: bool, default=True 
        Annotate the number of samples (right or wrong prediction ) in the plot. 
        Set ``False`` to mute the display. 
    
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
        
    Examples
    ----------
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models.premodels import p
    >>> from gofast.tools.utils import plot_confusion_matrices 
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'), test_size =.25  )  
    >>> # compose the models 
    >>> # from RBF, and poly 
    >>> models =[ p.SVM.rbf.best_estimator_,
             p.LogisticRegression.best_estimator_,
             p.RandomForest.best_estimator_ 
             ]
    >>> models 
    [SVC(C=2.0, coef0=0, degree=1, gamma=0.125), LogisticRegression(), 
     RandomForestClassifier(criterion='entropy', max_depth=16, n_estimators=350)]
    >>> # now fit all estimators 
    >>> fitted_models = [model.fit(X, y) for model in models ]
    >>> plot_confusion_matrices(fitted_models , Xt, yt)
    """
    pkg = pkg or 'sklearn'
    pkg= str(pkg).lower() 
    assert pkg in {"sklearn", "scikit-learn", 'yellowbrick', "yb"}, (
        f" Accepts only 'sklearn' or 'yellowbrick' packages, got {pkg!r}") 
    
    if not is_iterable( clfs): 
        clfs =[clfs]

    model_names = [get_estimator_name(name) for name in clfs ]
    # create a figure 
    subplot_kws = subplot_kws or dict (left=0.0625, right = 0.95, 
                                       wspace = 0.12)
    fig, axes = plt.subplots(1, len(clfs), figsize =(22, 6))
    fig.subplots_adjust(**subplot_kws)
    if not is_iterable(axes): 
       axes =[axes] 
    for kk, (model , mname) in enumerate(zip(clfs, model_names )): 
        ypred = model.predict(X)
        if pkg in ('sklearn', 'scikit-learn'): 
            plot_confusion_matrix(y, ypred, annot =annot , ax = axes[kk], 
                normalize= normalize , sample_weight= sample_weight ) 
            axes[kk].set_title (mname)
            
        elif pkg in ('yellowbrick', 'yb'):
            plot_yb_confusion_matrix(
                model, X, y, ax=axes[kk], encoder =encoder )
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_confusion_matrix(
    y_true: Union[ArrayLike, Series],
    y_pred: Union[ArrayLike, Series],
    view: bool = True,
    ax: Optional[plt.Axes] = None,
    annot: bool = True,
    **kws: Dict[str, Any]
) -> ArrayLike:
    """
    Plot a confusion matrix for a single classifier model to evaluate 
    the accuracy of a classification.

    Parameters
    ----------
    y_true : ndarray or Series
        An array or series of true target or class labels, representing 
        the actual classification outcomes.
    y_pred : ndarray or Series
        An array or series of predicted target or class labels, 
        as determined by the classifier.
    view : bool, optional
        If True, display the confusion matrix using matplotlib's imshow;
        otherwise, do not display. Default is True.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes 
        object is created. Default is None.
    annot : bool, optional
        If True, annotate the heat map with the number of samples in 
        each category. Default is True.
    kws : dict
        Additional keyword arguments to pass to 
        `sklearn.metrics.confusion_matrix`.

    Returns
    -------
    mat : ndarray
        The confusion matrix array.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.metrics import plot_confusion_matrix
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> model = AdaBoostClassifier()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> plot_confusion_matrix(y_test, y_pred)
    
    >>> #Import the required models and fetch a an Ababoost model 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.tools.utils import plot_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='ada' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ )
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    >>> # now plot the score 
    >>> plot_confusion_matrix (yt , ypred )
    

    Notes
    -----
    Confusion matrices are a crucial part of understanding the performance 
    of classifiers, not just in terms of overall accuracy but in recognizing 
    where misclassifications are occurring. This function leverages seaborn's 
    heatmap function to visualize the matrix effectively.
    """
    # Ensure y_true and y_pred are of consistent length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length.")

    # Compute the confusion matrix
    mat = confusion_matrix(y_true, y_pred, **kws)

    # Plot the confusion matrix if 'view' is True
    if view:
        if ax is None:
            fig, ax = plt.subplots()
        sns.heatmap(mat, square=True, annot=annot, cbar=False, fmt="d", ax=ax)
        ax.set_xlabel('True Labels')
        ax.set_ylabel('Predicted Labels')
        ax.set_title('Confusion Matrix')
        plt.show()

    return mat

def plot_cv(
    model_fn: Callable[[], Model],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 10,
    metric: str = 'accuracy'
) -> None:
    """
    Performs cross-validation and plots the performance metric across
    epochs for each fold. This function is particularly useful for 
    evaluating the consistency and stability of neural network models
    across different subsets of data.

    Parameters
    ----------
    model_fn : Callable[[], Model]
        A function that returns a compiled neural network model. The
        function should not take any arguments and must return a compiled
        Keras model.

    X : np.ndarray
        Training features, typically an array of shape (n_samples, n_features).

    y : np.ndarray
        Training labels or target values, typically an array of shape
        (n_samples,) or (n_samples, n_outputs).

    n_splits : int, optional
        Number of splits for the K-Fold cross-validation. Default is 5.

    epochs : int, optional
        Number of epochs for training each model during the cross-validation.
        Default is 10.

    metric : str, optional
        The performance metric to plot. Common choices are 'accuracy', 'loss',
        or any other metric included in the compiled model. Default is 'accuracy'.

    Examples
    --------
    >>> from keras.models import Sequential
    >>> from keras.layers import Dense
    >>> from gofast.plot.mlviz import plot_cv 
    >>> def create_model():
    ...     model = Sequential([
    ...         Dense(10, activation='relu', input_shape=(10,)),
    ...         Dense(1, activation='sigmoid')
    ...     ])
    ...     model.compile(optimizer='adam', loss='binary_crossentropy',
    ...                   metrics=['accuracy'])
    ...     return model
    ...
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(2, size=(100,))
    >>> plot_cv(create_model, X, y, n_splits=3, epochs=5)

    Notes
    -----
    This function utilizes KFold from sklearn to create training and 
    validation splits. It is essential that the model function provided 
    compiles the model with the necessary metrics as they are used to 
    monitor training performance.
    """
    kf = KFold(n_splits=n_splits)
    fold_performance = []
    validate_keras_model(model_fn, raise_exception= True )
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create a new instance of the model
        model = model_fn()

        # Train the model
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, verbose=0)

        # Store the metrics for this fold
        fold_performance.append(history.history[metric])

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, performance in enumerate(fold_performance, 1):
        plt.plot(performance, label=f'Fold {i}')

    plt.title(f'Cross-Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

def plot_actual_vs_predicted(
    y_true: ArrayLike, 
    y_pred: ArrayLike, 
    metrics: Optional[List[str]] = None, 
    point_color: str = 'blue', 
    line_color: str = 'red', 
    point_label: str = 'Prediction', 
    line_label: str = 'Ideal', 
    xlabel: str = 'Actual (true) value', 
    ylabel: str = 'Predicted values', 
    title: str = 'Actual (true) value vs predicted values', 
    show_grid: bool = True, 
    grid_style: str = '-', 
    point_size: float = 50, 
    line_style: str = '-', 
    ax: Optional[plt.Axes] = None, 
    fig_size: Optional[Tuple[int, int]] = (10, 8),
    **metrics_kws
) -> plt.Axes:
    """
    Plot a scatter graph of actual vs predicted values along with an ideal line 
    representing perfect predictions.
    
    Optionally calculate and display selected  metrics such as 
    MSE, RMSE, MAE, and R2.

    Parameters
    ----------
    y_true : ArrayLike
        The true values for comparison. Must be a one-dimensional array-like 
        object of numerical data.
    y_pred : ArrayLike
        The predicted values to be plotted against the true values. Must be 
        a one-dimensional array-like object of numerical data.
    metrics : Optional[List[str]], optional
        A list of strings indicating which metrics to calculate and display on 
        the plot. Possible values are 'mse', 'rmse', 'mae', and 'r2'. If None,
        no metrics are displayed. If ``**``, displays all metric values.
    point_color : str, optional
        The color for the scatter plot points. Default is 'blue'.
    line_color : str, optional
        The color for the ideal line. Default is 'red'.
    point_label : str, optional
        The label for the scatter plot points in the legend. Default is 'Prediction'.
    line_label : str, optional
        The label for the ideal line in the legend. Default is 'Ideal'.
    xlabel : str, optional
        The label for the X-axis. Default is 'Actual (true) value'.
    ylabel : str, optional
        The label for the Y-axis. Default is 'Predicted values'.
    title : str, optional
        The title of the plot. Default is 'Actual (true) value vs predicted values'.
    show_grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    grid_style : str, optional
        The style of the grid lines. Default is '-' (solid line).
    point_size : float, optional
        The size of the scatter plot points. Default is 50.
    line_style : str, optional
        The style of the ideal line. Default is '-' (solid line).
    ax : Optional[plt.Axes], optional
        The matplotlib Axes object to draw the plot on. If None, a new figure 
        and axes are created. Default is None.
    fig_size : Optional[Tuple[int, int]], optional
        The size of the figure in inches. Default is (10, 8).
    **metrics_kws
        Additional keyword arguments to pass to the metric functions.

    Returns
    -------
    ax : plt.Axes
        The matplotlib Axes object with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.mlviz import plot_actual_vs_predicted
    >>> y_true = np.array([1., 2., 3., 4., 5.])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    >>> plot_actual_vs_predicted(y_true, y_pred, metrics=['mse', 'rmse'], 
    ...                          point_color='green', line_color='r') 
    """
    # Validate inputs
    y_true, y_pred = validate_yy(y_true, y_pred, "continuous")

    if ax is None:
        plt.figure(figsize=fig_size)
        ax = plt.gca()

    # Plotting
    ax.scatter(y_true, y_pred, color=point_color, label=point_label,
               s=point_size, alpha=0.7)
    ideal_line = np.linspace(min(y_true), max(y_true), 100)
    ax.plot(ideal_line, ideal_line, color=line_color, label=line_label,
            linestyle=line_style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_grid:
        ax.grid(visible=show_grid, linestyle=grid_style)
  
    # Calculate and display metrics
    if isinstance (metrics, str): 
        metrics = [metrics]
    metrics_text = ""
    if metrics is not None:
        available_metrics = {
            'mse': mean_squared_error(y_true, y_pred, **metrics_kws),
            'mae': mean_absolute_error(y_true, y_pred, **metrics_kws),
            'r2': r2_score(y_true, y_pred, **metrics_kws)
        }
        available_metrics['rmse'] = np.sqrt(available_metrics['mse'])
        if metrics=='*': 
            metrics = list(available_metrics.keys())
        metrics = [ metric.lower().replace("_score", "") for metric in metrics]
        
        valid_metrics = list(set(metrics).intersection(set(available_metrics)))  
        if len(valid_metrics)!=0: 
            metrics_text = "\n".join(
                f"${name.upper()} = {value:.3f}$" for name, value in available_metrics.items() 
                if name in valid_metrics)
            metrics_box = ax.text(
                0.05, 0.95, metrics_text,  ha='left', va='top',
                transform=ax.transAxes, fontsize=9, 
                                  bbox=dict(boxstyle="round,pad=0.5",
                                            facecolor='white', 
                                            edgecolor='black', 
                                            alpha=0.7)
                    )
    
            # Dynamically place the legend
            legend_loc = 'best' if metrics_box.get_bbox_patch(
                ).get_extents().y0 > 0.5 else 'lower right'
            ax.legend(loc=legend_loc)
        else: 
            warnings.warn(f"Metric '{metrics}' is not recognized. "
                          f"Available metrics are {list(available_metrics.keys())}.", 
                          UserWarning)
    
    plt.tight_layout()
    plt.show()

    return ax

def plot_regression_diagnostics(
    x: ArrayLike,
    *ys: List[ArrayLike],
    titles: List[str]=None,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    figsize: Tuple[int, int] = (15, 5),
    ci: Optional[int] = 95, 
    **reg_kws
) -> plt.Figure:
    """
    Creates a series of plots to diagnose linear regression fits.

    Parameters
    ----------
    x : np.ndarray
        The independent variable data.
    ys : List[np.ndarray]
        A list of dependent variable datasets to be plotted against x.
    titles : List[str], optional 
        Titles for each subplot.
    xlabel : str, default='X'
        Label for the x-axis.
    ylabel : str, default='Y'
        Label for the y-axis.
    figsize : Tuple[int, int], default=(15, 5)
        Size of the entire figure.
    ci : Optional[int], default=95
        Size of the confidence interval for the regression estimate.
    reg_kws: dict, 
        Additional parameters passed to `seaborn.regplot`. 
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_regression_diagnostics
    >>> x = np.linspace(160, 170, 100)
    >>> # Homoscedastic noise
    >>> y1 = 50 + 0.6 * x + np.random.normal(size=x.size)  
    >>> # Heteroscedastic noise
    >>> y2 = y1 * np.exp(np.random.normal(scale=0.05, size=x.size))  
    >>> # Larger noise variance
    >>> y3 = y1 + np.random.normal(scale=0.5, size=x.size)  
    >>> titles = ['All assumptions satisfied', 'Nonlinear term in model',
                  'Heteroscedastic noise']
    >>> fig = plot_regression_diagnostics(x, y1, y2, y3, titles)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, len(ys), figsize=figsize, sharey=True)
    if len(ys)==1: 
        axes = [axes]
    default_titles = [ None for _ in range ( len(ys)) ]
    if isinstance (titles, str): 
        titles =[titles]
    if titles is not None: 
        # ensure titles met the length of y 
        titles = list(titles ) + default_titles
    else: titles = default_titles 
        
    for i, ax in enumerate(axes):
        sns.regplot(x=x, y=ys[i], ax=ax, ci=ci, **reg_kws)
        
        if titles[i] is not None: 
            ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')

    plt.tight_layout()
    

def plot_residuals_vs_leverage(
    residuals: ArrayLike,
    leverage: ArrayLike,
    cook_d: Optional[ArrayLike] = None,
    figsize: Tuple[int, int] = (8, 6),
    cook_d_threshold: float = 0.5,
    annotate: bool = True,
    scatter_kwargs: Optional[dict] = None,
    cook_d_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    annotation_kwargs: Optional[dict] = None
) -> plt.Axes:
    """
    Plots standardized residuals against leverage with Cook's 
    distance contours.

    Parameters
    ----------
    residuals : np.ndarray
        Standardized residuals from the regression model.
    leverage : np.ndarray
        Leverage values calculated from the model.
    cook_d : np.ndarray, optional
        Cook's distance for each observation in the model.
    figsize : Tuple[int, int], default=(8, 6)
        The figure size for the plot.
    cook_d_threshold : float, default=0.5
        The threshold for Cook's distance to draw a contour and potentially 
        annotate points.
    annotate : bool, default=True
        If True, annotate points that exceed the Cook's distance threshold.
    scatter_kwargs : dict, optional
        Additional keyword arguments for the scatter plot.
    cook_d_kwargs : dict, optional
        Additional keyword arguments for the Cook's distance ellipses.
    line_kwargs : dict, optional
        Additional keyword arguments for the horizontal line at 0.
    annotation_kwargs : dict, optional
        Additional keyword arguments for the annotations.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_residuals_vs_leverage
    >>> residuals = np.random.normal(0, 1, 100)
    >>> leverage = np.random.uniform(0, 0.2, 100)
    >>> # Randomly generated for example purposes
    >>> cook_d = np.random.uniform(0, 1, 100) ** 2  
    >>> ax = plot_residuals_vs_leverage(residuals, leverage, cook_d)
    >>> plt.show()
    """
    if scatter_kwargs is None:
        scatter_kwargs = {'edgecolors': 'k', 'facecolors': 'none'}
    if cook_d_kwargs is None:
        cook_d_kwargs = {'cmap': 'RdYlBu_r'}
    if line_kwargs is None:
        line_kwargs = {'linestyle': '--', 'color': 'grey', 'linewidth': 1}
    if annotation_kwargs is None:
        annotation_kwargs = {'textcoords': 'offset points', 
                             'xytext': (5,5), 'ha': 'right'}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(leverage, residuals, **scatter_kwargs)
    
    # Add Cook's distance contour if provided
    if cook_d is not None:
        levels = [cook_d_threshold, np.max(cook_d)]
        cook_d_collection = EllipseCollection(
            widths=2 * leverage, heights=2 * np.sqrt(cook_d), 
            angles=0, units='x',
            offsets=np.column_stack([leverage, residuals]),
            transOffset=ax.transData,
            **cook_d_kwargs
        )
        cook_d_collection.set_array(cook_d)
        cook_d_collection.set_clim(*levels)
        ax.add_collection(cook_d_collection)
        fig.colorbar(cook_d_collection, ax=ax, orientation='vertical')

    # Draw the horizontal line at 0 for residuals
    ax.axhline(y=0, **line_kwargs)
    
    # Annotate points with Cook's distance above threshold
    if annotate and cook_d is not None:
        for i, (xi, yi, ci) in enumerate(zip(leverage, residuals, cook_d)):
            if ci > cook_d_threshold:
                ax.annotate(f"{i}", (xi, yi), **annotation_kwargs)

    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    plt.show()
    
    return ax

def plot_residuals_vs_fitted(
    fitted_values: ArrayLike, 
    residuals: ArrayLike, 
    highlight: Optional[Tuple[int, ...]] = None, 
    figsize: Tuple[int, int] = (6, 4),
    title: str = 'Residuals vs Fitted',
    xlabel: str = 'Fitted values',
    ylabel: str = 'Residuals',
    linecolor: str = 'red',
    linestyle: str = '-',
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None
) -> plt.Axes:
    """
    Creates a residuals vs fitted values plot.
    
    Function is commonly used in regression analysis to assess the fit of a 
    model. It shows if the residuals have non-linear patterns that could 
    suggest non-linearity in the data or problems with the model.

    Parameters
    ----------
    fitted_values : np.ndarray
        The fitted values from a regression model.
    residuals : np.ndarray
        The residuals from a regression model.
    highlight : Tuple[int, ...], optional
        Indices of points to highlight in the plot.
    figsize : Tuple[int, int], default=(6, 4)
        Size of the figure to be created.
    title : str, default='Residuals vs Fitted'
        Title of the plot.
    xlabel : str, default='Fitted values'
        Label of the x-axis.
    ylabel : str, default='Residuals'
        Label of the y-axis.
    linecolor : str, default='red'
        Color of the line to be plotted.
    linestyle : str, default='-'
        Style of the line to be plotted.
    scatter_kws : dict, optional
        Additional keyword arguments to be passed to the `plt.scatter` method.
    line_kws : dict, optional
        Additional keyword arguments to be passed to the `plt.plot` method.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    See Also 
    ---------
    gofast.tools.mathex.calculate_residuals: 
        Calculate the residuals for regression, binary, or multiclass 
        classification tasks.
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_residuals_vs_fitted
    >>> fitted = np.linspace(0, 100, 100)
    >>> residuals = np.random.normal(0, 10, 100)
    >>> ax = plot_residuals_vs_fitted(fitted, residuals)
    >>> plt.show()
    """
    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(fitted_values, residuals, **scatter_kws)
    
    if highlight:
        ax.scatter(fitted_values[list(highlight)], 
                   residuals[list(highlight)], color='orange')

    sns.regplot(x=fitted_values, y=residuals, lowess=True, ax=ax,
                line_kws={'color': linecolor, 'linestyle': linestyle, **line_kws})

    # Annotate highlighted points
    if highlight:
        for i in highlight:
            ax.annotate(i, (fitted_values[i], residuals[i]))

    ax.axhline(0, color='grey', lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax










