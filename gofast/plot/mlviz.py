# -*- coding: utf-8 -*-
"""
The `mlviz` module provides a variety of visualization tools 
for plotting confusion matrices, ROC curves, learning curves, regression 
diagnostics, SHAP summaries, and more, facilitating comprehensive model 
performance analysis.
"""

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

from sklearn.base import BaseEstimator 
from sklearn.metrics import confusion_matrix, roc_curve,mean_absolute_error 
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error 
from sklearn.model_selection import learning_curve, KFold 
from sklearn.utils import resample
try: 
    from keras.models import Model
except : 
    pass 
from ..api.types import Optional, Tuple, Any, List, Union, Callable, NDArray 
from ..api.types import Dict, ArrayLike, DataFrame, Series, SparseMatrix
from ..tools.coreutils import is_iterable, make_obj_consistent_if
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import _is_cross_validated, validate_yy, validate_keras_model
from ..tools.validator import assert_xy_in, get_estimator_name, check_is_fitted
from .utils import _set_sns_style, _make_axe_multiple
from .utils import make_plot_colors  
from ._config import PlotConfig 


__all__= [ 
    'plot_confusion_matrices',
    'plot_confusion_matrix2', 
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
    observations (`reference`) as the angular coordinate and the 
    standard deviation as the radial coordinate.

    Parameters
    ----------
    y_preds : variable number of `np.ndarray`
        Each argument is a one-dimensional array containing the 
        predictions from different models. Each prediction array 
        should be the same length as the `reference` data array.

    reference : `np.ndarray`
        A one-dimensional array containing the reference data against 
        which the predictions are compared. This should have the same 
        length as each prediction array.

    names : list of `str`, optional
        A list of names for each set of predictions. If provided, this 
        list should be the same length as the number of ``y_preds``. 
        If not provided, predictions will be labeled as Prediction 1, 
        Prediction 2, etc.

    kind : `str`, optional
        Determines the angular coverage of the plot. If "default", the 
        plot spans 180 degrees. If "half_circle", the plot spans 90 
        degrees.

    fig_size : `tuple`, optional
        The size of the figure in inches. If not provided, defaults to 
        (10, 8).

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.mlviz import plot_taylor_diagram
    >>> y_preds = [np.random.normal(loc=0, scale=1, size=100),
    ...            np.random.normal(loc=0, scale=1.5, size=100)]
    >>> reference = np.random.normal(loc=0, scale=1, size=100)
    >>> plot_taylor_diagram(*y_preds, reference=reference, 
    ...                        names=['Model A', 'Model B'], 
    ...                        kind='half_circle')

    Notes
    -----
    Taylor diagrams provide a visual way of assessing multiple 
    aspects of prediction performance in terms of their ability to 
    reproduce observational data. It's particularly useful in the 
    field of meteorology but can be applied broadly to any predictive 
    models that need comparison to a reference.

    The angular coordinate on the Taylor Diagram represents the 
    correlation coefficient :math:`R` between each prediction and the 
    reference, calculated as:

    .. math::
        R = \frac{\sum_{i=1}^n (y_i - \bar{y})(x_i - \bar{x})}
        {\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2} 
        \sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}}

    where :math:`y_i` are the predictions, :math:`x_i` are the 
    observations, and :math:`\bar{y}` and :math:`\bar{x}` are the 
    means of the predictions and observations, respectively.

    The radial coordinate represents the standard deviation :math:`\sigma`
    of the predictions, calculated as:

    .. math::
        \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \bar{y})^2}

    See Also
    --------
    - [1] K. P. Taylor, "Summarizing multiple aspects of model performance 
      in a single diagram," Journal of Geophysical Research, vol. 106, 
      no. D7, pp. 7183-7192, 2001.

    References
    ----------
    .. [1] K. P. Taylor, "Summarizing multiple aspects of model performance 
       in a single diagram," Journal of Geophysical Research, vol. 106, 
       no. D7, pp. 7183-7192, 2001.
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
    in the regressors.  If not found, it requires training data (`X`, `y`) to 
    calculate the loss.

    Parameters
    ----------
    regs : Callable or list of `Callable`
        Single or list of regression estimators. Estimators should be already 
        fitted.
    X : `np.ndarray`, optional
        Feature matrix used for training the models, required if no 'cost_', 
        'loss_', or 'weights_' attributes are found.
    y : `np.ndarray`, optional
        Target vector used for training the models, required if no 'cost_', 
        'loss_', or 'weights_' attributes are found.
    fig_size : `tuple` of `int`, default (10, 4)
        The size of the figure to be plotted.
    marker : `str`, default 'o'
        Marker style for the plot points.
    savefig : `str`, optional
        Path to save the figure to. If None, the figure is shown.
    kws : `dict`
        Additional keyword arguments passed to `matplotlib.pyplot.plot`.

    Returns
    -------
    `List` of `matplotlib.axes.Axes`
        List of `Axes` objects with the plots.

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
    the scikit-learn estimator interface and have been fitted prior to 
    calling this function. If 'cost_', 'loss_', or 'weights_' attributes are 
    not found, and no training data (`X`, `y`) are provided, a ValueError will 
    be raised.
    
    The function logs the cost or loss to better handle values spanning several 
    orders of magnitude, and adds 1 before taking the logarithm to avoid 
    mathematical issues with log(0).

    The logarithm of the loss or cost is calculated as:

    .. math::
        \log_{10}(L + 1)

    where :math:`L` is the loss or cost value at each epoch.

    See Also
    --------
    - [1] R. O. Duda, P. E. Hart, and D. G. Stork, "Pattern Classification," 
      2nd edition, Wiley, 2000.

    References
    ----------
    .. [1] R. O. Duda, P. E. Hart, and D. G. Stork, "Pattern Classification," 
       2nd edition, Wiley, 2000.
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
    models: Union[BaseEstimator, List[BaseEstimator]], 
    X: np.ndarray,
    y: np.ndarray, 
    *, 
    cv: Optional[Union[int, Callable]] = None, 
    train_sizes: Optional[np.ndarray] = None, 
    baseline_score: float = 0.4,
    scoring: Optional[Union[str, Callable]] = None, 
    convergence_line: bool = True, 
    fig_size: Tuple[int, int] = (20, 6),
    sns_style: Optional[str] = None, 
    savefig: Optional[str] = None, 
    set_legend: bool = True, 
    subplot_kws: Optional[Dict[str, Any]] = None,
    **kws: Dict[str, Any]
) -> List[plt.Axes]:
    """
    Horizontally visualizes multiple models' learning curves. 

    Determines cross-validated training and test scores for different 
    training set sizes.

    Parameters 
    ----------
    models : `Union[BaseEstimator, List[BaseEstimator]]`
        An estimator instance or a list of estimator instances that implement 
        `fit` and `predict` methods which will be cloned for each validation. 
        
    X : `np.ndarray` of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : `np.ndarray` of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to `X` for classification or regression; `None` for 
        unsupervised learning.
   
    cv : `int`, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for `cv` are:
        - `None`, to use the default 5-fold cross-validation,
        - `int`, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        If `None`, the default 4-fold cross-validation is used.

    train_sizes : `np.ndarray` of shape (n_ticks,), optional
        Relative or absolute numbers of training examples that will be used 
        to generate the learning curve. If the dtype is float, it is regarded 
        as a fraction of the maximum size of the training set, i.e., it has 
        to be within (0, 1]. Otherwise, it is interpreted as absolute sizes 
        of the training sets.

    baseline_score : `float`, default=0.4 
        Base score to start counting in the score y-axis.

    scoring : `str` or callable, optional
        A `str` (see model evaluation documentation) or a scorer callable 
        object/function with signature `scorer(estimator, X, y)`.

    convergence_line : `bool`, default=True 
        Display the convergence line that indicates the level of bias 
        between the training and validation curve. 
        
    fig_size : `tuple` of (`int`, `int`), default=(20, 6)
        The matplotlib figure size given as a tuple of width and height.
        
    sns_style : `str`, optional
        The seaborn style.

    savefig : `str`, optional
        Path to save the figure to. If `None`, the figure is shown.
        
    set_legend : `bool`, default=True 
        Display legend in each figure. Note the default location of the 
        legend is 'best' from :func:`~matplotlib.Axes.legend`.
        
    subplot_kws : `dict`, optional
        Subplot keyword arguments passed to :func:`matplotlib.subplots_adjust`.
        Default is `dict(left=0.0625, right=0.95, wspace=0.1)`.
        
    kws : `dict`
        Additional keyword arguments passed to 
        :func:`sklearn.model_selection.learning_curve`.

    Returns
    -------
    `List` of `matplotlib.axes.Axes`
        List of `Axes` objects with the plots.

    Examples 
    --------
    (1) -> Plot via a meta-estimator already cross-validated. 
    
    >>> import watex  # must install watex to get the pretrained model (pip install watex)
    >>> from watex.models.premodels import p 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.mlviz import plot_learning_curves
    >>> X, y = fetch_data('bagoue prepared')  # yields a sparse matrix 
    >>> # collect 4 estimators already cross-validated from SVMs
    >>> models = [p.SVM.linear, p.SVM.rbf, p.SVM.sigmoid, p.SVM.poly]
    >>> plot_learning_curves(models, X, y, cv=4, sns_style='darkgrid')
    
    (2) -> Plot with multiple models not cross-validated yet.
    
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.svm import SVC 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> models = [LogisticRegression(), RandomForestClassifier(), SVC(), KNeighborsClassifier()]
    >>> plot_learning_curves(models, X, y, cv=4, sns_style='darkgrid')

    Notes
    -----
    This function assumes that the provided estimators are compatible with 
    the scikit-learn estimator interface and have been fitted prior to 
    calling this function. If `baseline_score` is not in the range (0, 1), a 
    `ValueError` is raised.
    
    The learning curves are generated by plotting the training and validation 
    scores against the number of training samples. The convergence line is 
    added to indicate the level of bias between the training and validation 
    curves.

    The training and validation scores are calculated as:

    .. math::
        \text{score} = \frac{1}{n} \sum_{i=1}^{n} \text{accuracy}(y_i, \hat{y}_i)

    See Also
    --------
    - [1] R. O. Duda, P. E. Hart, and D. G. Stork, "Pattern Classification," 
      2nd edition, Wiley, 2000.

    References
    ----------
    .. [1] R. O. Duda, P. E. Hart, and D. G. Stork, "Pattern Classification," 
       2nd edition, Wiley, 2000.
    """
    models = is_iterable(models, exclude_string=True, transform =True )
    
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
    effort : `List[float]`
        The effort values to be plotted on the x-axis. This typically 
        represents the proportion of total resources or input applied, 
        ranging from 0 to 1.
    yield_ : `List[float]`
        The yield values to be plotted on the y-axis. This typically 
        represents the proportion of total output or benefit gained, 
        also ranging from 0 to 1.
    title : `str`, optional
        The title of the plot. Default is "ABC plot".
    xlabel : `str`, optional
        The label for the x-axis. Default is "Effort".
    ylabel : `str`, optional
        The label for the y-axis. Default is "Yield".
    figsize : `Tuple[int, int]`, optional
        The size of the figure in inches. Default is (8, 6).
    abc_line_color : `str`, optional
        The color of the ABC line which represents the actual effort vs 
        yield relationship. Default is "blue".
    identity_line_color : `str`, optional
        The color of the identity line which represents a perfect balance 
        between effort and yield. Default is "magenta".
    uniform_line_color : `str`, optional
        The color of the uniform distribution line which represents 
        a consistent yield regardless of effort. Default is "green".
    abc_linestyle : `str`, optional
        The linestyle for the ABC line. Default is "-".
    identity_linestyle : `str`, optional
        The linestyle for the identity line. Default is "--".
    uniform_linestyle : `str`, optional
        The linestyle for the uniform line. Default is ":".
    linewidth : `int`, optional
        The width of the lines in the plot. Default is 2.
    legend : `bool`, optional
        Whether to display the legend on the plot. Default is True.
    set_annotations : `bool`, optional
        Whether to annotate the plot with information about sets A, B, and C. 
        Default is True.
    set_a : `Tuple[int, int]`, optional
        The annotation for set A in the format `(index, value)`, where `index` 
        is the position in the list and `value` is the corresponding yield. 
        Default is (0, 2).
    set_b : `Tuple[int, int]`, optional
        The annotation for set B in the format `(index, value)`, where `index` 
        is the position in the list and `value` is the corresponding yield. 
        Default is (0, 0).
    set_c : `Tuple[int, str]`, optional
        The annotation for set C in the format `(index, description)`, where 
        `index` is the position in the list and `description` is a label for 
        the yield. Default is (5, '+51 dropped').
    savefig : `Optional[str]`, optional
        The file path to save the figure. If `None`, the figure is not saved. 
        Default is None.
    
    Returns
    -------
    `matplotlib.axes.Axes`
        The matplotlib `Axes` object for the plot.
 
    See Also
    --------
    gofast.tools.mathex.compute_effort_yield: 
        Compute effort and yield values from importance data. 
        
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_abc_curve
    >>> effort = np.linspace(0, 1, 100)
    >>> yield_ = np.sqrt(effort)  # Non-linear yield
    >>> plot_abc_curve(effort, yield_, title='Effort-Yield Analysis')

    Notes
    -----
    The ABC curve is useful for evaluating the balance between effort
    and yield. The identity line shows perfect balance, while the 
    uniform line shows consistent yield regardless of effort.

    The ABC curve typically consists of three key reference lines:
    - The ABC line: Shows the actual relationship between effort and yield.
    - The identity line: Represents perfect balance, where every unit of 
      effort yields a proportional unit of yield.
    - The uniform line: Represents a scenario where yield increases 
      uniformly with effort.

    The ABC curve can be mathematically formulated as follows:
    
    .. math::
        y = f(x)
    
    where `y` is the yield and `x` is the effort. The identity line is 
    represented by:
    
    .. math::
        y = x
    
    The uniform line is represented by:
    
    .. math::
        y = \text{constant}

    References
    ----------
    .. [1] J. K. Author, "Analysis of ABC Curves," Journal of Data Analysis, 
       vol. 10, no. 2, pp. 123-130, 2020.
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
    
    A Confidence Interval (CI) is an estimate derived from observed data 
    statistics, indicating a range where a population parameter is likely 
    to be found at a specified confidence level. Introduced by Jerzy Neyman 
    in 1937, CI is a crucial concept in statistical inference. Common types 
    include CI for mean, median, the difference between means, a proportion, 
    and the difference in proportions.

    Parameters 
    ----------
    y : `Optional[Union[str, ArrayLike]]`, optional
        Dependent variable values. If a string, `y` should be a column name 
        in `data`. `data` cannot be `None` in this case.
    x : `Optional[Union[str, ArrayLike]]`, optional
        Independent variable values. If a string, `x` should be a column name 
        in `data`. `data` cannot be `None` in this case.
    data : `pd.DataFrame`, optional
        Input data structure. Can be a long-form collection of vectors that 
        can be assigned to named variables or a wide-form dataset that will 
        be reshaped.
    ci : `float`, default=0.95
        The confidence level for the interval. It must be between 0 and 1.
    kind : `str`, default='line'
        The type of plot to create. Options include 'line', 'reg', or 
        'bootstrap'.
    b_samples : `int`, default=1000
        The number of bootstrap samples to use for the 'bootstrap' method.
    sns_kws : `Dict`
        Additional keyword arguments passed to the seaborn plot function.

    Returns 
    -------
    `matplotlib.axes.Axes`
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
    
    Notes
    -----
    Confidence intervals provide a range of values which are used to estimate 
    the true value of a population parameter. They are calculated using 
    observed data and the specified confidence level :math:`(1 - \alpha)`.
    The width of the confidence interval gives us an idea about how uncertain 
    we are about the unknown parameter. A wider interval indicates more 
    uncertainty, while a narrower interval suggests more precision.

    For the bootstrap method, the confidence interval is estimated by 
    resampling the observed data with replacement and computing the statistic 
    of interest (e.g., the mean) for each resample. The distribution of these 
    resampled statistics is then used to estimate the confidence interval.

    The confidence interval for the mean :math:`\mu` can be expressed as:

    .. math::
        \bar{x} \pm Z \frac{\sigma}{\sqrt{n}}

    where :math:`\bar{x}` is the sample mean, :math:`Z` is the Z-value from 
    the standard normal distribution corresponding to the desired confidence 
    level, :math:`\sigma` is the population standard deviation, and :math:`n` 
    is the sample size.

    References
    ----------
    .. [1] Neyman, J. (1937). "Outline of a Theory of Statistical Estimation 
       Based on the Classical Theory of Probability". Philosophical 
       Transactions of the Royal Society of London. Series A, Mathematical 
       and Physical Sciences. 236 (767): 333–380.
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
    data: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (6, 6),
    scatter_s: int = 0.5,
    line_colors: Tuple[str, str, str] = ('firebrick', 'fuchsia', 'blue'),
    line_styles: Tuple[str, str, str] = ('-', '--', ':'),
    title: str = 'Different Standard Deviations',
    show_legend: bool = True
) -> plt.Axes:
    """
    Plots the confidence ellipse of a two-dimensional dataset.

    This function visualizes the confidence ellipse representing the 
    covariance of the provided `x` and `y` variables. The ellipses plotted 
    represent 1, 2, and 3 standard deviations from the mean.

    Parameters
    ----------
    x : `Union[str, ArrayLike]`
        The x-coordinates of the data points or column name in DataFrame.
        If `data` is provided, `x` should be a column name in `data`.
    y : `Union[str, ArrayLike]`
        The y-coordinates of the data points or column name in DataFrame.
        If `data` is provided, `y` should be a column name in `data`.
    data : `pd.DataFrame`, optional
        DataFrame containing `x` and `y` data. Required if `x` and `y` are 
        column names.
    figsize : `Tuple[int, int]`, optional
        Size of the figure (width, height). Default is (6, 6).
    scatter_s : `int`, optional
        The size of the scatter plot markers. Default is 0.5.
    line_colors : `Tuple[str, str, str]`, optional
        The colors of the lines for the 1, 2, and 3 standard deviation ellipses.
        Default is ('firebrick', 'fuchsia', 'blue').
    line_styles : `Tuple[str, str, str]`, optional
        The line styles for the 1, 2, and 3 standard deviation ellipses.
        Default is ('-', '--', ':').
    title : `str`, optional
        The title of the plot. Default is 'Different Standard Deviations'.
    show_legend : `bool`, optional
        If `True`, shows the legend. Default is `True`.

    Returns
    -------
    `plt.Axes`
        The matplotlib `Axes` containing the plot.

    Notes
    -----
    The confidence ellipse represents the region where the true values of 
    the variables lie with a specified confidence level, assuming a 
    bivariate normal distribution. The ellipses correspond to 1, 2, and 3 
    standard deviations from the mean, covering approximately 68%, 95%, and 
    99.7% of the data, respectively.

    The method to calculate the ellipse parameters is based on the 
    eigenvalue decomposition of the covariance matrix of the data, which 
    provides the lengths of the semi-major and semi-minor axes and the 
    orientation of the ellipse.

    The equation of the ellipse in the principal axis frame is given by:

    .. math::
        \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1

    where `a` and `b` are the lengths of the semi-major and semi-minor axes, 
    respectively.

    See Also
    --------
    gofast.tools.mathex.compute_confidence_ellipse : Function to compute 
        confidence ellipses.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import plot_confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> ax = plot_confidence_ellipse(x, y)
    >>> plt.show()

    References
    ----------
    .. [1] Carsten Schelp, "Plotting Confidence Ellipse," 2018. 
       Available: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
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
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> Ellipse:
    """
    Creates a covariance confidence ellipse of `x` and `y`.

    This function adds an `Ellipse` patch to the given `ax` which represents 
    the confidence interval of the covariance of the input data `x` and `y`. 
    The ellipse is defined by the specified number of standard deviations (`n_std`).

    Parameters
    ----------
    x : `Union[str, ArrayLike]`
        The x-coordinates of the data points or column name in DataFrame.
        If `data` is provided, `x` should be a column name in `data`.
    y : `Union[str, ArrayLike]`
        The y-coordinates of the data points or column name in DataFrame.
        If `data` is provided, `y` should be a column name in `data`.
    ax : `plt.Axes`
        The axes object where the ellipse will be plotted.
    n_std : `float`, optional
        The number of standard deviations to determine the ellipse's radius. 
        Default is 3.
    facecolor : `str`, optional
        The color of the ellipse's face. Default is 'none' (no fill).
    data : `pd.DataFrame`, optional
        DataFrame containing `x` and `y` data. Required if `x` and `y` are 
        column names.
    **kwargs
        Additional arguments passed to the `Ellipse` patch.

    Returns
    -------
    `Ellipse`
        The `Ellipse` object added to the axes.

    Raises
    ------
    `ValueError`
        If `x` and `y` are not of the same size.

    Notes
    -----
    The confidence ellipse represents the region where the true values of 
    the variables lie with a specified confidence level, assuming a 
    bivariate normal distribution. The ellipses correspond to 1, 2, and 3 
    standard deviations from the mean, covering approximately 68%, 95%, and 
    99.7% of the data, respectively.

    The ellipse parameters are calculated based on the eigenvalues and 
    eigenvectors of the covariance matrix of `x` and `y`.

    The equation of the ellipse in the principal axis frame is given by:

    .. math::
        \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1

    where `a` and `b` are the lengths of the semi-major and semi-minor axes, 
    respectively.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.mlviz import confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> fig, ax = plt.subplots()
    >>> confidence_ellipse(x, y, ax, n_std=2, edgecolor='red')
    >>> ax.scatter(x, y, s=3)
    >>> plt.show()

    References
    ----------
    .. [1] Carsten Schelp, "Plotting Confidence Ellipse," 2018. 
       Available: https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
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

def plot_roc_curves(
    clfs: List[Union[BaseEstimator, Any]], 
    X: Union[ArrayLike, SparseMatrix], 
    y: Union[ArrayLike, Series], 
    names: Optional[List[str]] = None, 
    colors: Optional[List[str]] = None, 
    ncols: int = 3, 
    score: bool = False, 
    kind: str = "inone",
    ax: Optional[plt.Axes] = None,  
    fig_size: Tuple[int, int] = (7, 7), 
    **roc_kws: Dict
) -> plt.Axes:
    """
    Quick plot of Receiving Operating Characteristic (ROC) of fitted models.

    Parameters
    ----------
    clfs : `List[Union[BaseEstimator, Any]]`
        List of models for ROC evaluation. Models should be scikit-learn or 
        XGBoost or GOFast estimators.
    X : `{array-like, sparse matrix} of shape (n_samples, n_features)`
        Training instances to cluster. It must be noted that the data will be 
        converted to C ordering, which will cause a memory copy if the given 
        data is not C-contiguous. If a sparse matrix is passed, a copy will 
        be made if it's not in CSR format.
    y : `Union[np.ndarray, pd.Series]` of length (n_samples,)
        An array or series of target or class values. Preferably, the array 
        represents the test class labels data for error evaluation.
    names : `List[str]`, optional
        List of model names. If not given, a raw name of the model is passed 
        instead.
    colors : `Union[str, List[str]]`, optional
        Colors to specify each model plot.
    ncols : `int`, default=3
        Number of plots to be placed inline before skipping to the next column. 
        This is feasible if `kind` is set to 'individual'.
    score : `bool`, default=False
        Append the Area Under the Curve (AUC) score to the legend.
    kind : `str`, default='inone'
        If 'individual', '2', or 'single', plot each ROC model separately. 
        Any other value groups ROC curves into a single plot.
    ax : `Optional[plt.Axes]`, optional
        Matplotlib axes to plot on.
    fig_size : `Tuple[int, int]`, default=(7, 7)
        Size of the figure.
    roc_kws : `Dict`
        Additional keyword arguments passed to the `sklearn.metrics.roc_curve` function.

    Returns
    -------
    `plt.Axes`
        The matplotlib axes containing the ROC plot(s).

    Examples
    --------
    >>> from gofast.plot.mlviz import plot_roc_curves 
    >>> from sklearn.datasets import make_moons 
    >>> from gofast.exlib import train_test_split, KNeighborsClassifier, SVC, XGBClassifier, LogisticRegression 
    >>> X, y = make_moons(n_samples=2000, noise=0.2)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
    >>> clfs = [m().fit(X_train, y_train) for m in (KNeighborsClassifier, SVC, XGBClassifier, LogisticRegression)]
    >>> plot_roc_curves(clfs, X_test, y_test)
    >>> plot_roc_curves(clfs, X_test, y_test, kind='2', ncols=4, fig_size=(10, 4))

    Notes
    -----
    The ROC curve is created by plotting the true positive rate (TPR) against 
    the false positive rate (FPR) at various threshold settings. It is often 
    used to evaluate the performance of a binary classifier.

    The area under the ROC curve (AUC) provides an aggregate measure of 
    performance across all possible classification thresholds. An AUC of 1 
    represents a perfect model, while an AUC of 0.5 represents a worthless model.

    The ROC curve can be mathematically represented as:

    .. math::
        TPR = \frac{TP}{TP + FN}
    
    .. math::
        FPR = \frac{FP}{FP + TN}

    See Also
    --------
    sklearn.metrics.roc_curve : Compute Receiver operating characteristic (ROC).

    References
    ----------
    .. [1] Fawcett, T. (2006). "An introduction to ROC analysis". Pattern Recognition Letters. 27 (8): 861–874.
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
    X: Union[ArrayLike, pd.DataFrame], 
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
    model : `Any`
        A trained model object that is compatible with SHAP explainer.

    X : `Union[ArrayLike, pd.DataFrame]`
        Input data for which the SHAP values are to be computed. If a 
        DataFrame is provided, the feature names are taken from the DataFrame 
        columns.

    feature_names : `Optional[List[str]]`, optional
        List of feature names if `X` is an array-like object without feature 
        names.

    plot_type : `str`, optional
        Type of the plot. Either 'dot' or 'bar'. The default is 'dot'.

    color_bar_label : `str`, optional
        Label for the color bar. The default is 'Feature value'.

    max_display : `int`, optional
        Maximum number of features to display on the summary plot. 
        The default is 10.

    show : `bool`, optional
        Whether to show the plot. The default is `True`. If `False`, 
        the function returns the figure object.

    plot_size : `Tuple[int, int]`, optional
        Size of the plot specified as (width, height). The default is (15, 10).

    cmap : `str`, optional
        Colormap to use for plotting. The default is 'coolwarm'.

    Returns
    -------
    `Optional[Figure]`
        The figure object if `show` is `False`, otherwise `None`.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.datasets import make_classification
    >>> from gofast.plot.mlviz import plot_shap_summary
    >>> X, y = make_classification(n_features=5, random_state=42, return_X_y=True)
    >>> model = RandomForestClassifier().fit(X, y)
    >>> plot_shap_summary(model, X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

    Notes
    -----
    SHAP (SHapley Additive exPlanations) values are a method to explain 
    individual predictions by computing the contribution of each feature 
    to the prediction. SHAP values are based on cooperative game theory 
    and provide a unified measure of feature importance.

    The SHAP summary plot provides a global view of the feature importance 
    and the distribution of the impacts of the features on the model output. 
    It combines feature importance with feature effects. Each point on the 
    summary plot is a Shapley value for a feature and an instance. The color 
    represents the value of the feature from low to high.

    The SHAP values for a feature :math:`i` are computed as:

    .. math::
        \phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} 
        [f(S \cup \{i\}) - f(S)]

    where :math:`N` is the set of all features, :math:`S` is a subset of 
    features, and :math:`f(S)` is the model prediction for features in :math:`S`.

    See Also
    --------
    shap.Explainer : SHAP explainer for different model types.

    References
    ----------
    .. [1] Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to  
           Interpreting Model Predictions". Advances in Neural Information 
           Processing Systems 30 (NIPS 2017).
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

def plot_confusion_matrix2(
    clf: Any, 
    Xt: Union[np.ndarray, DataFrame], 
    yt: Union[np.ndarray, pd.Series], 
    labels: Optional[List[str]] = None, 
    encoder: Optional[Callable[[], ...]] = None, 
    savefig: Optional[str] = None, 
    fig_size: Tuple[int, int] = (6, 6), 
    **kws: Any
) -> Any:
    """
    Confusion matrix plot using the 'yellowbrick' package.  
    
    Creates a heatmap visualization of the `sklearn.metrics.confusion_matrix()`.
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
    ----------
    clf : `Any`
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by `is_fitted`.
        
    Xt : `Union[np.ndarray, pd.DataFrame]`
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : `Union[np.ndarray, pd.Series]`
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  

    labels : `Optional[List[str]]`, optional
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the `fit()` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.
        
    encoder : `Optional[Union[dict, LabelEncoder]]`, optional
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to `fit()` or `score()`. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
    savefig : `Optional[str]`, optional
        The path to save the figures. Argument is passed to `matplotlib.Figure`
        class.
        
    fig_size : `Tuple[int, int]`, default=(6, 6)
        The matplotlib figure size given as a tuple of width and height.
        
    **kws : `Any`
        Additional keyword arguments passed to `yellowbrick.classifier.ConfusionMatrix`.
          
    Returns 
    -------
    cmo : `yellowbrick.classifier.confusion_matrix.ConfusionMatrix`
        Returns a yellowbrick confusion matrix object instance. 
    
    Examples 
    --------
    >>> from gofast.datasets import fetch_data
    >>> from sklearn.model_selection import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.plot.mlviz import plot_confusion_matrix2
    >>> X, Xt, y, yt  = train_test_split(*fetch_data('bagoue analysed'), test_size=0.25)  
    >>> pmo = pModels(model='xgboost') 
    >>> pmo.fit(X, y)
    >>> print(pmo.estimator_) 
    >>> ypred = pmo.predict(Xt) 
    >>> plot_confusion_matrix2(pmo.XGB.best_estimator_, Xt, yt)

    Notes
    -----
    A confusion matrix is a table that is often used to describe the performance
    of a classification model (or "classifier") on a set of test data for which 
    the true values are known. It allows the visualization of the performance of 
    an algorithm. The matrix compares the actual target values with those 
    predicted by the machine learning model.

    The confusion matrix itself is relatively simple to understand, but the 
    related terminology can be confusing. The main diagonal (from top left to 
    bottom right) represents the instances that are correctly classified. The 
    off-diagonal elements are those that are misclassified.

    The confusion matrix can be mathematically represented as:

    .. math::
        \begin{array}{cc}
        TN & FP \\
        FN & TP
        \end{array}

    where:
        - TP: True Positive
        - TN: True Negative
        - FP: False Positive
        - FN: False Negative

    See Also
    --------
    yellowbrick.classifier.confusion_matrix.ConfusionMatrix : 
        Yellowbrick confusion matrix visualizer.

    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", 
       JMLR 12, pp. 2825-2830, 2011.
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
    title: Optional[str] = None,  
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None,  
    fig_size: Tuple[int, int] = (8, 8),
    scatter_color: str = 'blue', 
    line_color: str = 'red', 
    line_style: str = '--', 
    annotate: bool = True, 
    ax: Optional[plt.Axes] = None, 
    **r2_score_kws: Any
) -> plt.Axes:
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
    y_true : `ArrayLike`
        The true target values. This can be an array-like structure such as 
        a list, numpy array, or pandas Series.

    y_pred : `ArrayLike`
        The predicted target values. This can be an array-like structure such 
        as a list, numpy array, or pandas Series.

    title : `Optional[str]`, optional
        The title of the plot. If `None`, defaults to 'Model Performance: 
        Actual vs Predicted'.

    xlabel : `Optional[str]`, optional
        The label for the x-axis. If `None`, defaults to 'Actual Values'.

    ylabel : `Optional[str]`, optional
        The label for the y-axis. If `None`, defaults to 'Predicted Values'.

    fig_size : `Tuple[int, int]`, optional
        The size of the figure in inches. Defaults to (8, 8).

    scatter_color : `str`, optional
        The color of the scatter plot points. Defaults to 'blue'.

    line_color : `str`, optional
        The color of the line representing perfect predictions. Defaults to 'red'.

    line_style : `str`, optional
        The style of the line representing perfect predictions. Defaults to '--'.

    annotate : `bool`, optional
        If `True`, annotates the plot with the R-squared value. Defaults to True.

    ax : `Optional[plt.Axes]`, optional
        The axes upon which to draw the plot. If `None`, a new figure and axes 
        are created.

    **r2_score_kws : `Any`, optional
        Additional keyword arguments to be passed to `sklearn.metrics.r2_score`.

    Returns
    -------
    `plt.Axes`
        The matplotlib axes containing the plot.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> import matplotlib.pyplot as plt

    # Generating synthetic data
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10, 
                               random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)

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

    Notes
    -----
    The R-squared (coefficient of determination) is a statistical measure 
    that represents the proportion of the variance for a dependent variable 
    that's explained by an independent variable or variables in a regression 
    model. It provides an indication of the goodness of fit and therefore a 
    measure of how well unseen samples are likely to be predicted by the model.

    The R-squared value is calculated as:

    .. math::
        R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}

    where :math:`SS_{res}` is the sum of squares of residuals and :math:`SS_{tot}` 
    is the total sum of squares.

    See Also
    --------
    sklearn.metrics.r2_score : Function to compute the R-squared, or coefficient 
                               of determination.

    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 
           pp. 2825-2830, 2011.
    """
    y_true, y_pred= validate_yy(y_true, y_pred, "continuous")
    if ax is None: 
        fig, ax = plt.subplots(figsize=fig_size)
    
    # Calculate R-squared value
    r_squared = r2_score(y_true, y_pred, **r2_score_kws)
    
    # Plot actual vs predicted values
    ax.scatter(y_true, y_pred, color=scatter_color, 
               label='Predictions vs Actual data')
    
    # Plot a line representing perfect predictions
    perfect_preds = [min(y_true.min(), y_pred.min()),
                     max(y_true.max(), y_pred.max())]
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
    clfs: List [BaseEstimator], 
    X: NDArray, 
    y: ArrayLike, *,  
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
            plot_confusion_matrix2(
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
) -> np.ndarray:
    """
    Plot a confusion matrix for a single classifier model to evaluate 
    the accuracy of a classification.

    Parameters
    ----------
    y_true : `Union[ArrayLike, pd.Series]`
        An array or series of true target or class labels, representing 
        the actual classification outcomes.
    y_pred : `Union[ArrayLike, pd.Series]`
        An array or series of predicted target or class labels, 
        as determined by the classifier.
    view : `bool`, optional
        If `True`, display the confusion matrix using matplotlib's imshow;
        otherwise, do not display. Default is `True`.
    ax : `Optional[plt.Axes]`, optional
        Pre-existing axes for the plot. If `None`, a new figure and axes 
        object is created. Default is `None`.
    annot : `bool`, optional
        If `True`, annotate the heat map with the number of samples in 
        each category. Default is `True`.
    kws : `Dict[str, Any]`
        Additional keyword arguments to pass to 
        `sklearn.metrics.confusion_matrix`.

    Returns
    -------
    `np.ndarray`
        The confusion matrix array.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from gofast.plot.mlviz import plot_confusion_matrix
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> model = AdaBoostClassifier()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> plot_confusion_matrix(y_test, y_pred)

    >>> #Import the required models and fetch an AdaBoost model 
    >>> # for instance then plot the confusion matrix 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.tools.utils import plot_confusion_matrix
    >>> # split the data. Note that fetch_data output X and y 
    >>> X, Xt, y, yt = train_test_split(*fetch_data('bagoue analysed'), test_size=0.25)  
    >>> # train the model with the best estimator 
    >>> pmo = pModels(model='ada') 
    >>> pmo.fit(X, y)
    >>> print(pmo.estimator_)
    >>> # Predict the score using the best estimator 
    >>> ypred = pmo.predict(Xt) 
    >>> # now plot the score 
    >>> plot_confusion_matrix(yt, ypred)

    Notes
    -----
    Confusion matrices are a crucial part of understanding the performance 
    of classifiers, not just in terms of overall accuracy but in recognizing 
    where misclassifications are occurring. This function leverages seaborn's 
    heatmap function to visualize the matrix effectively.

    The confusion matrix `C` for a binary classification task can be defined as:

    .. math::
        C = \begin{bmatrix}
        TP & FP \\
        FN & TN
        \end{bmatrix}

    where:
        - `TP` is the number of true positive predictions.
        - `TN` is the number of true negative predictions.
        - `FP` is the number of false positive predictions.
        - `FN` is the number of false negative predictions.

    The overall accuracy of the classifier can be calculated as:

    .. math::
        \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

    See Also
    --------
    sklearn.metrics.confusion_matrix : Compute confusion matrix to evaluate 
                                       the accuracy of a classification.

    References
    ----------
    .. [1] Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 
           pp. 2825-2830, 2011.
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
    model_fn: Callable[[], 'Model'],
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
    model_fn : `Callable[[], keras.Model]`
        A function that returns a compiled neural network model. The
        function should not take any arguments and must return a compiled
        Keras model.

    X : `np.ndarray`
        Training features, typically an array of shape (n_samples, n_features).

    y : `np.ndarray`
        Training labels or target values, typically an array of shape
        (n_samples,) or (n_samples, n_outputs).

    n_splits : `int`, optional
        Number of splits for the K-Fold cross-validation. Default is 5.

    epochs : `int`, optional
        Number of epochs for training each model during the cross-validation.
        Default is 10.

    metric : `str`, optional
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
    This function utilizes `KFold` from `sklearn.model_selection` to create training and 
    validation splits. It is essential that the model function provided 
    compiles the model with the necessary metrics as they are used to 
    monitor training performance.

    The cross-validation process involves splitting the data into `n_splits`
    folds, training the model on `n_splits - 1` folds, and validating it on
    the remaining fold. This process is repeated for each fold, and the
    performance metric is recorded for each epoch.

    The performance metric for each fold is then plotted to visualize the
    consistency and stability of the model across different subsets of the
    data.

    See Also
    --------
    `sklearn.model_selection.KFold` : Provides cross-validation iterator.
    
    References
    ----------
    .. [1] Chollet, F. (2015). Keras. https://github.com/fchollet/keras
    .. [2] Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 
           pp. 2825-2830, 2011.
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

    Optionally calculate and display selected metrics such as 
    MSE, RMSE, MAE, and R2.

    Parameters
    ----------
    y_true : `ArrayLike`
        The true values for comparison. Must be a one-dimensional array-like 
        object of numerical data.

    y_pred : `ArrayLike`
        The predicted values to be plotted against the true values. Must be 
        a one-dimensional array-like object of numerical data.

    metrics : `Optional[List[str]]`, optional
        A list of strings indicating which metrics to calculate and display on 
        the plot. Possible values are 'mse', 'rmse', 'mae', and 'r2'. If None,
        no metrics are displayed. If ``'*'``, displays all metric values.

    point_color : `str`, optional
        The color for the scatter plot points. Default is 'blue'.

    line_color : `str`, optional
        The color for the ideal line. Default is 'red'.

    point_label : `str`, optional
        The label for the scatter plot points in the legend. Default is 'Prediction'.

    line_label : `str`, optional
        The label for the ideal line in the legend. Default is 'Ideal'.

    xlabel : `str`, optional
        The label for the X-axis. Default is 'Actual (true) value'.

    ylabel : `str`, optional
        The label for the Y-axis. Default is 'Predicted values'.

    title : `str`, optional
        The title of the plot. Default is 'Actual (true) value vs predicted values'.

    show_grid : `bool`, optional
        Whether to show grid lines on the plot. Default is True.

    grid_style : `str`, optional
        The style of the grid lines. Default is '-' (solid line).

    point_size : `float`, optional
        The size of the scatter plot points. Default is 50.

    line_style : `str`, optional
        The style of the ideal line. Default is '-' (solid line).

    ax : `Optional[plt.Axes]`, optional
        The matplotlib Axes object to draw the plot on. If None, a new figure 
        and axes are created. Default is None.

    fig_size : `Optional[Tuple[int, int]]`, optional
        The size of the figure in inches. Default is (10, 8).

    **metrics_kws
        Additional keyword arguments to pass to the metric functions.

    Returns
    -------
    ax : `plt.Axes`
        The matplotlib Axes object with the plot.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.mlviz import plot_actual_vs_predicted
    >>> y_true = np.array([1., 2., 3., 4., 5.])
    >>> y_pred = np.array([1.1, 1.9, 3.1, 3.9, 4.9])
    >>> plot_actual_vs_predicted(y_true, y_pred, metrics=['mse', 'rmse'], 
    ...                          point_color='green', line_color='r') 

    Notes
    -----
    This function creates a scatter plot of the true values (`y_true`) against 
    the predicted values (`y_pred`) with an ideal line representing perfect 
    predictions where the true value equals the predicted value. The metrics 
    are calculated as follows:

    - Mean Squared Error (MSE): 
      .. math:: \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2

    - Root Mean Squared Error (RMSE): 
      .. math:: \text{RMSE} = \sqrt{\text{MSE}}

    - Mean Absolute Error (MAE): 
      .. math:: \text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|

    - R-squared (R2): 
      .. math:: R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}

    The calculated metrics are displayed on the plot if specified.

    See Also
    --------
    `matplotlib.pyplot.scatter` : To create scatter plots.
    `matplotlib.pyplot.plot` : To draw the ideal line.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
           Computing in Science & Engineering, 9(3), 90-95.
    .. [2] Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 
           pp. 2825-2830, 2011.
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
    titles: Optional[List[str]] = None,
    xlabel: str = 'X',
    ylabel: str = 'Y',
    figsize: Tuple[int, int] = (15, 5),
    ci: Optional[int] = 95, 
    **reg_kws: Any
) -> plt.Figure:
    """
    Creates a series of plots to diagnose linear regression fits.

    Parameters
    ----------
    x : `ArrayLike`
        The independent variable data.
    
    ys : `List[ArrayLike]`
        A list of dependent variable datasets to be plotted against `x`.
        
    titles : `Optional[List[str]]`, optional 
        Titles for each subplot. If not provided, subplots will not have titles.
        
    xlabel : `str`, default='X'
        Label for the x-axis.
        
    ylabel : `str`, default='Y'
        Label for the y-axis.
        
    figsize : `Tuple[int, int]`, default=(15, 5)
        Size of the entire figure.
        
    ci : `Optional[int]`, default=95
        Size of the confidence interval for the regression estimate.
        
    reg_kws : `Any`
        Additional parameters passed to `seaborn.regplot`.

    Returns
    -------
    `plt.Figure`
        The matplotlib Figure object containing the plots.

    Examples
    --------
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
    ...           'Heteroscedastic noise']
    >>> fig = plot_regression_diagnostics(x, y1, y2, y3, titles)
    >>> plt.show()

    Notes
    -----
    This function is useful for visually diagnosing the assumptions of linear regression 
    models, such as linearity, homoscedasticity, and the presence of outliers.

    Linear regression assumes that the relationship between the dependent variable `y`
    and the independent variable `x` can be modeled as a straight line. The confidence 
    interval (CI) provides a range of values that is likely to contain the true value of 
    the parameter being estimated.

    The mathematical representation of the linear regression model is:

    .. math::
        y = \beta_0 + \beta_1 x + \epsilon

    where:
        - :math:`y` is the dependent variable.
        - :math:`x` is the independent variable.
        - :math:`\beta_0` is the intercept.
        - :math:`\beta_1` is the slope.
        - :math:`\epsilon` is the error term.

    The confidence interval (CI) can be calculated as:

    .. math::
        \text{CI} = \hat{y} \pm t_{\alpha/2, n-2} \cdot SE

    where:
        - :math:`\hat{y}` is the predicted value.
        - :math:`t_{\alpha/2, n-2}` is the t-score for a given confidence level and degrees of freedom.
        - :math:`SE` is the standard error of the prediction.

    See Also
    --------
    seaborn.regplot : Plot data and a linear regression model fit.
    
    References
    ----------
    .. [1] Seaborn documentation: https://seaborn.pydata.org/generated/seaborn.regplot.html
    .. [2] Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis. Wiley.
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
    residuals : `ArrayLike`
        Standardized residuals from the regression model.
        
    leverage : `ArrayLike`
        Leverage values calculated from the model.
        
    cook_d : `Optional[ArrayLike]`, optional
        Cook's distance for each observation in the model. Default is `None`.
        
    figsize : `Tuple[int, int]`, default=(8, 6)
        The figure size for the plot.
        
    cook_d_threshold : `float`, default=0.5
        The threshold for Cook's distance to draw a contour and potentially 
        annotate points.
        
    annotate : `bool`, default=True
        If `True`, annotate points that exceed the Cook's distance threshold.
        
    scatter_kwargs : `Optional[dict]`, optional
        Additional keyword arguments for the scatter plot.
        
    cook_d_kwargs : `Optional[dict]`, optional
        Additional keyword arguments for the Cook's distance ellipses.
        
    line_kwargs : `Optional[dict]`, optional
        Additional keyword arguments for the horizontal line at 0.
        
    annotation_kwargs : `Optional[dict]`, optional
        Additional keyword arguments for the annotations.

    Returns
    -------
    `plt.Axes`
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

    Notes
    -----
    The plot of residuals vs leverage is a diagnostic tool for evaluating 
    the fit of a regression model. 

    Leverage is a measure of how far an observation deviates from the mean 
    of the independent variables. Standardized residuals are the residuals 
    divided by an estimate of their standard deviation.

    Cook's distance is a measure used in regression analysis to identify 
    influential data points. It combines the information of both the leverage 
    and the residuals to assess the influence of each observation:

    .. math::
        D_i = \frac{e_i^2}{p \cdot MSE} \left( \frac{h_i}{(1 - h_i)^2} \right)

    where:
        - :math:`D_i` is Cook's distance for the i-th observation.
        - :math:`e_i` is the residual for the i-th observation.
        - :math:`p` is the number of parameters in the model.
        - :math:`MSE` is the mean squared error of the model.
        - :math:`h_i` is the leverage of the i-th observation.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Linear regression.
    
    References
    ----------
    .. [1] Cook, R. D., & Weisberg, S. (1982). Residuals and Influence in 
      Regression. New York: Chapman & Hall.
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
    fitted_values : `ArrayLike`
        The fitted values from a regression model.
        
    residuals : `ArrayLike`
        The residuals from a regression model.
        
    highlight : `Optional[Tuple[int, ...]]`, optional
        Indices of points to highlight in the plot.
        
    figsize : `Tuple[int, int]`, default=(6, 4)
        Size of the figure to be created.
        
    title : `str`, default='Residuals vs Fitted'
        Title of the plot.
        
    xlabel : `str`, default='Fitted values'
        Label of the x-axis.
        
    ylabel : `str`, default='Residuals'
        Label of the y-axis.
        
    linecolor : `str`, default='red'
        Color of the line to be plotted.
        
    linestyle : `str`, default='-'
        Style of the line to be plotted.
        
    scatter_kws : `Optional[dict]`, optional
        Additional keyword arguments to be passed to the `plt.scatter` method.
        
    line_kws : `Optional[dict]`, optional
        Additional keyword arguments to be passed to the `plt.plot` method.

    Returns
    -------
    `plt.Axes`
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

    Notes
    -----
    Residuals vs Fitted plot is used to identify non-linearity, unequal error 
    variances, and outliers. Ideally, residuals should be randomly scattered 
    around 0, suggesting that the model fits well.

    The residual for each observation is calculated as:

    .. math::
        e_i = y_i - \hat{y}_i

    where:
        - :math:`e_i` is the residual for the i-th observation.
        - :math:`y_i` is the actual value for the i-th observation.
        - :math:`\hat{y}_i` is the fitted value for the i-th observation.

    Patterns in the residuals vs fitted plot can indicate issues with the 
    model fit, such as non-linearity or heteroscedasticity.

    See Also
    --------
    seaborn.regplot : Plot data and a linear regression model fit.
    
    References
    ----------
    .. [1] Anscombe, F. J. (1973). Graphs in Statistical Analysis. The American Statistician, 27(1), 17-21.
    .. [2] Chatterjee, S., & Hadi, A. S. (1988). Sensitivity Analysis in Linear Regression. New York: Wiley.
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










