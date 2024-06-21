# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from ..api.types import Optional, List, Tuple
from ..api.types import DataFrame, Array1D
from ..tools.coreutils import to_series_if 
from ..tools.funcutils import make_data_dynamic, ensure_pkg

__all__= [ "kaplan_meier_analysis", "dca_analysis" ]

@ensure_pkg(
    "lifelines","The 'lifelines' package is required for this function to run.")
@make_data_dynamic("numeric", capture_columns=True, dynamize=False)
def kaplan_meier_analysis(
    durations: DataFrame | np.ndarray,
    event_observed: Array1D,
    columns: Optional[List[str]]=None, 
    as_frame: bool=False, 
    view: bool = False,
    fig_size: Tuple[int, int] = (10, 6),
    **kws
):
    """
    Perform Kaplan-Meier Survival Analysis and optionally visualize the 
    survival function.

    The Kaplan-Meier estimator, also known as the product-limit estimator, is a 
    non-parametric statistic used to estimate the survival probability from 
    observed lifetimes. It is defined as:

    .. math::
        S(t) = \prod_{i: t_i < t} \left(1 - \frac{d_i}{n_i}\right)

    where \( S(t) \) is the probability of survival until time \( t \), 
    \( d_i \) is the number of death events at time \( t_i \), and \( n_i \) 
    is the number of subjects at risk of death just prior to time \( t_i \).


    Parameters
    ----------
    durations : np.ndarray
        Observed lifetimes (durations).
    event_observed : np.ndarray
        Boolean array, where 1 indicates the event is observed (failure)
        and 0 indicates the event is censored.
    view : bool, optional
        If True, displays the Kaplan-Meier survival function plot.
    columns : List[str], optional
        Specific columns to use for the analysis if `durations` is a DataFrame.
    view : bool, optional
        If True, displays the Kaplan-Meier survival function plot.
    fig_size : Tuple[int, int], optional
        Size of the figure for the Kaplan-Meier plot.
    **kws : dict
        Additional keyword arguments passed to `lifelines.KaplanMeierFitter`.

    Returns
    -------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier estimator.

    Returns
    -------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier estimator.

    Examples
    --------
    >>> from gofast.stats.survival_reliability import kaplan_meier_analysis
    >>> durations = [5, 6, 6, 2.5, 4, 4]
    >>> event_observed = [1, 0, 0, 1, 1, 1]
    >>> kmf = kaplan_meier_analysis(durations, event_observed)

    Using a DataFrame:
    >>> df = pd.DataFrame({'duration': [5, 6, 6, 2.5, 4, 4], 'event': [1, 0, 0, 1, 1, 1]})
    >>> kmf = kaplan_meier_analysis(df['duration'], df['event'], view=True)
    """
    from lifelines import KaplanMeierFitter

    durations = durations.squeeze()  

    kmf = KaplanMeierFitter(**kws)
    kmf.fit(durations, event_observed=event_observed)
    
    if view:
        plt.figure(figsize=fig_size)
        kmf.plot_survival_function()
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.show()
        
    if as_frame: 
        return to_series_if(
            kmf , value_names=["KaplanMeier-model"], name="KM_estimate") 
    
    return kmf

@ensure_pkg("skbio", "'scikit-bio' package is required for `dca_analysis` to run.")
@make_data_dynamic(capture_columns=True, dynamize=False)
def dca_analysis(
    data,
    columns: Optional[list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = (10, 6),
    **kws
):
    """
    Perform Detrended Correspondence Analysis (DCA) on ecological data to identify
    the main gradients in species abundance or occurrence data across sites.
    Optionally, visualize the species scores in the DCA space.

    DCA is an indirect gradient analysis approach which focuses on non-linear 
    relationships among variables. It's particularly useful in ecology for 
    analyzing species distribution patterns across environmental gradients.

    .. math::
        \\text{DCA is based on the eigen decomposition: } X = U \\Sigma V^T

    Where:
    - :math:`X` is the data matrix,
    - :math:`U` and :math:`V` are the left and right singular vectors,
    - :math:`\\Sigma` is a diagonal matrix containing the singular values.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Ecological dataset for DCA. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame. Useful for specifying
        subset of data for analysis.
    as_frame : bool, optional
        If True, returns the result as a pandas DataFrame. Useful for further
        data manipulation and analysis.
    view : bool, optional
        If True, displays a scatter plot of species scores in the DCA space. 
        Helpful for visual examination of species distribution patterns.
    cmap : str, optional
        Colormap for the scatter plot. Enhances plot aesthetics.
    fig_size : tuple, optional
        Size of the figure for the scatter plot. Allows customization of the plot size.
    **kws : dict
        Additional keyword arguments passed to the DCA function in `skbio`.

    Returns
    -------
    dca_result : OrdinationResults or DataFrame
        Results of DCA, including axis scores and explained variance. The format
        of the result is determined by the `as_frame` parameter.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.stats.survival_reliability import dca_analysis
    >>> X, y = make_classification(n_samples=100, n_features=5, n_informative=2)
    >>> dca_result = dca_analysis(X, as_frame=True, view=True)
    >>> print(dca_result.head())

    This function is an essential tool for ecologists and environmental scientists 
    looking to explore and visualize complex ecological datasets, revealing patterns 
    and relationships that might not be apparent from raw data alone.
    """
    from skbio.stats.ordination import detrended_correspondence_analysis
    
    # Perform DCA
    dca_result = detrended_correspondence_analysis(data, **kws)
    
    # Visualization
    if view:
        species_scores = dca_result.samples
        plt.figure(figsize=fig_size)
        scatter = plt.scatter(species_scores.iloc[:, 0],
                              species_scores.iloc[:, 1], cmap=cmap)
        plt.title('DCA Species Scores')
        plt.xlabel('DCA Axis 1')
        plt.ylabel('DCA Axis 2')
        plt.colorbar(scatter, label='Species Abundance')
        plt.show()
    
    # Convert to DataFrame if requested
    if as_frame:
        # Assuming 'samples' attribute contains species scores 
        # which are typical in DCA results
        dca_df = pd.DataFrame(dca_result.samples, columns=['DCA Axis 1', 'DCA Axis 2'])
        return dca_df
    
    return dca_result