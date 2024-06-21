# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
These functions offer a range of probability utilities suitable for large 
datasets, leveraging the power of NumPy and SciPy for efficient computation. 
They cover various aspects of probability calculations, from simple PDF and 
CDF computations to more complex scenarios like sampling from distributions.
"""

import numpy as np
import pandas as pd 
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson
import matplotlib.pyplot as plt

from ..api.types import Array1D, ArrayLike,  DataFrame, Series, NDArray 
from ..api.types import Union, List, Optional, NumPyFunction 
from ..decorators import Extract1dArrayOrSeries, Dataify 
from ..decorators import DataTransformer   
from ..tools.funcutils import ensure_pkg 

try: 
    import pymc3 as pm
except: pass 

__all__=[ 
    "stochastic_volatility_model", "hierarchical_linear_model", 
    "normal_pdf", "normal_cdf", "binomial_pmf", "poisson_logpmf", 
    "uniform_sampling", "plot_normal_cdf", "plot_normal_cdf2", 
    "plot_normal_pdf", 
    ]

@Extract1dArrayOrSeries(column=0, method='soft', verbose=True)
@ensure_pkg("pymc3")
def stochastic_volatility_model(
    arr: Union[Array1D, Series, DataFrame], 
    column: Union[str, int] = None,  
    sigma_init: float = 1.0, 
    mu: float = 0.0, 
    sampling: bool = False,
    return_inference_data: bool = True,
    n_samples: int = 1000
 ): 
    """
    Computes a stochastic volatility model using PyMC3 for time-series data.
    
    The model captures the phenomenon of heteroskedasticity in financial 
    time-series data, representing non-constant volatility. This implementation 
    provides flexibility in adjusting the initial model parameters and supports 
    both Pandas DataFrames and NumPy ndarrays as input.
    
    The stochastic volatility model is defined by the following equations:
    
    .. math::
        y[t] \sim \mathcal{N}(0, \exp(2 \cdot v[t])),
        v[t] \sim \mathcal{N}(v[t-1], \sigma^2),
    
    where :math:`y[t]` is the observed value at time :math:`t`, and :math:`v[t]` 
    is the log volatility at time :math:`t`. :math:`\sigma` is a model parameter 
    controlling the volatility of the volatility (vol of vol).
    
    Parameters
    ----------
    arr : Union[np.ndarray, pd.Series, pd.DataFrame]
        Time-series data representing returns. If a DataFrame is passed, a 
        specific column can be selected with the `column` parameter.
    column : Union[str, int], optional
        Specifies the column to use when a DataFrame is passed. It can be either 
        the column name (str) or the column index (int). Default is None, which 
        selects the first column if a DataFrame is passed.
    sigma_init : float, optional
        Initial value for the sigma parameter in the Exponential distribution, 
        which represents the prior belief about the volatility. Default is 1.0.
    mu : float, optional
        The mean parameter for the Gaussian distribution of the observed returns.
        Default is 0.0.
    sampling : bool, optional
        If True, performs sampling on the model to generate posterior 
        distributions. Default is False.
    return_inference_data : bool, optional
        If True and `sampling` is True, returns the generated samples as 
        an ArviZ InferenceData object; otherwise, returns a MultiTrace object.
        Default is True.
    n_samples : int, optional
        The number of samples to draw in the sampling process. Default is 1000.
    
    Returns
    -------
    model : pymc3.Model
        A PyMC3 model object.
    trace : Union[pm.backends.base.MultiTrace, pm.backends.arviz.InferenceData], optional
        The trace resulting from sampling, returned only if `sampling=True`.
        The format depends on `return_inference_data`.
    
    Raises
    ------
    TypeError
        If the input data type is not numeric.
    
    Examples
    --------
    >>> from gofast.stats.probs import stochastic_volatility_model
    >>> import numpy as np
    >>> returns = np.random.normal(0, 1, 100)
    >>> model = stochastic_volatility_model(returns)
    >>> model, trace = stochastic_volatility_model(returns, sampling=True, n_samples=1000)
    """
    import pymc3 as pm

    if not isinstance(arr, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a NumPy ndarray or Pandas Series.")

    with pm.Model() as model:
        sigma = pm.Exponential('sigma', 1.0 / sigma_init)
        v = pm.GaussianRandomWalk('v', sigma=sigma, shape=len(arr))
        pm.Normal('y', mu=mu, sd=pm.math.exp(v), observed=arr)

        if sampling:
            with model:
                trace = pm.sample(n_samples, return_inferencedata=return_inference_data)
    if sampling:
        return model, trace
    else:
        return model

@ensure_pkg("pymc3")
def hierarchical_linear_model(
    X: ArrayLike, 
    y: Array1D, 
    groups: ArrayLike,
    mu: float = 0.0,
    sd_mu_beta: float = 10.0,
    sd_sigma_beta: float = 10.0,
    sd_sigma: float = 10.0,
    return_inference_data: bool =True,
    sampling: bool=False, 
    n_samples: int = 1000
): 
    """
    Builds a hierarchical linear model using PyMC3, ideal for grouped or 
    hierarchical data.

    Hierarchical or multilevel models are a type of regression model where 
    parameters are designed to vary at more than one level, making them 
    suitable for datasets that involve groups or hierarchies.
    
    The hierarchical linear model can be expressed using the following equations:

    .. math::
        y[i] \sim \mathcal{N}(X[i] \cdot \beta[\text{{group}}[i]], \sigma[\text{{group}}[i]])

    .. math::
        \beta[\text{{group}}[i]] \sim \mathcal{N}(\mu_{\beta}, \sigma_{\beta})

    where :math:`y[i]` is the observed value, :math:`X[i]` is the predictor
    for observation :math:`i`,  and :math:`\beta[\text{{group}}[i]]` is the 
    group-specific coefficient.
    

    Parameters
    ----------
    X : np.ndarray
        The predictor variables matrix.
    y : np.ndarray
        The response variable vector.
    groups : np.ndarray
        An array-like object indicating the grouping variable for hierarchy. 
        Each element corresponds to the group of the respective observation in X and y.
    mu : float, optional
        The mean value for the normal distribution of `mu_beta`. 
        Default is 0.0.
    sd_mu_beta : float, optional
        The standard deviation for the normal distribution of `mu_beta`.
        Default is 10.0.
    sd_sigma_beta : float, optional
        The standard deviation for the half-normal distribution of `sigma_beta`. 
        Default is 10.0.
    sd_sigma : float, optional
        The standard deviation for the half-normal distribution of `sigma`.
        Default is 10.0.
    return_inference_data : bool, optional
        If True and sampling is performed, the function returns an ArviZ 
        InferenceData object; otherwise, it returns a MultiTrace object. 
        Default is True.
    sampling : bool, optional
        If True, performs sampling on the model to generate posterior 
        distributions. Default is False.
    n_samples : int, optional
        The number of posterior samples to draw. Default is 1000.
    Returns
    -------
    model : pm.Model
        The PyMC3 model object, ready for sampling.
    trace : Union[pm.backends.base.MultiTrace, pm.backends.arviz.InferenceData], optional
        The trace resulting from the sampling process, returned only if 
        sampling is performed. The format depends on `return_inference_data`.

    Example
    -------
    >>> import numpy as np
    >>> from gofast.stats.probs import hierarchical_linear_model
    >>> X = np.random.normal(size=(100, 3))
    >>> y = np.random.normal(size=100)
    >>> groups = np.random.randint(0, 5, size=100)
    >>> model, trace = hierarchical_linear_model(X, y, groups, n_samples=1000)
    """
    
    unique_groups = np.unique(groups)
    group_idx = np.array([np.where(unique_groups == g)[0][0] for g in groups])

    with pm.Model() as model:
        # Hyperpriors
        mu_beta = pm.Normal('mu_beta', mu=0, sd=10, shape=X.shape[1])
        sigma_beta = pm.HalfNormal('sigma_beta', 10)
        sigma = pm.HalfNormal('sigma', sd=10, shape=len(unique_groups))

        # Group-specific intercepts and slopes
        beta = pm.Normal('beta', mu=mu_beta, sd=sigma_beta, shape=(
            len(unique_groups), X.shape[1]))
        # Expected value per observation
        y_est = pm.math.dot(X, beta[group_idx].T)

        # Model error
        pm.Normal('y_obs', mu=y_est, sd=sigma[group_idx], observed=y)

        if sampling and n_samples > 0:
            trace = pm.sample(n_samples, return_inferencedata=return_inference_data)
            return model, trace

    return model

@DataTransformer(mode='lazy')
@Dataify(enforce_df=False, ignore_mismatch=True, fail_silently=True ) 
def normal_pdf(
    data: ArrayLike, 
    mean: float = 0, 
    std_dev: float = 1, 
    scale: float = 1, 
    loc: float = 0,
    columns: List[str]=None, 
    **kws
    )-> ArrayLike:
    """
    Compute the Probability Density Function (PDF) for a normal 
    distribution.

    The formula for the PDF of a normal distribution is:

    .. math::
        f(x | \mu, \sigma) = \frac{1}{\sigma \sqrt{2 \pi}}
        \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)

    Where :math:`\mu` is the mean, :math:`\sigma` is the standard deviation, 
    and :math:`x` is the data point.

    Parameters
    ----------
    data : ndarray
        Data for which the PDF is to be computed.
    mean : float, optional
        Mean (:math:`\mu`) of the normal distribution. Default is 0.
    std_dev : float, optional
        Standard deviation (:math:`\sigma`) of the normal distribution. 
        Default is 1.
    scale : float, optional
        Scaling factor for the PDF, not part of the standard normal 
        distribution formula. Applied as a multiplier to the result. 
        Default is 1.
    loc : float, optional
        Location parameter, used to shift the PDF along the x-axis. 
        Applied before scaling. Default is 0.
    columns : List[str], optional
        Specifies the columns of the dataframe to be used if `data` is a 
        DataFrame. Useful for operations on specific DataFrame columns.
    kws: dict, 
       Keyword arguments passed to :func:`scipy.stats.norm.pdf`. 
    Returns
    -------
    pdf_values : ndarray
        PDF values for the input data, scaled and shifted as specified.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.probs import normal_pdf 
    >>> data = np.array([1, 2, 3])
    >>> normal_pdf(data, mean=0, std_dev=1)
    array([...])
    
    To apply scaling and location shift:

    >>> normal_pdf(data, mean=0, std_dev=1, scale=0.5, loc=2)
    array([...])
    """
    shifted_data = (data - loc) / std_dev
    pdf_values = norm.pdf(shifted_data, mean, 1, **kws) * scale
    return pdf_values

@DataTransformer(mode='lazy')
@Dataify(enforce_df=False, ignore_mismatch=True, fail_silently=True) 
def normal_cdf(
    data: ArrayLike, 
    mean: float = 0, 
    std_dev: float = 1.0, 
    scale: float = 1.0, 
    shift: float = 0.0, 
    columns: List[str] = None
):
    """
    Compute the Cumulative Distribution Function (CDF) for a normal 
    distribution, with optional scaling and shifting of the input data.

    The CDF of a normal distribution at a point x is given by:
    
    .. math::
        F(x; \\mu, \\sigma) = \\frac{1}{2} \\left[1 + \\text{erf}\\left(\
        \\frac{x - \\mu}{\\sigma \\sqrt{2}}\\right)\\right]
            
    where :math:`\mu` is the mean, :math:`\sigma` is the standard deviation, 
    and erf is the error function.

    Parameters
    ----------
    data : ArrayLike
        Data for which the CDF is to be computed.
    mean : float, optional
        Mean (:math:`\mu`) of the normal distribution. Defaults to 0.
    std_dev : float, optional
        Standard deviation (:math:`\sigma`) of the normal distribution.
        Defaults to 1.0.
    scale : float, optional
        Scaling factor to be applied to the data. Defaults to 1.0.
    shift : float, optional
        Shift to be applied to the data after scaling. Defaults to 0.0.
    columns : List[str], optional
        Specifies the columns of the dataframe to be used if `data` is a 
        DataFrame. Useful for operations on specific DataFrame columns.
    kws: dict, 
       Keyword arguments passed to :func:`scipy.stats.norm.cdf`. 
    Returns
    -------
    ndarray
        CDF values for the input data, after optional scaling and shifting.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3])
    >>> normal_cdf(data, mean=0, std_dev=1)
    array([...])

    With scaling and shifting:

    >>> normal_cdf(data, mean=0, std_dev=1, scale=2.0, shift=1.0)
    array([...])
    """
    scaled_data = scale * data + shift
    return norm.cdf(scaled_data, mean, std_dev)

def binomial_pmf(
    trials: int, 
    p_success: float, 
    n_successes: Union[int, List[int]], 
    return_type: Union[list, NumPyFunction] = np.array
    ) -> Union[float, List, ArrayLike]:
    """
    Compute the Probability Mass Function (PMF) for a binomial distribution,
    supporting vectorized inputs and custom return types. 
    The PMF is defined as:
    
    .. math::
        P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}
    
    where :math:`n` is the number of trials, :math:`k` is the number of 
    successful trials, and :math:`p` is the probability of success on a 
    single trial.

    Parameters
    ----------
    trials : int
        Number of trials, where `trials` must be a non-negative integer.
    p_success : float
        Probability of success on each trial, where `p_success` must 
        be in the range [0, 1].
    n_successes : int or List[int]
        The number(s) of successes for which to compute the PMF.
    return_type : type, optional
        The type of the returned object (e.g., `list`, `np.ndarray`). 
        The default is `np.ndarray`. If ``None`` returns `pmf_values` instead.
    Returns
    -------
    float, T
        PMF values for the given number(s) of successes, returned in 
        the specified `return_type`.
    Notes
    -----
    - This function uses `scipy.stats.binom.pmf` for PMF calculation.
    - The function supports both single integer and list of integers as 
      `n_successes` to allow
      computing multiple PMF values in one call.
      
    Examples
    --------
    >>> from gofast.stats.probs import binomial_pmf
    >>> binomial_pmf(trials=10, p_success=0.5, n_successes=5)
    array(0.24609375)

    >>> binomial_pmf(trials=10, p_success=0.5, n_successes=[0, 5, 10], return_type=list)
    [0.0009765625, 0.24609375, 0.0009765625]

    """
    if not (0 <= p_success <= 1):
        raise ValueError("p_success must be between 0 and 1.")
    if not isinstance(trials, int) or trials < 0:
        raise ValueError("trials must be a non-negative integer.")
    if hasattr(n_successes, '__iter__') and not isinstance(n_successes, str):
        pmf_values = [binom.pmf(n, trials, p_success) for n in n_successes]
    else:
        pmf_values = binom.pmf(n_successes, trials, p_success)
    
    return pmf_values if return_type is None else return_type(pmf_values)

def poisson_logpmf(
    data: ArrayLike, 
    lambda_param: float,
    *args, **kws
    ) -> ArrayLike:
    """
    Compute the logarithm of the Probability Mass Function (PMF) for a 
    Poisson distribution.
    
    The log PMF of a Poisson distribution for a given value `k` is given by:
    
    .. math::
        \\log(P(k; \\lambda)) = k \\log(\\lambda) - \\lambda - \\log(k!)
    
    where :math:`\\lambda` is the expected number of occurrences within a 
    fixed interval and `k` is the actual number of occurrences observed.
    
    The function is useful for computing log PMF values when the PMF values are
    extremely small and might underflow when represented as normal 
    floating-point numbers.

    Parameters
    ----------
    data : ArrayLike
        The observed data for which the log PMF is to be computed.
        Can be a single value or an array of values.
    lambda_param : float
        The rate parameter (\\lambda) of the Poisson distribution, which
        is the expected number of occurrences. Must be non-negative.
    *args : Additional positional arguments for `scipy.stats.poisson.logpmf`.
    **kws : Additional keyword arguments for `scipy.stats.poisson.logpmf`.

    Returns
    -------
    np.ndarray
        The log PMF values for the input data, computed using 
        the specified \\lambda.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3])
    >>> poisson_logpmf(data, lambda_param=2)
    array([...])

    Computing for a single value:
    
    >>> poisson_logpmf(4, lambda_param=2)
    ...

    Using additional arguments:
    
    >>> poisson_logpmf(data, lambda_param=2, loc=0)  # loc is an example of *args or **kws
    array([...])

    Notes
    -----
    The Poisson distribution is appropriate for data that represent counts 
    per unit of time or space, assuming that these events occur with a 
    constant mean rate and independently of the time since the last event.
    """
    # Check if lambda_param is non-negative
    if lambda_param < 0:
        raise ValueError("`lambda_param` must be non-negative.")
    
    # Convert data to a numpy array to handle both scalar and array inputs
    data_array = np.asarray(data)
    
    # Check if data contains only non-negative values
    if np.any(data_array < 0):
        raise ValueError("`data` must contain non-negative values.")
    
    # Check if data contains integers or values that can be considered 
    # as integers (e.g., float with .0)
    if not np.all(np.mod(data_array, 1) == 0):
        raise ValueError("`data` must contain integer values or values "
                         "that can be considered integers.")
    return poisson.logpmf(data_array, lambda_param, *args, **kws)

def uniform_sampling(
    low: float, 
    high: float, 
    size: int
    ) -> NDArray[float]:
    """
    Generate samples from a uniform distribution over the interval `[low, high)`.
    The uniform distribution is defined as:

    .. math::
        f(x; a, b) = \\frac{1}{b - a}

    for :math:`a \\leq x < b`, and 0 otherwise, where `a` is the lower boundary,
    and `b` is the upper boundary of the distribution.

    Parameters
    ----------
    low : float
        Lower boundary (:math:`a`) of the distribution.
    high : float
        Upper boundary (:math:`b`) of the distribution. 
        Must be greater than `low`.
    size : int
        Number of samples to generate.

    Returns
    -------
    NDArray[float]
        An array of samples drawn from the uniform distribution.

    Examples
    --------
    Generate five samples from a uniform distribution ranging from 0 to 10:

    >>> uniform_sampling(low=0, high=10, size=5)
    array([...])

    Generate a single sample from a uniform distribution ranging from -5 to 5:

    >>> uniform_sampling(low=-5, high=5, size=1)
    array([...])

    Notes
    -----
    This function uses `numpy.random.uniform` to generate random samples,
    which draws samples from a uniform distribution over the specified interval.
    Each sample is equally likely to be drawn from any part within the interval.
    """
    # Check and convert singleton numpy arrays to floats for 'low' and 'high'
    if isinstance(low, np.ndarray) and low.size == 1:
        low = float(low)
    elif not isinstance(low, (float, int)):
        raise TypeError("`low` must be a numeric value or a singleton numpy array.")
    
    if isinstance(high, np.ndarray) and high.size == 1:
        high = float(high)
    elif not isinstance(high, (float, int)):
        raise TypeError("`high` must be a numeric value or a singleton numpy array.")

    # Check if 'size' is an integer
    if not isinstance(size, int):
        raise TypeError("`size` must be an integer.")
    
    # Ensure 'low' and 'high' are treated as floats
    low = float(low)
    high = float(high)
    
    # Validate the logical relationship between 'high' and 'low'
    if high <= low:
        raise ValueError("The `high` parameter must be greater than `low`.")
    
    # Ensure 'size' is a positive integer
    if size < 1:
        raise ValueError("The `size` parameter must be a positive integer.")
    
    return np.random.uniform(low, high, size)

def visualize_uniform_sampling(
    low: float, 
    high: float, 
    size: int,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Generates uniform samples and visualizes them with a histogram on a 
    given Matplotlib Axes object or a new figure, ensuring the `low`, 
    `high`, and `size` parameters are within valid ranges.

    Parameters
    ----------
    low : float
        The lower boundary of the distribution.
    high : float
        The upper boundary of the distribution. Must be greater than `low`.
    size : int
        The number of samples to generate. Must be positive.
    ax : plt.Axes, optional
        The Matplotlib Axes object to plot on. If None, a new figure 
        and axes are created.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object with the histogram.

    Raises
    ------
    ValueError
        If `low` >= `high` or if `size` <= 0.

    Examples
    --------
    Visualizing with default axes:
    
    >>> from gofast.stats.probs import visualize_uniform_sampling
    >>> visualize_uniform_sampling(low=0, high=10, size=1000)
    
    Visualizing on an existing Axes object with custom parameters:
    
    >>> fig, ax = plt.subplots()
    >>> visualize_uniform_sampling(low=10, high=20, size=500, ax=ax)
    """
    if high <= low:
        raise ValueError("`high` must be greater than `low`.")
    if size <= 0:
        raise ValueError("`size` must be a positive integer.")
    
    samples = np.random.uniform(low, high, size)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title('Histogram of Uniformly Distributed Samples')
    ax.set_xlabel('Sample value')
    ax.set_ylabel('Frequency')
    ax.axvline(x=low, color='red', linestyle='--', label='Low (min value)')
    ax.axvline(x=high, color='green', linestyle='--', label='High (max value)')
    ax.legend()

    return ax

def plot_normal_cdf2(
    mean=0, 
    std_dev=1, 
    x_range=None, 
    resolution=1000, 
    title='Normal Distribution CDF', xlabel='x', ylabel='CDF', 
    figsize=(8, 6), 
    line_style='-', 
    line_color='blue', 
    line_width=2
    ):
    """
    Plot the Cumulative Distribution Function (CDF) of a normal distribution.

    Parameters
    ----------
    mean : float, optional
        The mean (\mu) of the normal distribution. Defaults to 0.
    std_dev : float, optional
        The standard deviation (\sigma) of the normal distribution. 
        Defaults to 1.
    x_range : tuple(float, float), optional
        The range of x values over which to plot the CDF. Defaults to 
        mean \u00B1 4*std_dev.
    resolution : int, optional
        The number of points to calculate the CDF at, increasing this improves
        the plot quality. Defaults to 1000.
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    figsize : tuple(int, int), optional
        The size of the figure (width, height) in inches. Defaults to (8, 6).
    line_style : str, optional
        The style of the plot line (e.g., '-', '--', ':'). Defaults to '-'.
    line_color : str, optional
        The color of the plot line. Defaults to 'blue'.
    line_width : float, optional
        The width of the plot line. Defaults to 2.

    """
    if x_range is None:
        x_range = (mean - 4*std_dev, mean + 4*std_dev)
    x_values = np.linspace(x_range[0], x_range[1], resolution)
    cdf_values = norm.cdf(
    x_values, 
    mean, 
    std_dev
    )

    plt.figure(figsize=figsize)
    plt.plot(x_values, cdf_values, line_style, color=line_color, 
             linewidth=line_width)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    
def plot_normal_cdf(
    mean: float = 0, 
    std_dev: float = 1.0, 
    scale: float = 1.0, 
    shift: float = 0.0, 
    range_min: float = -5, 
    range_max: float = 5, 
    num_points: int = 1000,
    cdf_values: Optional[np.ndarray] = None,
    x_values: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize the Cumulative Distribution Function (CDF) of a normal 
    distribution on a given Matplotlib Axes object or a new figure, with 
    optional scaling and shifting.

    Parameters
    ----------
    mean : float, optional
        Mean (μ) of the normal distribution. Defaults to 0.
    std_dev : float, optional
        Standard deviation (σ) of the normal distribution. Must be
        positive. Defaults to 1.0.
    scale : float, optional
        Scaling factor for the CDF, not part of the standard normal 
        distribution formula. Applied as a multiplier to the result. 
        Default is 1.0.
    shift : float, optional
        Shift applied to the data after scaling. Default is 0.0.
    range_min : float, optional
        Minimum value of the range over which to plot the CDF.
    range_max : float, optional
        Maximum value of the range over which to plot the CDF.
    num_points : int, optional
        Number of points to compute the CDF at within the range.
    ax : plt.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure 
        and axes are created.

    Returns
    -------
    plt.Axes
        The Matplotlib Axes object with the plot.

    Examples
    --------
    Plotting with default parameters:
    
    >>> plot_normal_cdf()
    
    Plotting on an existing Axes object with custom parameters:
    
    >>> fig, ax = plt.subplots()
    >>> plot_normal_cdf(mean=2, std_dev=2, scale=0.5, shift=1, ax=ax)
    """
    if cdf_values is not None and x_values is None:
        raise ValueError("`x_values` must be provided when `cdf_values` is given.")
        
    if cdf_values is None: 
        if std_dev <= 0:
            raise ValueError("Standard deviation `std_dev` must be positive.")
        if num_points <= 0:
            raise ValueError("Number of points `num_points` must be positive.")
        if range_min >= range_max:
            raise ValueError("`range_min` must be less than `range_max`.")
        
        x_values = np.linspace(range_min, range_max, num_points)
        scaled_x = scale * (x_values + shift)
        cdf_values = norm.cdf(scaled_x, mean, std_dev)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, cdf_values, label='CDF', color='blue')
    ax.set_title('Normal Distribution CDF')
    ax.set_xlabel('Value')
    ax.set_ylabel('CDF')
    ax.grid(True)
    ax.legend()
    return ax

def plot_normal_pdf(
    mean: float = 0, 
    std_dev: float = 1.0, 
    scale: float = 1.0, 
    loc: float = 0.0, 
    range_min: float = -5, 
    range_max: float = 5, 
    num_points: int = 1000,
    pdf_values: Optional[np.ndarray] = None,
    x_values: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Visualize the Probability Density Function (PDF) of a normal 
    distribution, with optional scaling and shifting, on a given Matplotlib
    Axes object or a new figure.

    Parameters
    ----------
    mean : float, optional
        Mean (μ) of the normal distribution. Defaults to 0.
    std_dev : float, optional
        Standard deviation (σ) of the normal distribution. Must be
        positive. Defaults to 1.0.
    scale : float, optional
        Scaling factor for the PDF, not part of the standard normal
        distribution formula. Applied as a multiplier to the result.
        Default is 1.
    loc : float, optional
        Location parameter, used to shift the PDF along the x-axis.
        Applied before scaling. Default is 0.
    range_min : float, optional
        The minimum value of the range over which to plot the PDF.
    range_max : float, optional
        The maximum value of the range over which to plot the PDF.
    num_points : int, optional
        The number of points to compute the PDF at within the range.
    pdf_values : np.ndarray, optional
        Pre-calculated PDF values. If provided, `mean`, `std_dev`,
        `scale`, `loc`, `range_min`, `range_max`, and `num_points` are ignored.
    x_values : np.ndarray, optional
        The x-values corresponding to `pdf_values`. Required if `pdf_values`
        is provided.
        
    ax : plt.Axes, optional
        The Matplotlib Axes object to plot on. If None, a new figure
        and axes are created.
    Returns
    -------
    plt.Axes
        The Matplotlib Axes object with the plot.

    Examples
    --------
    Plotting with default parameters:
    
    >>> from gofast.stats.probs import plot_normal_pdf
    >>> plot_normal_pdf()
    
    Plotting on an existing Axes object with custom parameters:
    
    >>> fig, ax = plt.subplots()
    >>> plot_normal_pdf(mean=2, std_dev=2, loc=0, scale=0.5, ax=ax)
    """
    if pdf_values is not None and x_values is None:
        raise ValueError("`x_values` must be provided when `pdf_values` is given.")
    
    if pdf_values is None:
        if std_dev <= 0:
            raise ValueError("`std_dev` must be positive.")
        if num_points <= 0:
            raise ValueError("`num_points` must be positive.")
        if range_min >= range_max:
            raise ValueError("`range_min` must be less than `range_max`.")

        x_values = np.linspace(range_min, range_max, num_points)
        shifted_data = (x_values - loc) / std_dev
        pdf_values = norm.pdf(shifted_data, mean, std_dev) * scale
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_values, pdf_values, label='PDF', color='blue')
    ax.set_title('Normal Distribution PDF')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True)
    ax.legend()
    
    return ax


if __name__=='__main__':
    # Example of using the function
    plot_normal_cdf(mean=0, std_dev=1)
    # Example usage
    visualize_uniform_sampling(low=0, high=10, size=1000)
    # Example usage of the function
    plot_normal_cdf(mean=0, std_dev=1.0, scale=1.0, shift=0.0)
    # Example usage of the function
    plot_normal_pdf(mean=0, std_dev=1.0, scale=1.0, loc=0.0)
    # Example usage of the function

    visualize_uniform_sampling(low=0, high=10, size=1000)