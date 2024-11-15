.. _survival_reliability:

Survival and Reliability Analysis
===============================

.. currentmodule:: gofast.stats.survival_reliability

The :mod:`gofast.stats.survival_reliability` module provides specialized functions for survival and reliability analysis, including Kaplan-Meier survival analysis and decision curve analysis (DCA). This module implements robust methods for time-to-event analysis, reliability metrics, and clinical decision making.

Key Features
------------
- **Survival Analysis**:
  Comprehensive tools for analyzing time-to-event data.

  - :func:`~gofast.stats.survival_reliability.kaplan_meier_analysis`: Non-parametric survival function estimation
  - :func:`~gofast.stats.survival_reliability.cox_proportional_hazards`: Cox regression modeling
  - :func:`~gofast.stats.survival_reliability.nelson_aalen`: Nelson-Aalen cumulative hazard estimation

- **Reliability Metrics**:
  Methods for assessing system and component reliability.

  - :func:`~gofast.stats.survival_reliability.reliability_function`: Reliability function estimation
  - :func:`~gofast.stats.survival_reliability.hazard_rate`: Hazard rate calculation
  - :func:`~gofast.stats.survival_reliability.mean_time_to_failure`: MTTF estimation

- **Clinical Decision Analysis**:
  Tools for evaluating clinical decision strategies.

  - :func:`~gofast.stats.survival_reliability.dca_analysis`: Decision curve analysis
  - :func:`~gofast.stats.survival_reliability.net_benefit`: Net benefit calculation
  - :func:`~gofast.stats.survival_reliability.threshold_probability`: Decision threshold analysis

Function Descriptions
--------------------

kaplan_meier_analysis
~~~~~~~~~~~~~~~~~~~
Perform Kaplan-Meier Survival Analysis and optionally visualize the survival function. The Kaplan-Meier estimator, also known as the product-limit estimator, is a non-parametric statistic used to estimate the survival probability from observed lifetimes.

Mathematical Expression:

.. math::

    S(t) = \prod_{i: t_i < t} \left(1 - \frac{d_i}{n_i}\right)

where:
- S(t) is the probability of survival until time t
- d_i is the number of death events at time t_i
- n_i is the number of subjects at risk just prior to time t_i

Parameters:
    - durations (DataFrame | ndarray): Observed times
    - event_observed (Array1D): Event indicator (1 if event occurred, 0 if censored)
    - columns (Optional[List[str]]): Column names for durations
    - as_frame (bool): Return results as DataFrame
    - view (bool): Display survival curve plot
    - fig_size (Tuple[int, int]): Figure dimensions for plot

Returns:
    - KMResult: Survival analysis results including survival function estimates

Examples:

.. code-block:: python

    from gofast.stats.survival_reliability import kaplan_meier_analysis
    import numpy as np
    import pandas as pd

    # Example 1: Basic Kaplan-Meier analysis
    times = np.array([2, 3, 5, 6, 8, 10])
    events = np.array([1, 0, 1, 1, 0, 1])
    km_results = kaplan_meier_analysis(times, events)
    print("Survival Probabilities:", km_results.survival_function)

    # Example 2: Analysis with visualization
    data = pd.DataFrame({
        'time': np.random.exponential(50, 200),
        'event': np.random.binomial(1, 0.7, 200)
    })
    km_results = kaplan_meier_analysis(
        data['time'],
        data['event'],
        view=True,
        fig_size=(10, 6)
    )

    # Example 3: Comparing multiple groups
    groups = np.random.choice(['A', 'B'], 200)
    times = np.random.exponential(50, 200)
    events = np.random.binomial(1, 0.7, 200)
    
    for group in ['A', 'B']:
        mask = (groups == group)
        km_results = kaplan_meier_analysis(
            times[mask],
            events[mask],
            view=True,
            label=f'Group {group}'
        )

dca_analysis
~~~~~~~~~~~
Perform Decision Curve Analysis (DCA) to evaluate prediction models or diagnostic tests. DCA is a method for evaluating and comparing prediction models that accounts for clinical preferences through a range of threshold probabilities.

Mathematical Expression:

.. math::

    \text{Net Benefit} = \frac{\text{True Positives}}{n} - 
    \frac{\text{False Positives}}{n} \times \frac{p_t}{1-p_t}

where:
- n is the total number of patients
- p_t is the threshold probability
- True/False Positives are counted at the given threshold

Parameters:
    - probabilities (array-like): Predicted probabilities
    - outcomes (array-like): Actual outcomes
    - threshold_range (tuple): Range of threshold probabilities
    - view (bool): Display DCA curve

Returns:
    - DCAResult: Net benefit calculations and curve data

Examples:

.. code-block:: python

    from gofast.stats.survival_reliability import dca_analysis

    # Example 1: Basic DCA
    predictions = np.random.random(100)
    outcomes = np.random.binomial(1, predictions)
    dca_results = dca_analysis(predictions, outcomes)
    print("Net Benefit at 0.5 threshold:", dca_results.net_benefit[50])

    # Example 2: Comparing multiple models
    model1_pred = np.random.random(100)
    model2_pred = np.random.random(100)
    true_outcomes = np.random.binomial(1, 0.3, 100)

    models = {
        'Model 1': model1_pred,
        'Model 2': model2_pred
    }

    for name, preds in models.items():
        dca_results = dca_analysis(
            preds,
            true_outcomes,
            view=True,
            label=name
        )

reliability_function
~~~~~~~~~~~~~~~~~
Estimate the reliability function (survival function) from lifetime data.

Mathematical Expression:

.. math::

    R(t) = P(T > t) = 1 - F(t)

where:
- T is the lifetime random variable
- F(t) is the cumulative distribution function

Parameters:
    - lifetimes (array-like): Observed lifetimes
    - censoring (array-like): Censoring indicators
    - method (str): Estimation method ('km', 'nelson-aalen')

Examples:

.. code-block:: python

    from gofast.stats.survival_reliability import reliability_function

    # Example: Reliability function estimation
    lifetimes = np.random.weibull(2, 100)
    censoring = np.random.binomial(1, 0.8, 100)
    rel_est = reliability_function(lifetimes, censoring)
    print("Reliability at median time:", rel_est.at_time(np.median(lifetimes)))

hazard_rate
~~~~~~~~~~
Calculate the hazard rate (failure rate) from lifetime data.

Mathematical Expression:

.. math::

    h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t | T \geq t)}{\Delta t}

Parameters:
    - lifetimes (array-like): Observed lifetimes
    - events (array-like): Event indicators
    - bandwidth (float): Smoothing bandwidth for estimation

Examples:

.. code-block:: python

    from gofast.stats.survival_reliability import hazard_rate

    # Example: Hazard rate calculation
    times = np.random.exponential(50, 100)
    events = np.random.binomial(1, 0.7, 100)
    hz_rate = hazard_rate(times, events)
    print("Hazard rate at t=10:", hz_rate.at_time(10))

mean_time_to_failure
~~~~~~~~~~~~~~~~~
Estimate the Mean Time To Failure (MTTF) from lifetime data.

Mathematical Expression:

.. math::

    MTTF = \int_0^\infty R(t) dt

Parameters:
    - lifetimes (array-like): Observed lifetimes
    - censoring (array-like): Censoring indicators
    - method (str): Estimation method

Examples:

.. code-block:: python

    from gofast.stats.survival_reliability import mean_time_to_failure

    # Example: MTTF estimation
    lifetimes = np.random.weibull(2, 100)
    censoring = np.random.binomial(1, 0.8, 100)
    mttf = mean_time_to_failure(lifetimes, censoring)
    print("Estimated MTTF:", mttf)

Best Practices
-------------
1. **Data Preparation**:
   - Handle censored observations appropriately
   - Check for temporal bias in follow-up
   - Validate proportional hazards assumption when applicable

2. **Analysis Considerations**:
   - Use appropriate methods for heavy censoring
   - Consider competing risks when relevant
   - Validate model assumptions

3. **Visualization Guidelines**:
   - Include confidence intervals in survival curves
   - Mark censored observations on plots
   - Show number at risk in time intervals

See Also
--------
- :mod:`gofast.stats.inferential`: For statistical testing
- :mod:`gofast.visualization`: For additional plotting utilities
- :mod:`gofast.metrics`: For performance metrics

References
----------
.. [1] Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations.
       Journal of the American Statistical Association, 53(282), 457-481.

.. [2] Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for
       evaluating prediction models. Medical Decision Making, 26(6), 565-574.

.. [3] Klein, J. P., & Moeschberger, M. L. (2003). Survival Analysis: Techniques for Censored
       and Truncated Data. Springer Science & Business Media.