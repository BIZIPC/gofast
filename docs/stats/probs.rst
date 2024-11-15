.. _probs:

Probability Functions
=====================

.. currentmodule:: gofast.stats.probs

The :mod:`gofast.stats.probs` module provides a comprehensive suite of probability functions and distributions. This module implements various probability calculations, distributions, and sampling methods optimized for large-scale statistical computations.

Key Features
------------
- **Distribution Functions**:
  Core probability distribution functions including PDF, CDF, and quantile functions.

  - :func:`~gofast.stats.probs.normal_pdf`: Probability density function for normal distribution
  - :func:`~gofast.stats.probs.normal_cdf`: Cumulative distribution function for normal distribution
  - :func:`~gofast.stats.probs.binomial_pmf`: Probability mass function for binomial distribution
  - :func:`~gofast.stats.probs.poisson_pmf`: Probability mass function for Poisson distribution

- **Sampling Methods**:
  Functions for generating random samples from various distributions.

  - :func:`~gofast.stats.probs.uniform_sampling`: Generate uniform random samples
  - :func:`~gofast.stats.probs.importance_sampling`: Perform importance sampling
  - :func:`~gofast.stats.probs.rejection_sampling`: Implement rejection sampling method

- **Stochastic Processes**:
  Tools for modeling and simulating stochastic processes.

  - :func:`~gofast.stats.probs.markov_chain`: Generate Markov chain sequences
  - :func:`~gofast.stats.probs.random_walk`: Simulate random walk processes
  - :func:`~gofast.stats.probs.brownian_motion`: Generate Brownian motion paths

- **Advanced Probability Models**:
  Implementation of complex probability models and calculations.

  - :func:`~gofast.stats.probs.bayesian_update`: Perform Bayesian probability updates
  - :func:`~gofast.stats.probs.mixture_model`: Create and sample from mixture models

Function Descriptions
----------------------

normal_pdf
~~~~~~~~~~~
Calculates the probability density function (PDF) for the normal distribution.

Mathematical Expression:

.. math::

    f(x|\mu,\sigma) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

where:
- μ is the mean
- σ is the standard deviation

Parameters:
    - x (array-like): Points at which to evaluate the PDF
    - mu (float): Mean of the distribution
    - sigma (float): Standard deviation of the distribution

Returns:
    - ndarray: PDF values at x

Examples:

.. code-block:: python

    from gofast.stats.probs import normal_pdf
    import numpy as np

    # Example 1: Standard normal distribution
    x = np.linspace(-3, 3, 100)
    pdf = normal_pdf(x, mu=0, sigma=1)
    print(f"PDF at x=0: {pdf[50]:.4f}")  # Should be ~0.3989

    # Example 2: Custom normal distribution
    pdf = normal_pdf(x, mu=1, sigma=2)
    
    # Example 3: Multiple distributions
    pdfs = {
        'Standard': normal_pdf(x, 0, 1),
        'Shifted': normal_pdf(x, 1, 1),
        'Wider': normal_pdf(x, 0, 2)
    }

normal_cdf
~~~~~~~~~~
Computes the cumulative distribution function (CDF) for the normal distribution.

Mathematical Expression:

.. math::

    F(x|\mu,\sigma) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]

Parameters:
    - x (array-like): Points at which to evaluate the CDF
    - mu (float): Mean of the distribution
    - sigma (float): Standard deviation of the distribution

Examples:

.. code-block:: python

    from gofast.stats.probs import normal_cdf

    # Example 1: Standard normal CDF
    x = np.linspace(-3, 3, 100)
    cdf = normal_cdf(x)
    print(f"CDF at x=0: {cdf[50]:.4f}")  # Should be ~0.5

    # Example 2: Probability calculations
    prob = normal_cdf(1) - normal_cdf(-1)
    print(f"P(-1 < X < 1): {prob:.4f}")

binomial_pmf
~~~~~~~~~~~
Calculates the probability mass function (PMF) for the binomial distribution.

Mathematical Expression:

.. math::

    P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}

Parameters:
    - k (int): Number of successes
    - n (int): Number of trials
    - p (float): Probability of success

Examples:

.. code-block:: python

    from gofast.stats.probs import binomial_pmf

    # Example 1: Basic probability calculation
    prob = binomial_pmf(k=3, n=10, p=0.5)
    print(f"P(X=3) for Bin(10,0.5): {prob:.4f}")

    # Example 2: Multiple probabilities
    k_values = np.arange(11)
    probs = [binomial_pmf(k, n=10, p=0.5) for k in k_values]

poisson_pmf
~~~~~~~~~~
Computes the probability mass function for the Poisson distribution.

Mathematical Expression:

.. math::

    P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}

Parameters:
    - k (int): Number of events
    - lambda_ (float): Average rate of events

Examples:

.. code-block:: python

    from gofast.stats.probs import poisson_pmf

    # Example 1: Single probability
    prob = poisson_pmf(k=2, lambda_=3)
    print(f"P(X=2) for Poisson(3): {prob:.4f}")

    # Example 2: Distribution plot
    k_range = np.arange(10)
    probs = [poisson_pmf(k, lambda_=3) for k in k_range]

uniform_sampling
~~~~~~~~~~~~~~
Generates uniform random samples within specified bounds.

Parameters:
    - size (int): Number of samples
    - low (float): Lower bound
    - high (float): Upper bound
    - seed (int): Random seed for reproducibility

Examples:

.. code-block:: python

    from gofast.stats.probs import uniform_sampling

    # Example 1: Basic sampling
    samples = uniform_sampling(1000, 0, 1)
    
    # Example 2: Custom range with seed
    samples = uniform_sampling(1000, -5, 5, seed=42)

importance_sampling
~~~~~~~~~~~~~~~~
Performs importance sampling using a proposal distribution.

Mathematical Expression:

.. math::

    \mathbb{E}[f(X)] \approx \frac{1}{n}\sum_{i=1}^n f(x_i)\frac{p(x_i)}{q(x_i)}

Parameters:
    - target_pdf (callable): Target probability density function
    - proposal_pdf (callable): Proposal probability density function
    - proposal_sampler (callable): Function to generate samples from proposal
    - n_samples (int): Number of samples

Examples:

.. code-block:: python

    from gofast.stats.probs import importance_sampling
    
    def target(x):
        return np.exp(-x**2/2) / np.sqrt(2*np.pi)
    
    def proposal(x):
        return np.exp(-abs(x)) / 2
    
    # Perform importance sampling
    samples, weights = importance_sampling(
        target_pdf=target,
        proposal_pdf=proposal,
        proposal_sampler=lambda n: np.random.laplace(0, 1, n),
        n_samples=1000
    )

markov_chain
~~~~~~~~~~~
Generates a Markov chain sequence based on transition probabilities.

Parameters:
    - transition_matrix (array-like): Matrix of transition probabilities
    - initial_state (int): Starting state
    - n_steps (int): Number of steps in chain

Examples:

.. code-block:: python

    from gofast.stats.probs import markov_chain

    # Example 1: Simple two-state Markov chain
    P = np.array([[0.7, 0.3],
                  [0.4, 0.6]])
    chain = markov_chain(P, initial_state=0, n_steps=100)

brownian_motion
~~~~~~~~~~~~~
Simulates paths of Brownian motion (Wiener process).

Mathematical Expression:

.. math::

    W(t) - W(s) \sim N(0, t-s)

Parameters:
    - n_steps (int): Number of time steps
    - T (float): Total time
    - n_paths (int): Number of paths to simulate

Examples:

.. code-block:: python

    from gofast.stats.probs import brownian_motion

    # Generate Brownian motion paths
    paths = brownian_motion(n_steps=1000, T=1.0, n_paths=5)

bayesian_update
~~~~~~~~~~~~~
Performs Bayesian probability updates given prior and likelihood.

Mathematical Expression:

.. math::

    P(A|B) = \frac{P(B|A)P(A)}{P(B)}

Parameters:
    - prior (array-like): Prior probabilities
    - likelihood (array-like): Likelihood values
    - evidence (float): Total probability of evidence

Examples:

.. code-block:: python

    from gofast.stats.probs import bayesian_update

    # Example: Update probabilities with new evidence
    prior = np.array([0.3, 0.7])
    likelihood = np.array([0.8, 0.4])
    posterior = bayesian_update(prior, likelihood)

mixture_model
~~~~~~~~~~~
Creates and samples from a mixture of probability distributions.

Parameters:
    - components (list): List of distribution objects
    - weights (array-like): Mixing weights
    - n_samples (int): Number of samples to generate

Examples:

.. code-block:: python

    from gofast.stats.probs import mixture_model
    
    # Create mixture of two normal distributions
    components = [
        {'dist': 'normal', 'params': {'mu': 0, 'sigma': 1}},
        {'dist': 'normal', 'params': {'mu': 3, 'sigma': 0.5}}
    ]
    weights = [0.7, 0.3]
    
    samples = mixture_model(components, weights, n_samples=1000)

Best Practices
-------------
1. **Numerical Stability**:
   - Use log-space calculations for small probabilities
   - Check for overflow/underflow in exponential calculations
   - Implement stable sampling methods

2. **Sampling Efficiency**:
   - Use vectorized operations where possible
   - Consider using importance sampling for rare events
   - Implement proper seeding for reproducibility

3. **Distribution Selection**:
   - Choose appropriate distributions based on data characteristics
   - Validate distribution assumptions
   - Consider mixture models for complex distributions

See Also
--------
- :mod:`gofast.stats.descriptive`: For descriptive statistics
- :mod:`gofast.stats.inferential`: For statistical inference
- :mod:`gofast.stats.utils`: For statistical utility functions

References
----------
.. [1] Ross, S. M. (2014). Introduction to Probability Models.
       Academic Press.

.. [2] Robert, C. P., & Casella, G. (2004). Monte Carlo Statistical Methods.
       Springer.

.. [3] Durrett, R. (2019). Probability: Theory and Examples.
       Cambridge University Press.