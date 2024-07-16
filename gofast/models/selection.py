# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides advanced search algorithms for hyperparameter tuning, including 
evolutionary methods and gradient-based approaches to optimize machine 
learning model configurations."""

import warnings 
import random
import itertools 
import numpy as np 

from sklearn.base import  clone
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler

from ._selection import GeneticBaseSearch, BaseSwarmSearch  
from ._selection import GradientBaseSearch, AnnealingBaseSearch
from .utils import apply_param_types 

__all__=["SwarmSearchCV", "GradientSearchCV", "AnnealingSearchCV", 
         "GeneticSearchCV", "EvolutionarySearchCV", "SequentialSearchCV", 
         ]

class SwarmSearchCV(BaseSwarmSearch):
    """
    Particle Swarm Optimization  for hyperparameter tuning of estimators.

    SwarmSearchCV implements a Particle Swarm Optimization (PSO) algorithm to 
    find the optimal hyperparameters for a given estimator. It maintains a 
    population of particles, where each particle represents a potential solution.
    These particles  move through the hyperparameter space influenced by their 
    own best known position and the best known positions of other particles.
    PSO is a computational method that iteratively improves a set of candidate 
    solutions concerning  a measure of quality, inspired by social behaviors 
    such as bird flocking or fish schooling [1]_.
    
    See more in :ref:`User Guide`. 
    
    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameters names (`str`) as keys and lists or tuples 
        as values,specifying the hyperparameter search space.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model on 
        the test set.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    n_particles : int, default=30
        Number of particles in the swarm. Each particle represents a potential 
        solution in the hyperparameter space. A larger number of particles can 
        increase the diversity of solutions explored but also increases 
        computational complexity. The optimal number depends on the 
        dimensionality and complexity of the hyperparameter space.

    max_iter : int, default=10
        Maximum number of iterations for the optimization process. This defines 
        how many times the swarm will update the particles' positions. More 
        iterations allow more opportunities for finding better solutions but 
        increase the computational time.

    inertia_weight : float, default=0.9
        Inertia weight to balance the global and local exploration. It influences 
        the momentum of the particles, with a larger inertia weight facilitating 
        exploration (searching new areas) and a smaller one favoring exploitation 
        (fine-tuning solutions). Proper tuning of this parameter is crucial for 
        the balance between exploration and exploitation.

    cognitive_coeff : float, default=2.0
        Cognitive coefficient to guide particles towards their best known 
        position. It reflects the particle's tendency to return to its own 
        historically best-found position. A higher cognitive coefficient means 
        that particles are more influenced by their own memory, potentially 
        leading to faster convergence but increasing the  risk of getting 
        trapped in local optima.

    social_coeff : float, default=2.0
        Social coefficient to guide particles towards the swarm's best known 
        position. It represents the particle's inclination to move towards 
        the best position found by any particle in the swarm. A higher social 
        coefficient increases the mutual sharing of information among particles, 
        which can enhance the swarm's ability to converge towards promising 
        regions of the search space.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm.
    
    n_jobs : int, default=1
        The number of jobs to run in parallel for `cross_val_score`. -1 means
        using all available processors. This can speed up the fitness evaluation,
        especially for computationally intensive models.
        
    verbose : int, default=0
        Controls the verbosity of output during the optimization process.
        Higher values increase the verbosity. A value of 0 (default) will
        suppress output, while higher values will provide more detailed
        information about the progress of the algorithm, including the 
        current iteration and best score.
        
    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.


    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A record of each particle's position and score at each iteration.
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

    Methods
    -------
    fit(X, y=None, groups=None, **fit_params):
        Run fit with all sets of parameters.

    _run_search(evaluate_candidates):
        Run the optimization algorithm to find the best parameters. 
        
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from gofast.models.selection import SwarmSearchCV
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = SwarmSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)
    
    >>> param_space = {'C': (0.1, 10), 'kernel': ['linear', 'rbf']}
    >>> swarm_search = SwarmSearchCV(estimator=SVC(), param_space=param_space, cv=5)
    >>> X, y = load_iris(return_X_y=True)
    >>> swarm_search.fit(X, y)

    Notes
    -----
    Particle Swarm Optimization (PSO) is a computational method that 
    optimizes a problem by iteratively trying to improve a candidate 
    solution with regard to a given measure of quality [2]_. PSO optimizes 
    a problem by having a population of candidate solutions, here dubbed 
    particles, and moving these particles around in the search-space 
    according to simple mathematical formulae over the particle's 
    position and velocity. Each particle's movement is influenced by its 
    local best known position and is also guided toward the best known 
    positions in the search-space, which are updated as better positions 
    are found by other particles.

    The PSO algorithm is particularly useful for optimization problems 
    with a large search space and complex, multimodal objective 
    functions.

    The velocity and position update equations for a particle are given 
    by:

    .. math::
        v_{i}(t+1) = \omega v_{i}(t) + c_1 r_1 (p_{i} - x_{i}(t)) 
                     + c_2 r_2 (g - x_{i}(t))

        x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    where:
    - :math:`v_{i}(t)` is the velocity of particle `i` at time `t`.
    - :math:`x_{i}(t)` is the position of particle `i` at time `t`.
    - :math:`\omega` is the inertia weight.
    - :math:`c_1` and :math:`c_2` are the cognitive and social coefficients.
    - :math:`r_1` and :math:`r_2` are random values in [0, 1].
    - :math:`p_{i}` is the best known position of particle `i`.
    - :math:`g` is the global best known position.
    
    See Also
    --------
    sklearn.model_selection.GridSearchCV : 
        Exhaustive search over specified parameter values.
    sklearn.model_selection.RandomizedSearchCV :
        Randomized search on hyperparameters.

    References
    ----------
    .. [1] Eberhart, R., and Kennedy, J. (1995). "A new optimizer using particle 
           swarm theory". In Proceedings of the Sixth International Symposium 
           on Micro Machine and Human Science, 39-43.
    .. [2] Kennedy, J., and Eberhart, R. (1995). "Particle Swarm Optimization". 
           In Proceedings of IEEE International Conference on Neural Networks, 
           IV:1942-1948.
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        n_particles=30, 
        max_iter=10, 
        inertia_weight=0.9, 
        cognitive_coeff=2.0, 
        social_coeff=2.0,
        random_state=None,
        n_jobs=1, 
        verbose=0,
        pre_dispatch="2*n_jobs", 
        refit=True, 
        error_score=np.nan, 
        return_train_score=True, 
        ):
        super().__init__(
            estimator=estimator, 
            param_space=param_space, 
            scoring=scoring, 
            cv= cv, 
            n_particles=n_particles, 
            inertia_weight=inertia_weight, 
            cognitive_coeff=cognitive_coeff, 
            social_coeff=social_coeff, 
            pre_dispatch=pre_dispatch, 
            refit=refit, 
            error_score=error_score, 
            return_train_score=return_train_score, 
            n_jobs=n_jobs, 
            verbose=verbose, 
        )
        self.max_iter = max_iter
        self.random_state = random_state
        
    def _run_search(self, evaluate_candidates):
        """
        Execute the search for the best hyperparameters using Particle 
        Swarm Optimization.
    
        This method iterates over the swarm of particles, each representing 
        a set of hyperparameters. At each iteration, it evaluates the 
        performance of each particle, updates personal and global
        best scores and positions, and then moves the particles accordingly.
    
        Parameters
        ----------
        evaluate_candidates : callable
            A function to evaluate a list of candidates. Although PSO does 
            the evaluation internally, this is kept for compatibility with 
            sklearn's API.
    
        Notes
        -----
        The method updates the `best_score_`, `best_params_`, and 
        `best_estimator_` attributes of the class, along with the detailed 
        results in `cv_results_`.
    
        """
        # Initialize particles and global bests
        particles = self._initialize_particles()
        global_best_position = None
        global_best_score = -np.inf
        global_best_candidates = []

        for iteration in range(self.max_iter):
            for particle in particles:
                # Evaluate the current position using a separate method
                current_score = self._evaluate_particle(particle, self.X, self.y)

                # Update particle's personal best
                if particle['best_score'] < current_score:
                    particle['best_position'] = particle['position'].copy()
                    particle['best_score'] = current_score

                # Update global best
                if current_score > global_best_score:
                    global_best_position = particle['position'].copy()
                    global_best_score = current_score

                # Verbose logging for detailed analysis
                if self.verbose > 3:
                    print(f"Particle position: {particle['position']}")
                    print(f"Particle velocity: {particle['velocity']}")
                    print(f"Particle current score: {current_score}")

            # Add the best candidate of this iteration
            global_best_candidates.append(global_best_position)

            # Move particles based on updated positions
            self._move_particles(particles, global_best_position)

            # Optional: Print progress if verbose
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter},"
                      f" Best position: {global_best_position},"
                      f" Best Score: {global_best_score}")

        # Re-evaluate candidates to prevent IndexError and update test scores
        if global_best_candidates:
            evaluate_candidates(global_best_candidates)

        # Store and process the results
        self._store_search_results(particles)

        # Log completion and results if verbose
        if self.verbose:
            print("Optimization completed.")
            print(f"Best score: {global_best_score:.4f}")
            print(f"Best parameters: {global_best_position}")

    
    def _store_search_results(self, particles):
        """
        Store the results of the search in a structured format.
    
        This method prepares and sorts the cv_results_ attribute, which 
        includes the hyperparameters, mean test scores, and rankings of 
        each particle.
    
        Parameters
        ----------
        particles : list of dicts
            The list of particles representing the hyperparameter search space.
    
        """
            
        self.cv_results_ = [{'params': p['position'],
                             'mean_test_score': p['best_score'],
                             'std_test_score': 0}
                            for p in particles]
        self.cv_results_.sort(key=lambda x: x['mean_test_score'], reverse=True)
        for rank, result in enumerate(self.cv_results_, start=1):
            result['rank_test_score'] = rank
            
            
    
class GradientSearchCV(GradientBaseSearch):
    r"""
    Implements gradient-based hyperparameter optimization for estimators.

    This class is specifically designed for optimizing hyperparameters of
    estimators using a gradient-based approach. It is particularly effective
    for continuous and differentiable hyperparameters, where it employs
    gradient information to iteratively adjust the parameters, aiming to
    minimize a predefined loss function. The approach mirrors the principles
    of gradient descent, commonly used in optimizing machine learning models,
    but is tailored for hyperparameter tuning.

    The fundamental concept behind this optimization is encapsulated in the
    following mathematical formulation:

    .. math::
        \Theta_{\text{new}} = \Theta_{\text{old}} - \eta \cdot \nabla_\Theta J(\Theta)

    where:
    - \Theta represents the hyperparameters.
    - \eta is the learning rate, controlling the step size in the optimization.
    - J(\Theta) denotes the loss function, usually the negative cross-validation score.
    - \nabla_\Theta J(\Theta) is the gradient of J with respect to the 
      hyperparameters.

    Direct computation of the gradient in the hyperparameter space is often
    challenging. This implementation approximates the gradient, making the
    method most suitable for scenarios where hyperparameters are continuous
    and their gradient can be reasonably estimated.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator, which is the machine learning model
        for which hyperparameters are to be optimized.

    param_space : dict
        A dictionary mapping each hyperparameter name (as a string) to its
        search range (as a tuple). This range specifies the continuous
        interval within which the hyperparameters will be optimized.

    scoring : str, callable, list/tuple, dict, or None, default=None
        The strategy to evaluate the performance of the model on the test set.
        This can be a string denoting a pre-defined scoring method, a callable
        for defining custom scoring, or None, in which case the default scorer
        of the estimator is used.

    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy. This could be an
        integer specifying the number of folds in a KFold, a CV splitter object,
        or any iterable yielding train/test splits.

    max_iter : int, default=100
        The maximum number of iterations allowed in the optimization process.

    alpha : float, default=0.01
        The step size for updating hyperparameters. A higher value might lead
        to faster convergence but risks overshooting, while a smaller value
        ensures more precise convergence at the expense of speed.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    n_jobs : int, default=1
        The number of jobs to run in parallel for fitting and scoring.
        A value of -1 uses all available processors, which can expedite
        cross-validation and fitness evaluation for computationally intensive
        models or large datasets.

    verbose : int, default=0
        Controls the verbosity of the process. A value of 0 means no output,
        and higher values increase the detail of messages, such as progress
        of iterations and current scores.
        
    Attributes
    ----------

    convergence_history_ : list of tuples
        A record of the optimization process, where each tuple contains a set of
        hyperparameters and the corresponding score at each iteration.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.
        
    Examples
    --------
    >>> from gofast.models.selection import GradientSearchCV
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVR
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': (0.1, 10), 'gamma': (0.001, 0.1)}
    >>> search = GradientSearchCV(estimator=SVR(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    References
    ----------
    - Bengio, Y., 2000. Gradient-based optimization of hyperparameters.
    - Rasmussen, C.E., Williams, C.K.I., 2006. Gaussian Processes for Machine Learning.
    
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=None, 
        max_iter=100, 
        alpha=0.01, 
        random_state=None,
        pre_dispatch="2*n_jobs", 
        refit=True, 
        error_score=np.nan, 
        return_train_score=True, 
        n_jobs=1, 
        verbose=0     
    ):
        super().__init__( 
            estimator=estimator, 
            scoring=scoring,
            cv=cv, 
            pre_dispatch=pre_dispatch, 
            refit=refit, 
            n_jobs=n_jobs, 
            error_score=np.nan, 
            verbose=verbose,
            return_train_score=return_train_score 
        )
        self.param_space = param_space
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """
        Conducts a gradient-based hyperparameter optimization.

        This method integrates a gradient descent approach for numeric 
        parameters with an exhaustive search over categorical parameters, 
        optimizing the hyperparameter space of the given estimator.

        The method operates by iterating over every combination of categorical 
        parameters and applying a gradient-based optimization on the numeric 
        parameters for each combination.

        Parameters
        ----------
        evaluate_candidates : callable
            A function provided by the parent class to evaluate a given set of 
            hyperparameters. It updates the cv_results_ attribute with the 
            evaluation results.

        The method proceeds as follows:
        1. Splits the hyperparameter space into numeric and categorical 
           parameters.
        2. Iterates over all combinations of categorical parameters.
        3. For each combination, performs gradient-based optimization on 
           numeric parameters.
        4. Evaluates performance for each set of hyperparameters using 
           `evaluate_candidates.`
        5. Updates the best score, parameters, and estimator based on 
           evaluation results.
        """
        # Initialize the structures to record search results
        self._initialize_search_results()

        # Distinguish between numeric and categorical parameters
        numeric_params, categorical_params = self._split_params()

        # Iterate over all combinations of categorical parameters
        for cat_combination in itertools.product(*categorical_params.values()):
            cat_params = dict(zip(categorical_params, cat_combination))
            self._optimize_numeric_params(cat_params, numeric_params, evaluate_candidates)

    def _initialize_search_results(self):
        """Initializes the structures to store results of the search."""
        self.cv_results_ = {'params': [], 'mean_test_score': [], 'std_test_score': []}
        self.convergence_history_ = []
        self.best_params_ = self.best_score_ = self.best_estimator_ = None

    def _split_params(self):
        """Splits parameters into numeric and categorical."""
        is_numeric = lambda x: isinstance(x[0], (int, float))
        numeric_params = {k: v for k, v in self.param_space.items() if is_numeric(v)}
        categorical_params = {k: v for k, v in self.param_space.items() 
                              if not is_numeric(v)
                              }
        return numeric_params, categorical_params

    def _optimize_numeric_params(
            self, cat_params, numeric_params, evaluate_candidates):
        """
        Optimizes numeric parameters using gradient-based approach for a
        given set of categorical parameters.
        
        Parameters
        ----------
        cat_params : dict
            A dictionary containing the current combination of categorical 
            hyperparameters. Each key is the name of the categorical 
            hyperparameter, and the corresponding value is its current setting.
    
        numeric_params : dict
            A dictionary containing the ranges of numeric hyperparameters.
            Each key is the name of the numeric hyperparameter, and the 
            corresponding value is a list or range of possible values.
    
        evaluate_candidates : callable
            A function that evaluates a set of hyperparameters. It should 
            accept a list of dictionaries where each dictionary represents 
            a set of hyperparameters to be evaluated.
        """
        current_params = {**cat_params, **{
            param: np.mean(bounds) for param, bounds in numeric_params.items()}}
        for iteration in range(self.max_iter):
            gradients = self._calculate_gradients(
                current_params, numeric_params, evaluate_candidates
                )
            self._update_numeric_params(current_params, gradients)
            try: 
                self._evaluate_and_update_best_params(
                    current_params, evaluate_candidates, iteration
                    )
            except : 
                continue 

    def _calculate_gradients(
            self, current_params, numeric_params, evaluate_candidates
            ):
        """
        Calculates gradients for numeric parameters.
        
        Parameters
        ----------
        current_params : dict
            A dictionary representing the current combination of hyperparameters 
            (both numeric and categorical).
            The numeric parameters in this dictionary are updated through the 
            optimization process.
    
        numeric_params : dict
            A dictionary containing the ranges of numeric hyperparameters.
    
        evaluate_candidates : callable
            A function to evaluate a set of hyperparameters and update `cv_results_`.
        """
        gradients = {}
        for param, bounds in numeric_params.items():
            delta = current_params[param] * 0.01
            increased_params, decreased_params = self._create_param_variants(
                current_params, param, delta
                )
            increased_params = apply_param_types(self.estimator, increased_params)
            decreased_params = apply_param_types(self.estimator, decreased_params)
            print(increased_params)
            evaluate_candidates([increased_params])
            evaluate_candidates([decreased_params])

            if len(self.cv_results_['mean_test_score']) >= 2:
                gradients[param] = self._compute_gradient(
                    increased_params, decreased_params)

        return gradients
    
    def _create_param_variants(self, params, param_name, delta):
        """
        Creates two variants of the parameters with increased and 
        decreased values for a given parameter.
        
        Parameters
        ----------
        params : dict
            The current set of hyperparameters.
    
        param_name : str
            The name of the numeric hyperparameter for which variants are to
            be created.
    
        delta : float
            The amount by which the parameter value should be varied to
            create its variants.
        """
        increased_params = params.copy()
        decreased_params = params.copy()
        increased_params[param_name] += delta
        decreased_params[param_name] -= delta
        return increased_params, decreased_params
    
    def _compute_gradient(self, increased_params, decreased_params):
        """
        Computes the gradient based on increased and decreased 
        parameter scores.
        
        Parameters
        ----------
        increased_params : dict
            A set of hyperparameters where the value of one specific 
            parameter has been increased by a small delta.
    
        decreased_params : dict
            A set of hyperparameters where the value of the same specific
            parameter has been decreased by the same small delta.
        """
        increased_score = self.cv_results_['mean_test_score'][-2]
        decreased_score = self.cv_results_['mean_test_score'][-1]
        return (increased_score - decreased_score) / (2 * (
            increased_params - decreased_params))
    
    def _update_numeric_params(self, params, gradients):
        """
        Updates the numeric parameters based on the computed 
        gradients.
        
        Parameters
        ----------
        params : dict
            The current set of hyperparameters. This method updates the numeric
            parameters in this dictionary based on the calculated gradients.
    
        gradients : dict
            A dictionary of gradients for the numeric parameters. Each key is 
            the name of a numeric hyperparameter, and its value is the gradient 
            that has been calculated for it.
        """
        for param, gradient in gradients.items():
            params[param] += self.alpha * gradient
            
    def _evaluate_and_update_best_params(
            self, params, evaluate_candidates, iteration):
        """
        Evaluates the current parameter set and updates the best
        parameters if necessary.
        
        Parameters
        ----------
        params : dict
            The current set of hyperparameters to be evaluated.
    
        evaluate_candidates : callable
            A function to evaluate the current set of hyperparameters 
            and update `cv_results_`.
    
        iteration : int
            The current iteration number in the optimization process. 
            Used for verbose logging.
        """
        evaluate_candidates([params])
        current_score = self.cv_results_['mean_test_score'][-1]
        if self.verbose > 0:
            print(f"Iteration {iteration + 1}/{self.max_iter}, "
                  f"Current score: {current_score:.4f}")
        if current_score > self.best_score_:
            self.best_score_ = current_score
            self.best_params_ = params.copy()
            self.best_estimator_ = clone(self.estimator).set_params(**params)
            
class AnnealingSearchCV(AnnealingBaseSearch):
    r"""
    AnnealingSearchCV implements a simulated annealing algorithm for tuning 
    hyperparameters of a given estimator. 
    
    Simulated annealing is a probabilistic technique used to approximate the 
    global optimum of a given function. The approach is particularly effective 
    at avoiding local minima in complex hyperparameter spaces by allowing 
    uphill moves (acceptance of worse solutions) with a probability that 
    as the 'temperature' of the algorithm is lowered over iterations.

    The underlying principle of this algorithm is inspired by the physical process 
    of annealing in metallurgy, where materials are heated and then gradually cooled 
    to improve their properties. In the context of hyperparameter optimization, 
    simulated annealing starts at a high temperature and gradually cools down, 
    exploring different configurations of hyperparameters. At each step, a new 
    hyperparameter set is generated and assessed. If the new set is better, it 
    is accepted; if it is worse, it might still be accepted with a certain 
    probability that depends on the temperature and the extent of deterioration 
    in solution quality. This probability is given by the formula:

    .. math::
        P(\text{{accept}}) = \exp\left(-\frac{{\Delta E}}{{kT}}\right)

    where :math:`\Delta E` is the change in the objective function (score difference 
    between the current and new solution), :math:`T` is the current temperature, 
    and :math:`k` is a constant that influences the acceptance probability of 
    worse solutions.

    As the temperature decreases over iterations, the probability of accepting 
    worse solutions diminishes, thus allowing the algorithm to gradually shift 
    its focus from exploration of the search space to exploitation of the best 
    found solutions.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator used for fitting data.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_space: dict or list of dicts
        Dictionary mapping parameter names (strings) to lists of parameter 
        settings to try. Specifies the hyperparameter search space.
        If a list is given, it is sampled uniformly. 
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
        
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string ;
        - a callable that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        If None, the estimator's score method is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`Scikit-Learn User Guide <cross_validation>` for the 
        various cross-validation strategies that can be used here.

    init_temp : float, default=1.0
        Starting temperature for the annealing process. The parameter 
        represents the starting temperature for the simulated annealing 
        process. In the context of simulated annealing, the temperature is a 
        metaphorical concept that controls the likelihood of accepting worse 
        solutions during the search process. A higher initial temperature 
        allows the algorithm to explore the hyperparameter space more freely, 
        increasing the chances of accepting solutions that may not be optimal 
        but could lead to better solutions in subsequent iterations. As the 
        temperature decreases, the algorithm becomes more conservative, 
        focusing more on refining and exploiting the best solutions found so 
        far.

        Setting a higher initial temperature can be useful when the search 
        space is large or complex, as it encourages a broader exploration 
        initially. Conversely, a lower initial temperature might be beneficial
        when the search space is smaller or when prior knowledge suggests that 
        the optimal solution lies close to the initial guess.
        
    alpha : float, default=0.9
        The cooling rate, dictating how quickly the temperature decreases 
        in each iteration.
        
        It determines how quickly the temperature decreases after each iteration. 
        The value of `alpha` is typically set between 0 and 1, with values closer 
        to 1 resulting in a slower cooling process. A slower cooling 
        (i.e., a higher `alpha`) allows more thorough exploration of the search 
        space, as the algorithm accepts suboptimal solutions with higher 
        probability for a longer period. In contrast, a faster cooling 
        (lower alpha) sharpens the algorithm's focus on exploitation of the 
        best-found solutions more quickly.
        
        A careful balance is needed when setting `alpha`. Too slow a cooling 
        rate (very high alpha) might cause the algorithm to spend too much 
        time exploring and not enough time refining the best solutions, 
        potentially leading to longer runtimes without significant improvement. 
        On the other hand, too fast a cooling rate (very low alpha) might cause 
        the algorithm to converge prematurely to a local optimum.
        
    max_iter : int, default=100
        Maximum number of iterations to perform during the optimization process.

        Each iteration involves generating a new set of hyperparameters, 
        evaluating them, and then deciding whether to accept this new set 
        based on the current temperature and the acceptance criterion. 
        The `max_iter` parameter thus controls the length of the optimization 
        process. 
        The value of ``max_iter`` should be chosen based on the complexity 
        of the hyperparameter space and computational resources. A higher 
        number of iterations allows for more extensive exploration and can 
        potentially lead to better optimization results, especially in complex 
        or multi-modal search spaces. However, it also increases the 
        computational cost. A lower max_iter value will result in quicker 
        searches but might compromise the quality of the solution, especially 
        if the search space is large or complex.
 
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`Scikit-Learn User Guide <cross_validation>` for the 
        various cross-validation strategies that can be used here.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = AnnealingSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    The simulated annealing process is stochastic. The results may vary between 
    runs due to the probabilistic nature of the acceptance criterion and the 
    initial selection of hyperparameters. The algorithm is effective for exploring 
    complex search spaces and can potentially escape local optima by accepting 
    suboptimal solutions, especially at higher temperatures.
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        init_temp=1.0, 
        alpha=0.9, 
        max_iter=100, 
        random_state=None, 
        n_jobs=1, 
        verbose=0,
        pre_dispatch="2*n_jobs", 
        refit=True,
        error_score=np.nan,
        return_train_score=True
    ):
        super().__init__(
            estimator=estimator, 
            scoring=scoring, 
            cv=cv, 
            init_temp=init_temp, 
            alpha=alpha, 
            max_iter=max_iter, 
            random_state=random_state, 
            n_jobs=n_jobs, 
            verbose=verbose,
            pre_dispatch=pre_dispatch, 
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score
        )
        self.param_space=param_space 
        
    def _run_search(self, evaluate_candidates):
        """
        Executes the simulated annealing search algorithm for hyperparameter 
        optimization.
    
        This method conducts a search for the optimal hyperparameters using the 
        simulated annealing approach. It starts with a random set of hyperparameters 
        and an initial temperature. In each iteration, it explores the hyperparameter 
        space by generating a new set of parameters and evaluating their fitness. The 
        decision to move to the new set of parameters is based on the annealing 
        acceptance criterion, which allows probabilistic uphill moves at higher 
        temperatures.
    
        The temperature is gradually reduced in each iteration, thus reducing the 
        likelihood of accepting worse solutions and focusing the search more on 
        exploitation rather than exploration. This process continues until the maximum 
        number of iterations is reached.
    
        At the end of the search, the best set of parameters found during the process 
        is stored, along with its corresponding fitness score and the fitted estimator.
    
        Parameters
        ----------
        evaluate_candidates : callable
            A function to evaluate a given set of parameters. It should take a list 
            of hyperparameters and return their evaluation scores. In this method, 
            `evaluate_candidates` is used indirectly through the `_evaluate_fitness` 
            function.
    
        Attributes
        ----------
        best_params_ : dict
            The best hyperparameter setting found during the annealing process.
    
        best_score_ : float
            The highest score achieved by any hyperparameter setting during the 
            annealing process.
    
        best_estimator_ : estimator object
            The estimator fitted with the `best_params_`.
    
        Notes
        -----
        - The `_random_hyperparameters` method is used to generate new sets of 
          parameters for evaluation.
        - The `_evaluate_fitness` method computes the fitness score for a given set 
          of parameters.
        - The `_acceptance_criterion` method determines whether to accept the new 
          set of parameters based on the current temperature and score differences.
    
        The annealing process is stochastic, and the results might vary between runs 
        due to the probabilistic nature of the acceptance criterion and the initial 
        random parameter selection.
        """
        random.seed(self.random_state)
        temperature = self.init_temp

        current_params = self._random_hyperparameters()
        current_score = self._evaluate_fitness(current_params)

        candidate_params =[current_params ]
        for iteration in range(self.max_iter):
            next_params = self._random_hyperparameters()
            next_score = self._evaluate_fitness(next_params)
            # check current param whether to fit the criterion
            if self._acceptance_criterion(current_score, next_score, temperature):
                current_params, current_score = next_params, next_score
                candidate_params.append ( current_params)
     
            temperature *= self.alpha
            
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter}, "
                      f"Temp: {temperature:.4f}, Score: {next_score:.4f}")
                
        # Call evaluate_candidates with the formatted results
        out=evaluate_candidates(candidate_params)
        # Extract scores and update best parameters and estimator
        scores = out["mean_test_score"]
        
        for idx, score in enumerate(scores):
            if score > current_score:
                best_score= score
                current_params = candidate_params[idx]
                current_score = score 
       
        if self.verbose:
            print("Optimization completed.")
            best_score = getattr (self, 'best_score_', current_score)
            best_params= getattr (self, 'best_params_', current_params)
            print(f"Best score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")

class GeneticSearchCV(GeneticBaseSearch):
    r"""
    A genetic algorithm-based hyperparameter optimization for estimators.

    GeneticSearchCV implements a genetic algorithm for hyperparameter tuning of 
    any estimator compliant with the scikit-learn API. It iteratively evolves a 
    population of hyperparameter sets towards better performance, measured via 
    cross-validation on the provided data.
    
    The genetic algorithm is based on processes observed in natural evolution,
    such as inheritance, mutation, selection, and crossover. Mathematically,
    it aims to optimize a set of parameters by treating each parameter set as 
    an individual in a population. The fitness of each individual is determined 
    by its performance, measured using cross-validation.

    .. math::
        \text{Fitness(individual)} = \text{CrossValScore}(\text{estimator}, 
                                                          \text{parameters}, 
                                                          \text{data})

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator. The object to use to fit the data.

    param_space : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter 
        settings to try as values.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:
        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:
        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    
    n_population : int, default=10
        Number of individuals (parameter sets) in each generation. This 
        represents the size of the population of solutions that the algorithm 
        maintains at any given time. A larger population may increase the 
        diversity of solutions, potentially improving the algorithm's ability 
        to escape local optima, but it also increases computational complexity.
    
    n_generations : int, default=10
        Number of generations for the evolutionary algorithm. Each generation 
        involves selecting the best-performing individuals from the current 
        population, applying crossover and mutation operations to create a new 
        generation, and then evaluating this new generation. A higher number of 
        generations allows more opportunities for the algorithm to improve the 
        solutions but also requires more computational time.
    
    mutation_prob : float, default=0.1
        The probability of mutation for each parameter in an individual. Mutation 
        introduces random changes to the parameters, helping to maintain diversity 
        within the population and potentially allowing the algorithm to explore 
        new areas of the solution space. The mutation rate needs to be balanced: 
        too high a rate may turn the search into a random walk, while too low may 
        lead to premature convergence.
    
    crossover_prob : float, default=0.5
        The probability of crossover between pairs of individuals. Crossover 
        combines parts of two parent solutions to create offspring solutions. It 
        allows the algorithm to recombine good characteristics from different 
        individuals, potentially leading to better solutions. Similar to mutation 
        rate, an appropriate crossover rate is crucial for the efficiency of the 
        algorithm. Too high a rate might disrupt good solutions, while too low 
        a rate can slow down the convergence.
    
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    selection_method : {'tournament', 'roulette'}, default='tournament'
        Method used to select individuals for breeding. Supported methods are 
        'tournament' and 'roulette'.
        
        - 'tournament': A selection strategy where a number of individuals are 
          randomly chosen from the population, and the one with the best fitness 
          is selected for crossover. This process is repeated until enough 
          individuals are selected. This method introduces selection pressure, 
          favoring stronger individuals, but still provides chances for weaker 
          individuals to be chosen. The size of the tournament controls the 
          selection pressure: larger tournaments tend to favor stronger 
          individuals.

        - 'roulette': Also known as 'fitness proportionate selection', this method 
          involves selecting individuals based on their fitness proportion relative 
          to the population. Each individual gets a portion of the roulette wheel, 
          with fitter individuals receiving larger portions. A random selection is 
          then made similar to spinning a roulette wheel. This method can lead to 
          faster convergence as it strongly favors individuals with high fitness, 
          but might reduce genetic diversity in the population and can suffer from 
          the issue of 'fitness scaling'.

    tournament_size : int, default=3
        Number of individuals to be selected for each tournament. This parameter 
        is used only when selection_method is 'tournament'.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    n_jobs : int, default=1
        The number of jobs to run in parallel for both fitting and scoring. -1 means 
        using all processors.

    verbose : int, default=0
        Controls the verbosity of output during the fitting process. Higher values 
        increase the verbosity. A value of 0 (default) will suppress output, while 
        higher values will provide more detailed information about the progress of 
        the algorithm, including the current generation number and the best score 
        at each generation.
        
    Attributes
    ----------
    cv_results_ : list of dicts
        A list containing the scores and hyperparameters of each generation. 
        Each dict has the following keys:
            - 'params': list of parameter sets tested in the generation
            - 'scores': list of mean cross-validation scores for each parameter set
            - 'generation': integer representing the generation number

    best_params_ : dict
        The parameter setting that gave the best results on the hold out data.
        This attribute is updated continuously as better parameters are found.

    best_score_ : float
        The highest mean cross-validation score achieved by `best_params_`.

    best_estimator_ : estimator object
        The estimator fitted with the `best_params_`. This estimator is cloned 
        from the original estimator passed to `GeneticSearchCV` and fitted to 
        the data passed to the `fit` method.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = GeneticSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> search.best_params_

    References
    ----------
    - Goldberg, D. E. (1989). Genetic algorithms in search, optimization, and 
      machine learning.
    - Holland, J.H., 1992. Adaptation in natural and artificial systems: an 
      introductory analysis with applications to biology, control, and 
      artificial intelligence.
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        n_population=10, 
        n_generations=10, 
        crossover_prob=0.5, 
        mutation_prob=0.2,
        selection_method='tournament', 
        tournament_size=3, 
        random_state=None, 
        n_jobs=1, 
        verbose=0, 
        refit=True,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
        ):
        super().__init__(
            estimator, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            cv=cv, 
            verbose=verbose, 
            n_population=n_population, 
            n_generations=n_generations, 
            crossover_prob=crossover_prob, 
            mutation_prob= mutation_prob, 
            selection_method=selection_method, 
            tournament_size=tournament_size,
            random_state=random_state,
            refit=refit, 
            pre_dispatch=pre_dispatch, 
            error_score=error_score, 
            return_train_score= return_train_score, 
        ) 
        self.param_space = param_space 
     
    def _run_search(self, evaluate_candidates):
        """
        Run the genetic algorithm for hyperparameter optimization.

        This method overrides the _run_search method from GeneticBaseSearch. It 
        uses a genetic algorithm to evolve a population of hyperparameter sets 
        towards better performance, as measured via cross-validation.

        Parameters
        ----------
        evaluate_candidates : callable
            A function provided by BaseSearchCV for evaluating a list of 
            candidates, where each candidate is a dict of parameter settings.
        """
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        # Generate initial population
        population = self._generate_population()

        for generation in range(self.n_generations):
            if self.verbose:
                print(f"Generation {generation + 1}/{self.n_generations}:")
            # Evaluate current generation
            candidate_params = [individual for individual in population]
            out = evaluate_candidates(candidate_params)
            # Extract scores and update best parameters and estimator
            scores = out["mean_test_score"]
            params= out["params"]
            #print(len(params), len(scores))
            for score, candidate_param in zip(scores, params):
                if score >= self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = candidate_param #s[idx]
                    self.best_estimator_ = clone(self.estimator).set_params(
                        **self.best_params_)

            # Verbose output for the generation
            if self.verbose:
                print(f"    Best score in this generation: {self.best_score_:.4f}")

            # Store results for the generation
            self.cv_results_.append({
                'params': candidate_params,
                'scores': scores,
                'generation': generation
            })
            # Create next generation
            population = self._create_next_generation(population, scores)

        if self.verbose:
            print("Optimization completed.")
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best parameters: {self.best_params_}")

    def _evaluate_individual(self, individual, X, y, scoring):
        """
        Evaluate a single individual (set of hyperparameters) using cross-validation.
    
        Parameters
        ----------
        individual : dict
            Dictionary representing an individual's hyperparameters.
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.
        scoring : str or callable
            Scoring method to use on the test set.
    
        Returns
        -------
        float
            Mean cross-validation score for the individual.
        """
        estimator = clone(self.estimator).set_params(**individual)
        return np.mean(cross_val_score(estimator, X, y, cv=self.cv, scoring=scoring))

    def _crossover(self, parent1, parent2):
        """
        Apply crossover operation between two parents.

        This method combines two individuals (parents) to produce a new individual
        (offspring) by randomly swapping some of their hyperparameters.

        Parameters
        ----------
        parent1 : dict
            The first parent individual.
        parent2 : dict
            The second parent individual.

        Returns
        -------
        offspring : dict
            The resulting offspring individual.
        """
        offspring = parent1.copy()
        for param in parent1:
            if np.random.rand() < self.crossover_prob:
                offspring[param] = parent2[param]
        return offspring

    def _create_next_generation(self, current_generation, scores):
        """
        Create the next generation from the current population.
    
        This method applies selection to choose individuals from the current 
        generation, uses crossover to create offspring, and then applies mutation 
        to these offspring. The resulting generation replaces the current one.
    
        Parameters
        ----------
        current_generation : list
            The current population of individuals.
        scores : list
            The scores for each individual in the current population.
    
        Returns
        -------
        next_generation : list
            The new generation of individuals.
        """
        next_generation = []
        selected = self._select(current_generation, scores)
    
        # Pairing for crossover
        for i in range(0, len(selected), 2):
            parent1, parent2 = ( 
                selected[i], selected[np.random.randint(len(selected))]
                )
            offspring1 = self._crossover(parent1, parent2)
            offspring2 = self._crossover(parent2, parent1)
            next_generation.append(self._mutate(offspring1))
            if len(next_generation) < len(current_generation):
                next_generation.append(self._mutate(offspring2))
    
        # In case of odd number, randomly add one more to complete the population
        if len(next_generation) < len(current_generation):
            next_generation.append(selected[np.random.randint(len(selected))])
    
        return next_generation

    def _mutate(self, individual):
        """
        Apply mutation to an individual's hyperparameters.
    
        Each hyperparameter of the individual has a chance to be mutated based on 
        the mutation probability. The method randomly alters the hyperparameter, 
        selecting a new value from the parameter grid.
    
        Parameters
        ----------
        individual : dict
            An individual's hyperparameters.
    
        Returns
        -------
        mutated_individual : dict
            The mutated individual.
        """
        mutated_individual = individual.copy()
        for param, values in self.param_space.items():
            if np.random.rand() < self.mutation_prob:
                mutated_individual[param] = np.random.choice(values)
        return mutated_individual
    
class EvolutionarySearchCV(GeneticBaseSearch):
    r"""
    Evolutionary algorithm-based hyperparameter optimization for estimators.

    EvolutionarySearchCV implements an evolutionary algorithm for tuning  
    hyperparameters of a given estimator. Evolutionary algorithms mimic natural 
    evolutionary processes such as mutation, recombination, and selection to 
    iteratively improve solutions (parameter sets) to an optimization problem, 
    particularly useful for complex and non-linear objective functions.

    An evolutionary algorithm evolves a population of candidate solutions. 
    The key operations involved are:

    1. Selection: Selecting the fittest individuals from the current population 
    to create a mating pool. Selection can be based on different strategies, 
    such as tournament selection or fitness proportionate selection (roulette wheel).

    2. Crossover: Combining pairs of individuals in the mating pool to create 
    offspring. This is mathematically represented as:
    
    .. math::
        \text{{offspring}} = \text{{crossover}}(\text{{parent_1}}, \text{{parent_2}})

    where the crossover function combines parts of parent_1 and parent_2.

    3. Mutation: Introducing random variations to some of the offspring's parameters, 
    which provides genetic diversity and allows exploration of new regions in the 
    solution space. This can be represented as:

    .. math::
        \text{{mutated_offspring}} = \text{{mutate}}(\text{{offspring}})

    where the mutate function alters some parameters of the offspring randomly.

    These operations are repeated over multiple generations, with the population 
    in each generation being selected from the fittest individuals of the previous 
    generation and their offspring.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_space : dict
        Dictionary with parameter names (`str`) as keys and lists of possible 
        values as values,specifying the hyperparameter search space.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.
        
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    n_population : int, default=10
        Number of individuals (parameter sets) in each generation. This represents 
        the size of the population of solutions that the algorithm maintains at any 
        given time. A larger population may increase the diversity of solutions, 
        potentially improving the algorithm's ability to escape local optima, but 
        it also increases computational complexity.

    n_generations : int, default=10
        Number of generations for the evolutionary algorithm. Each generation 
        involves selecting the best-performing individuals from the current 
        population, applying crossover and mutation operations to create a new 
        generation, and then evaluating this new generation. A higher number of 
        generations allows more opportunities for the algorithm to improve the 
        solutions but also requires more computational time.

    mutation_prob : float, default=0.1
        The probability of mutation for each parameter in an individual. Mutation 
        introduces random changes to the parameters, helping to maintain diversity 
        within the population and potentially allowing the algorithm to explore new 
        areas of the solution space. The mutation rate needs to be balanced: too 
        high a rate may turn the search into a random walk, while too low may lead 
        to premature convergence.

    crossover_prob : float, default=0.5
        The probability of crossover between pairs of individuals. Crossover 
        combines parts of two parent solutions to create offspring solutions. It 
        allows the algorithm to recombine good characteristics from different 
        individuals, potentially leading to better solutions. Similar to mutation 
        rate, an appropriate crossover rate is crucial for the efficiency of the 
        algorithm. Too high a rate might disrupt good solutions, while too low 
        a rate can slow down the convergence.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose : int
        Controls the verbosity: the higher, the more messages.
        A higher value gives more detailed messages (e.g., progress of generations).

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm. Used for reproducible results.
        
    n_jobs : int, default=1
        The number of jobs to run in parallel for evaluating the fitness of
        individuals. -1 means using all processors.
        
    verbose : int, default=0
        Controls the verbosity of output during the optimization process.
        A higher value gives more detailed messages (e.g., progress of generations).
        
    Attributes
    ----------
    best_params_ : dict
        The parameter setting that yielded the best results on the holdout data.

    best_score_ : float
        The score of the best_params_ on the holdout data.

    best_estimator_ : estimator object
       Estimator fitted with the best_params_.

    cv_results_ : list of dicts
       A list containing the scores and hyperparameters of each generation.
       
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }
            
        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from gofast.models.selection import EvolutionarySearchCV
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = EvolutionarySearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)

    Notes
    -----
    Evolutionary algorithms are particularly suited for optimization problems 
    with complex landscapes, as they can escape local optima and handle a 
    diverse range of parameter types. These algorithms can be computationally 
    intensive but are highly parallelizable.
    """
    def __init__(
        self, 
        estimator, 
        param_space, 
        scoring=None, 
        cv=3, 
        n_population=10, 
        n_generations=10, 
        mutation_prob=0.1, 
        crossover_prob=0.5, 
        selection_method='tournament', 
        tournament_size=3,  
        random_state=None, 
        n_jobs=1, 
        verbose=0,
        refit=True,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
        ):
        super().__init__(
            estimator=estimator, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            cv=cv, 
            verbose=verbose, 
            n_population=n_population, 
            n_generations=n_generations, 
            crossover_prob=crossover_prob, 
            mutation_prob=mutation_prob, 
            selection_method=selection_method, 
            tournament_size=tournament_size,
            random_state=random_state,
            refit=refit, 
            pre_dispatch=pre_dispatch, 
            error_score=error_score, 
            return_train_score=return_train_score, 
        ) 
        self.param_space = param_space 
    
    def _run_search(self, evaluate_candidates):
        """
        Run the evolutionary algorithm for hyperparameter optimization.
        Overrides the _run_search method from GeneticBaseSearch.
        
        Run the evolutionary algorithm for hyperparameter optimization.

        This method applies an evolutionary algorithm to optimize the 
        hyperparameters of the given estimator. It involves processes like 
        selection, mutation, and crossover to evolve a population of 
        hyperparameter sets towards better performance.

        The implementation parallelizes the fitness evaluation to enhance 
        computational efficiency,especially beneficial for large populations 
        or complex models.
        """
        self.cv_results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None
        
        # Initialize population
        population = self._initialize_population()
    
        for generation in range(self.n_generations):
            if self.verbose:
                print(f"Generation {generation + 1}/{self.n_generations}")
    
            # Evaluate fitness
            candidate_params = [individual for individual in population]
            out = evaluate_candidates(candidate_params)
            # Extract scores
            scores = out["mean_test_score"]
            params= out["params"]
            # Check if scores array is empty
            if len(scores) == 0:
                if self.verbose: 
                    warnings.warn(
                        "Scores array is empty. Skipping this generation.")
                continue
    
            # Find the index of the best score
            # best_idx = np.argmax(scores)
            # if scores[best_idx] > self.best_score_:
            #     self.best_score_ = scores[best_idx]
            #     self.best_params_ = candidate_params[best_idx]
            #     self.best_estimator_ = clone(self.estimator).set_params(
            #         **self.best_params_)
            for score, candidate_param in zip(scores, params):
                if score >= self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = candidate_param #s[idx]
                    self.best_estimator_ = clone(self.estimator).set_params(
                        **self.best_params_)
            # Store results for the generation
            self.cv_results_.append({
                'params': candidate_params,
                'scores': scores,
                'generation': generation
            })
    
            # Selection, Crossover, and Mutation to create next generation
            population = self._evolve(population, scores)
    
            if self.verbose:
                print(f"Best score in this generation: {self.best_score_:.4f}")
    
        if self.verbose:
            print("Optimization completed.")
            print(f"Best score: {self.best_score_:.4f}")
            if self.best_params_ is not None:
                print(f"Best parameters: {self.best_params_}")

    def _evaluate_individual_fitness(self, individual, X, y):
        """
        Evaluate the fitness of an individual hyperparameter set.

        Parameters
        ----------
        individual : dict
            An individual's hyperparameters.

        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        fitness_score : float
            The fitness score of the individual.
        """
        estimator = clone(self.estimator).set_params(**individual)
        score = np.mean(cross_val_score(
            estimator, X, y, cv=self.cv, scoring=self.scoring, 
            n_jobs= self.n_jobs)
            )
        return score

    def _initialize_population(self):
        """
        Initialize the population for the evolutionary algorithm.

        Creates an initial population of individuals, where each individual 
        is a set of hyperparameters randomly chosen from the specified 
        parameter space.

        Returns
        -------
        population : list of dicts
            A list where each dict represents an individual's hyperparameters.
        """
        population = []
        for _ in range(self.n_population):
            individual = {}
            for param, values in self.param_space.items():
                if isinstance(values, list):
                    individual[param] = np.random.choice(values)
                    # Assuming continuous range
                elif isinstance(values, tuple) and len(values) == 2:  
                    individual[param] = np.random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Invalid parameter space for {param}.")
            population.append(individual)
        return population

    def _evaluate_fitness(self, population, X, y):
        """
        Evaluate the fitness of each individual in the population.

        The fitness of an individual is determined by the cross-validation score
        of the model trained with its hyperparameters.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals, where each individual is a 
            dict of hyperparameters.

        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        fitness_scores : list
            A list containing the fitness score of each individual in the population.
        """
        fitness_scores = []
        for individual in population:
            estimator = clone(self.estimator).set_params(**individual)
            score = np.mean(cross_val_score(estimator, X, y, cv=self.cv,
                                            scoring=self.scoring, n_jobs= self.n_jobs))
            fitness_scores.append(score)
        return fitness_scores

    def _evolve(self, population, fitness_scores):
        """
        Evolve the population using genetic operations.

        This method applies selection, crossover, and mutation to create a new
        generation of individuals.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals.

        fitness_scores : list
            The fitness scores for each individual in the population.

        Returns
        -------
        new_population : list of dicts
            The new generation of individuals.
        """
        new_population = []

        # Selection: e.g., tournament selection
        selected_individuals = self._select(population, fitness_scores)

        # Crossover and Mutation: Pair selected individuals and apply 
        # crossover and mutation
        while len(new_population) < len(population):
            parent1, parent2 = self._select_parents(selected_individuals)
            child1, child2 = self._crossover(parent1, parent2)
            # Apply mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            new_population.extend([child1, child2])

        return new_population[:len(population)]
    
    def _select_parents(self, selected_individuals):
        """
        Randomly select two parents from the selected individuals.
    
        Parameters
        ----------
        selected_individuals : list of dicts
            The individuals selected for breeding.
    
        Returns
        -------
        parent1, parent2 : tuple of dicts
            Two randomly chosen parents from the selected individuals.
        """
        parent1 = random.choice(selected_individuals)
        parent2 = random.choice(selected_individuals)
        while parent1 == parent2:
            parent2 = random.choice(selected_individuals)
        return parent1, parent2

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce offspring.
    
        Parameters
        ----------
        parent1 : dict
            The first parent individual's hyperparameters.
        parent2 : dict
            The second parent individual's hyperparameters.
    
        Returns
        -------
        offspring1, offspring2 : tuple of dicts
            Two offspring produced by crossover.
        """
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        for param in parent1:
            if random.random() < self.crossover_prob:
                offspring1[param], offspring2[param] = ( 
                    offspring2[param], offspring1[param]
                    )
        return offspring1, offspring2

    def _mutate(self, individual):
        """
        Mutate an individual's hyperparameters.
    
        Parameters
        ----------
        individual : dict
            The individual's hyperparameters.
    
        Returns
        -------
        mutated_individual : dict
            The mutated individual.
        """
        mutated_individual = individual.copy()
        for param, values in self.param_space.items():
            if random.random() < self.mutation_prob:
                if isinstance(values, list):
                    mutated_individual[param] = random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    mutated_individual[param] = random.uniform(values[0], values[1])
                else:
                    raise ValueError(f"Invalid parameter space for {param}.")
        return mutated_individual

    def _record_results(self, population, fitness_scores):
        """
        Record the results of the current generation.

        This method updates the `cv_results_` attribute with the hyperparameters
        and fitness scores of each individual in the current population.

        Parameters
        ----------
        population : list of dicts
            The current population of individuals, each represented as 
            a dict of hyperparameters.

        fitness_scores : list
            The fitness scores corresponding to each individual in the population.
        """
        for individual, score in zip(population, fitness_scores):
            self.cv_results_.append({'params': individual, 'score': score})
            # Update the best parameters and score if the current individual is better
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = individual
                self.best_estimator_ = clone(self.estimator).set_params(**individual)

class SequentialSearchCV (BaseSearchCV):
    r"""
    Sequential Model-Based Optimization (SMBO) for hyperparameter tuning 
    of estimators.
    
    SequentialSearchCV implements a Bayesian optimization strategy for tuning 
    hyperparameters. It builds a probabilistic model (Gaussian Process by default) 
    of the objective function and uses this model to select hyperparameters 
    to evaluate in the true objective function, balancing exploration and 
    exploitation.

    Sequential Model-Based Optimization (SMBO), particularly Bayesian optimization, 
    involves the iterative update of a surrogate model, typically a Gaussian Process,
    which approximates the true objective function. The surrogate model is updated 
    based on observations, and an acquisition function is used to guide the 
    search for new hyperparameters.
    
    Given a set of observations \( \{(X_i, y_i)\} \), where \( X_i \) represents
    hyperparameters and \( y_i \) their corresponding performance scores, 
    the surrogate model \( P(y|X) \) is updated to reflect this data. The 
    acquisition function \( A(X; P) \) is then optimized to select the next 
    set of hyperparameters to evaluate:
    
    \[ X_{\text{new}} = \underset{X}{\text{argmax}} \, A(X; P) \]
    
    Popular choices for the acquisition function \( A \) include Expected 
    Improvement (EI), Probability of Improvement (PI), and Upper Confidence 
    Bound (UCB). These functions aim to balance exploration
    (evaluating untested hyperparameters) and exploitation 
    (focusing on areas of the hyperparameter space known to yield 
     high performance):
    
    - Expected Improvement (EI): Maximizes the expected improvement over 
      the current best observation.
    - Probability of Improvement (PI): Maximizes the probability of improving 
      over the current best observation.
    - Upper Confidence Bound (UCB): Selects hyperparameters based on a 
    trade-off between their predicted mean performance and uncertainty.
    
    These strategies enable efficient navigation of the hyperparameter space, 
    leveraging the surrogate model to make informed decisions about which
    hyperparameters to evaluate next.
    
    Parameters
    ----------
    estimator : estimator object
        The machine learning estimator to be optimized.
    
    param_space : dict
        Dictionary with parameters names as keys and distributions or lists 
        as values, defining the search space for each hyperparameter.
    
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_parameter`);
        - a callable (see :ref:`scoring`) that returns a single value.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

        If None, the estimator's score method is used.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given the ``cv_results``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``RandomizedSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        .. versionchanged:: 0.20
            Support for callable added.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
        See :term:`Glossary <random_state>`.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.84        |...|       3       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.84, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.67, 0.70],
            'std_test_score'     : [0.01, 0.24, 0.00],
            'rank_test_score'    : [1, 3, 2],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        For multi-metric evaluation, this attribute is present only if
        ``refit`` is specified.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is not available if ``refit`` is
        ``False``. See ``refit`` parameter for more information.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `n_features_in_` when fit.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if
        `best_estimator_` is defined (see the documentation for the `refit`
        parameter for more details) and that `best_estimator_` exposes
        `feature_names_in_` when fit.

    
    Examples
    --------
    >>> from gofast.models.selection import SequentialSearchCV
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> X, y = load_iris(return_X_y=True)
    >>> param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    >>> search = SequentialSearchCV(estimator=SVC(), param_space=param_space)
    >>> search.fit(X, y)
    >>> print(search.best_params_)
    
    Notes
    -----
    SMBO is particularly useful for optimization problems with expensive evaluations,
    as it efficiently narrows down the search space using the surrogate model.
    
    References
    ----------
    - Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian Optimization 
       of Machine Learning Algorithms. Advances in Neural Information Processing Systems.
       
    """
    _required_parameters = ["estimator", "param_space"]

    def __init__(
        self,
        estimator,
        param_space,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_space = param_space
        self.n_iter = n_iter
        self.random_state = random_state
        
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates):
        """
        Perform the search over the parameter space.

        This method samples a fixed number of candidate parameter sets from the
        defined parameter space. It uses `ParameterSampler` to generate parameter
        combinations, which are then evaluated to find the best set of parameters
        for the given estimator.

        The method is specifically designed to be called internally by the `fit`
        method of the base class during the hyperparameter optimization process.

        Parameters
        ----------
        evaluate_candidates : callable
            A function provided by the base class that takes a list of candidate
            parameter settings. It evaluates each parameter setting using 
            cross-validation and records the results.

        Attributes
        ----------
        param_space : dict
            Dictionary with parameters names as keys and distributions or lists of
            parameters to try. Distributions must provide a `rvs` method for sampling
            (such as those from `scipy.stats.distributions`).
        
        n_iter : int
            Number of parameter settings that are sampled. `n_iter` trades off runtime
            vs quality of the solution.
        
        random_state : int, RandomState instance or None, default=None
            Pseudo-random number generator state used for random uniform sampling from
            lists of possible values instead of scipy.stats distributions.

        Notes
        -----
        `_run_search` does not return any value; instead, it works by side effect,
        calling the `evaluate_candidates` function and thus modifying the state of
        the search object with the results of the evaluation.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.svm import SVC
        >>> from gofast.models.selection import SequentialSearchCV
        >>> X, y = load_iris(return_X_y=True)
        >>> param_space = {'C': scipy.stats.expon(scale=100),
                           'gamma': scipy.stats.expon(scale=.1)}
        >>> search = SequentialSearchCV(estimator=SVC(), param_space=param_space, 
                                        n_iter=50)
        >>> search._run_search(search._get_param_iterator)
        """
        evaluate_candidates(
            ParameterSampler(
                self.param_space, self.n_iter, random_state=self.random_state
            )
        )
