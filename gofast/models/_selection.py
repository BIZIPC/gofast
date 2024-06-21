# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" Optimization search"""

import random
import numpy as np 

from abc import abstractmethod
from sklearn.model_selection import cross_val_score
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import  clone

from ..tools.validator import _is_numeric_dtype 
from .utils import apply_param_types 
# from gofast.models.utils import params_combinations

class PSOBaseSearch(BaseSearchCV):
    """
    Abstract base class for hyperparameter optimization using Particle Swarm 
    Optimization (PSO).

    This class extends the BaseSearchCV class from scikit-learn, providing a 
    foundation for implementing PSO-based hyperparameter search strategies. 
    It initializes the search with a set of parameters specific to the 
    PSO algorithm.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    param_space : dict
        Dictionary with parameters names (`str`) as keys and lists or tuples as 
        values, specifying the hyperparameter search space.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy.

    scoring : str, callable, list/tuple, dict, or None, default=None
        Strategy to evaluate the performance of the cross-validated model on the 
        test set.

    n_particles : int, default=30
        Number of particles in the swarm. Each particle represents a potential 
        solution in the hyperparameter space.

    inertia_weight : float, default=0.9
        Inertia weight to balance the global and local exploration in the swarm.

    cognitive_coeff : float, default=2.0
        Cognitive coefficient to guide particles towards their best known 
        position.

    social_coeff : float, default=2.0
        Social coefficient to guide particles towards the swarm's best known 
        position.

    n_jobs : int, default=1
        The number of jobs to run in parallel for `cross_val_score`. -1 means 
        using all processors.

    verbose : int, default=0
        Controls the verbosity of output during the optimization process.

    pre_dispatch : str or int, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel 
        execution.

    refit : bool, default=True
        If True, refit the estimator using the best found parameters on the 
        whole dataset.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=True
        If False, the `cv_results_` attribute will not include training scores.

    Notes
    -----
    This is an abstract class and cannot be instantiated directly. Instead, it 
    should be subclassed to implement specific PSO algorithms for hyperparameter 
    optimization.

    See Also
    --------
    BaseSearchCV : The super class from which this class inherits.
    """

    @abstractmethod 
    def __init__(self, 
        estimator, 
        param_space, 
        cv=None, 
        scoring=None, 
        n_particles=30, 
        inertia_weight=0.9, 
        cognitive_coeff=2.0, 
        social_coeff=2.0,
        n_jobs=1, 
        verbose=0,
        pre_dispatch="2*n_jobs", 
        refit=True, 
        error_score=np.nan, 
        return_train_score=True, 
    ):
        super().__init__(
            estimator=estimator, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            refit=refit, 
            pre_dispatch=pre_dispatch, 
            error_score=np.nan,
            cv=cv, 
            verbose=verbose,
            return_train_score= return_train_score,
        )
        self.param_space = param_space
        self.n_particles = n_particles
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
    
    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fit the PSO-based search to the dataset.
    
        This method initializes the search process for optimal hyperparameters 
        using the Particle Swarm Optimization approach. It stores the input data, 
        then delegates to the `fit` method of the superclass 
        (`BaseSearchCV` in scikit-learn), which handles the actual fitting 
        process.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Target relative to X for classification or regression; None for
            unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of the estimator.
    
        Returns
        -------
        self : object
            The instance itself.
        """
        self.X = X.copy()
        self.y = y.copy()
        super().fit(X, y, groups=groups, **fit_params)
        return self

    def _run_search(self, evaluate_candidates):
        """
        Run the search for optimal hyperparameters.
    
        This method should be implemented in subclasses to define the specific
        steps and logic of the Particle Swarm Optimization search algorithm. The
        method is responsible for iterating over the search space, evaluating
        candidates, and updating the search state.
    
        Parameters
        ----------
        evaluate_candidates : callable
            A function that evaluates a list of candidate hyperparameter sets. This
            function is expected to be implemented by subclasses and is typically
            used to assess the performance of each candidate solution.
    
        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass, calling this method
            will raise a NotImplementedError.
    
        Notes
        -----
        Subclasses should implement this method to provide the search logic specific
        to Particle Swarm Optimization or other search strategies. The method should
        iteratively evaluate and update candidate solutions, ultimately setting the
        best solution found in the search process.
        """
        raise NotImplementedError("_run_search not implemented.")

    def _initialize_particles(self):
        """
        Initialize particles with random positions and velocities.
    
        Returns
        -------
        particles : list of dicts
            A list of particles, where each particle is represented as a
            dict containing 'position', 'velocity', 'best_position', 
            and 'best_score'.
        """
        particles = []
        for _ in range(self.n_particles):
            particle = {'position': self._random_hyperparameters(),
                        'velocity': self._random_velocity(),
                        'best_position': None,
                        'best_score': -np.inf}
            particles.append(particle)
            
        return particles

    def _random_hyperparameters(self):
        """
        Generate a set of hyperparameters by randomly selecting values from 
        the defined search space.
    
        This method iterates through each hyperparameter defined in the 
        parameter space, and selects a value based on the type of the 
        hyperparameter (categorical, continuous range,
        single-value numeric).
    
        Returns
        -------
        dict
            A dictionary of hyperparameters, where each key-value pair 
            corresponds to a hyperparameter and its randomly selected value.
    
        Raises
        ------
        ValueError
            If the hyperparameter's definition in the parameter space is 
            invalid or not supported.
        """
        hyperparameters = {}
        for param, values in self.param_space.items():
            if _is_categorical(values):
                hyperparameters[param] = _choose_categorical(values)
            elif _is_continuous_range(values):
                hyperparameters[param] = _choose_continuous(values)
            elif _is_single_numeric(values):
                hyperparameters[param] = _choose_single_numeric(values)
            else:
                raise ValueError(f"Invalid parameter space for {param}.")
    
        return hyperparameters

    def _random_velocity(self):
        """
        Randomly generate initial velocities for the particles.
    
        The velocity of a particle represents the rate of change of its position 
        (hyperparameters). These are typically small values or zeros.
    
        Returns
        -------
        velocity : dict
            A dictionary representing the velocity of a particle. Each key-value pair 
            corresponds to a parameter and its velocity.
        """
        velocity = {}
        for param in self.param_space.keys():
            # Assuming a small range for velocity 
            # initialization, e.g., [-0.1, 0.1]
            velocity[param] = random.uniform(-0.1, 0.1)
        return velocity

    def _evaluate_particle(self, particle, X, y):
        """
        Evaluate the fitness of a particle's position.
    
        Parameters
        ----------
        particle : dict
            A particle representing a set of hyperparameters.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The fitness score of the particle's position.
        """
        #estimator_name = self.estimator.__class__.__name__
        estimator = clone(self.estimator)
        
        # Apply parameter types to particle['position']
        particle['position'] = apply_param_types(estimator, particle['position'])
        
        estimator.set_params(**particle['position'])
        score = np.mean(cross_val_score(
            estimator, X, y, cv=self.cv,
            scoring=self.scoring, n_jobs=self.n_jobs)
        )
        return score
    
    def _evaluate_particle0(self, particle, X, y):
        """
        Evaluate the fitness of a particle's position.
    
        Parameters
        ----------
        particle : dict
            A particle representing a set of hyperparameters.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The fitness score of the particle's position.
        """
        estimator = clone(self.estimator).set_params(**particle['position'])
        score = np.mean(cross_val_score(
            estimator, X, y, cv=self.cv,
            scoring=self.scoring, n_jobs=self.n_jobs)
            )
        return score

    def _move_particles(self, particles, global_best):
        """
        Update the positions and velocities of particles in the swarm.
    
        This method iteratively updates each particle's velocity and position 
        based on the Particle Swarm Optimization (PSO) rules. It handles both 
        categorical and continuous hyperparameters, ensuring that the updated 
        positions respect the bounds of the hyperparameter space.
    
        Parameters
        ----------
        particles : list of dict
            The current swarm of particles. Each particle is represented as a
            dictionary containing its position, velocity, and best known position.
        global_best : dict
            The best known position among all particles in the swarm.
    
        Notes
        -----
        The method skips the velocity and position update for categorical 
        hyperparameters, as their update mechanism differs from continuous 
        hyperparameters.
        """
        for particle in particles:
            self._update_particle(particle, global_best)
    
    def _update_particle(self, particle, global_best):
        """
        Update a single particle's position and velocity.
    
        Parameters
        ----------
        particle : dict
            A single particle, containing its current position, velocity, and
            best known position.
        global_best : dict
            The best known position among all particles in the swarm.
    
        """
        for param, value in particle['position'].items():
            # Skip update for categorical hyperparameters
            if isinstance(value, str):
                continue
            # Generate random coefficients
            r1, r2 = random.random(), random.random()
    
            # Calculate the new velocity
            cognitive_velocity = self.cognitive_coeff * r1 * (
                particle['best_position'][param] - value)
            social_velocity = self.social_coeff * r2 * (global_best[param] - value)
            particle['velocity'][param] = (
                self.inertia_weight * particle['velocity'][param] + 
                cognitive_velocity + social_velocity
            )
    
            # Update the position
            new_position = value + particle['velocity'][param]
            particle['position'][param] = self._clip_position(
                param, new_position
                )
    
    def _clip_position(self, param, position):
        """
        Clip the position of a hyperparameter to its defined range.
    
        Parameters
        ----------
        param : str
            The name of the hyperparameter.
        position : float
            The proposed new position (value) of the hyperparameter.
    
        Returns
        -------
        float
            The clipped position of the hyperparameter within its defined range.
    
        """
        if isinstance(self.param_space[param], list):
            # Categorical parameters: find the nearest valid value
            return min(self.param_space[param], key=lambda x: abs(x - position))
        elif isinstance(self.param_space[param], tuple):
            # Continuous parameters: clip to the range
            return np.clip(position, self.param_space[param][0], 
                           self.param_space[param][1]
                           )
        else:
            raise ValueError(f"Invalid parameter space for {param}.")
    
class GradientBaseSearch(BaseSearchCV):
    """
    GradientBaseSearch object, an abstract base class for gradient-based 
    hyperparameter search.

    This class is designed to be subclassed to implement specific gradient-based 
    hyperparameter optimization strategies. It initializes the search process with 
    the provided estimator and additional configuration parameters.

    Parameters
    ----------
    estimator : estimator object
        The machine learning estimator to be used. This should be a 
        scikit-learn-compatible estimator that supports fitting and predicting.

    scoring : str, callable, list/tuple, or dict, default=None
        A string (see scikit-learn model evaluation documentation) or a scorer callable
        object / function with signature `scorer(estimator, X, y)`. If `None`, 
        the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs 
        for cv are:
        - An integer, specifying the number of folds in a KFold,
        - A CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

    pre_dispatch : int or str, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel execution.
        Reducing this number can be useful to avoid an explosion of memory consumption
        when more jobs get dispatched than CPUs can process. This parameter can be:
        - None, in which case all the jobs are immediately created and spawned.
        - An int, giving the exact number of total jobs that are spawned.
        - A string, giving an expression as a function of n_jobs, as in "2*n_jobs".

    refit : bool, default=True
        If True, the estimator is refitted with the best found parameters on the whole
        dataset. For multiple metric evaluation, this parameter needs to be set to a 
        string denoting the scorer that would be used to find the best parameters for 
        refitting the estimator at the end.
        
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
 
    n_jobs : int, default=1
        The number of jobs to run in parallel for both fitting and scoring. If -1, then
        the number of jobs is set to the number of CPU cores.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages. Levels of verbosity:
        - 0: no output,
        - 1 or more: the higher the number, the more detailed the messages.
    
    Notes
    -----
    This is an abstract class and cannot be instantiated directly. Subclasses should
    implement specific strategies for gradient-based optimization and must override
    this constructor to handle any additional initialization they require.
    """
    @abstractmethod
    def __init__(
        self, 
        estimator, 
        *, 
        scoring=None, 
        cv=None, 
        pre_dispatch="2*n_jobs", 
        refit=True, 
        n_jobs=1, 
        verbose=0, 
        error_score=np.nan, 
        return_train_score=True, 
        
        ):
        super().__init__(
            estimator=estimator, 
            scoring=scoring, 
            n_jobs=n_jobs, 
            cv=cv, 
            verbose=verbose,
            refit=refit, 
            pre_dispatch=pre_dispatch, 
            error_score=np.nan, 
            return_train_score=return_train_score, 
        )
        
    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fit the GradientBasedSearchCV model.
    
        This method fits the model to the data provided. It calls the 
        `fit` method of the superclass (likely a scikit-learn BaseSearchCV class) 
        to perform the actual fitting process, handling all aspects of 
        hyperparameter search and cross-validation.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data where n_samples is the number of samples and 
            n_features is the number of features. 
            This is the data on which the model will be trained.
    
        y : array-like, shape (n_samples,), optional (default=None)
            The target variable for supervised learning problems. It's an 
            array of length n_samples. 
            This is the variable that the model will be trained to predict.
    
        groups : array-like, shape (n_samples,), optional (default=None)
            Group labels for the samples used while splitting the dataset into
            train/test set. 
            Only used in conjunction with a "Group" `cv` instance (e.g., `GroupKFold`).
    
        **fit_params : dict of string -> object
            Parameters passed to the `fit` method of the estimator. For example, 
            it could be parameters like `sample_weight` or other parameters 
            related to the model fitting process.
    
        Returns
        -------
        self : object
            Returns an instance of self.
    
        Notes
        -----
        This method delegates most of the heavy lifting to the superclass 
        `fit` method, which will typically be an implementation of a 
        hyperparameter search strategy. Optionally, additional processing 
        before or after this call can be added, such as data preprocessing or 
        logging. In this specific implementation, there is a commented line
        where a copy of the training data (`X`) and target values (`y`) could 
        be stored as class attributes, which could be useful in certain 
        scenarios.
        """
        super().fit(X, y, groups=groups, **fit_params)
        return self

    def _run_search(self, evaluate_candidates):
        """
        Placeholder for the hyperparameter search algorithm.
    
        This method should be implemented in subclasses to define the actual
        hyperparameter search strategy. It is intended to encapsulate the logic 
        for exploring the hyperparameter space of the estimator.
    
        The method is expected to interact with the `evaluate_candidates` 
        function, which is responsible for evaluating a set of hyperparameters 
        and updating the search results accordingly.
    
        Parameters
        ----------
        evaluate_candidates : callable
            A function that takes a list of hyperparameter combinations in the 
            form of dictionaries, evaluates them, and records the results. 
            This function is integral to the search process, as it performs 
            the actual evaluation of parameter combinations.
    
        Raises
        ------
        NotImplementedError
            This is a placeholder method and should be overridden in subclasses. 
            If this method is called without being overridden, a 
            NotImplementedError is raised to indicate that the search strategy 
            has not been implemented.
        """
        raise NotImplementedError("_run_search not implemented.")

    def _compute_cv_score(self, X, y, params):
        """
        Compute the cross-validation score for a given set of
        hyperparameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.

        y : array-like of shape (n_samples,)
            Target values.

        params : dict
            Hyperparameters of the estimator.

        Returns
        -------
        scores : dict
            Dictionary containing the mean and standard deviation of the 
            cross-validation scores.
        """
        params = apply_param_types (self.estimator, params)
        estimator = clone(self.estimator).set_params(**params)
        scores = cross_val_score(estimator, X, y, cv=self.cv, 
                                 scoring=self.scoring, n_jobs=self.n_jobs
                                 )
        return {'mean': np.mean(scores), 'std': np.std(scores)}

    def _record_cv_results(self, params, scores):
        """
        Record the cross-validation results for a set of hyperparameters.

        Parameters
        ----------
        params : dict
            Hyperparameters of the estimator.

        scores : dict
            Dictionary containing the mean and standard deviation of the 
            cross-validation scores.
        """
        self.cv_results_['params'].append(params)
        self.cv_results_['mean_test_score'].append(scores['mean'])
        self.cv_results_['std_test_score'].append(scores['std'])
        
class AnnealingBaseSearch(BaseSearchCV):
    """
    Abstract base class for implementing simulated annealing-based 
    hyperparameter search with cross-validation.

    This class serves as a foundation for creating custom search strategies 
    using the simulated annealing technique, integrated with cross-validation. 
    It provides a structured way to explore hyperparameter spaces by 
    probabilistically accepting suboptimal solutions, especially effective in 
    avoiding local minima in complex spaces.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator that will be used for fitting data.

    scoring : str, callable, list, tuple, or dict, optional
        Strategy to evaluate the performance of the cross-validated model. 
        Can be a string, callable, list, tuple, or dict.

    cv : int, cross-validation generator or iterable, default=3
        Determines the cross-validation splitting strategy.

    init_temp : float, default=1.0
        Initial temperature for the simulated annealing process.

    alpha : float, default=0.9
        Cooling rate for the annealing process. Determines how quickly the 
        temperature decreases.

    max_iter : int, default=100
        Maximum number of iterations for the optimization process.

    random_state : int, RandomState instance or None, optional
        Controls the randomness of the estimator for reproducible results.

    n_jobs : int, default=1
        Number of jobs to run in parallel for `cross_val_score`. -1 means using 
        all processors.

    verbose : int, default=0
        Controls the verbosity of output during the optimization process.
        
    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution.
        
    refit : bool, default=True
        If True, refit the estimator using the best found parameters.

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.

    return_train_score : bool, default=True
        If True, include training scores in the `cv_results_` attribute.

    Note
    ----
    This is an abstract class and cannot be instantiated directly. Subclasses 
    should implement the necessary methods for the specific simulated annealing 
    algorithm.
    
    """
    @abstractmethod
    def __init__(
        self, 
        estimator, 
        *, 
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
            n_jobs=n_jobs, 
            refit=refit, 
            cv=cv, 
            verbose=verbose, 
            error_score=error_score, 
            pre_dispatch=pre_dispatch, 
            return_train_score=return_train_score
        )
        self.init_temp = init_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """
        Executes a search using the simulated annealing algorithm by repeatedly 
        invoking `evaluate_candidates`.
    
        In this implementation, unlike GridSearchCV or RandomizedSearchCV, the
        search strategy is based on simulated annealing. This probabilistic 
        technique is effective for finding global optima in complex search 
        spaces, as it allows for exploring suboptimal solutions in the early 
        stages (at higher temperatures) and gradually focuses on exploitation
        (at lower temperatures).
    
        The `_run_search` method in AnnealingBaseSearch orchestrates this 
        process by iteratively adjusting hyperparameters and evaluating their 
        performance. The method starts with an initial temperature and cools 
        down in each iteration, allowing uphill moves (acceptance of worse solutions) 
        with decreasing probability.
    
        Parameters
        ----------
        evaluate_candidates : callable
            A function that evaluates a given set of parameters. It takes:
                - a list of candidates, each a dict of parameter settings.
                - an optional `cv` parameter to use different dataset splits or
                  subsampling methods for evaluation.
                - an optional `more_results` dict to be added to `cv_results_`.
                  Values should be lists with a length equal to `n_candidates`.
    
            Returns a dictionary of evaluation results, formatted similarly to 
            `cv_results_`.
    
        Examples
        --------
        ::
    
            def _run_search(self, evaluate_candidates):
                'Example implementation of simulated annealing search'
                current_params = self._random_hyperparameters()
                current_score = self._evaluate_fitness(current_params)
    
                for iteration in range(self.max_iter):
                    next_params = self._random_hyperparameters()
                    next_score = self._evaluate_fitness(next_params)
                    if self._acceptance_criterion(current_score, next_score, temperature):
                        current_params, current_score = next_params, next_score
    
                    # Reduce temperature for next iteration
                    temperature *= self.alpha
    
        Note: The simulated annealing process is inherently stochastic. The 
        acceptance of suboptimal solutions decreases as the 'temperature' 
        parameter reduces over iterations, leading the search towards more 
        promising regions of the solution space. 
    
        """
        raise NotImplementedError("_run_search not implemented.")

    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Fit the AnnealingBaseSearch to the data.
    
        This method starts the simulated annealing process for hyperparameter 
        optimization of the given estimator. It adapts the traditional `fit` 
        method by incorporating the annealing algorithm to iteratively explore 
        and evaluate different sets of hyperparameters.
    
        The method delegates the evaluation of candidates (sets of hyperparameters) 
        to the `_run_search` method, which conducts the search according to the 
        simulated annealing strategy. It accepts the training data and any 
        additional fit parameters, handling them appropriately as per the 
        requirements of the underlying estimator.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (class labels in classification, real numbers in 
            regression).
    
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. This parameter is passed to the `cv` splitter.
    
        **fit_params : dict of string -> object
            Additional parameters to pass to the fit method of the estimator.
    
        Returns
        -------
        self : AnnealingSearchBase
            This method returns the instance itself, with the best found 
            parameters, score, and estimator after the annealing search process. 
    
        Notes
        -----
        The `fit` method internally uses a cross-validation approach to evaluate
        the performance of different hyperparameter sets. The selection of 
        hyperparameters in each iteration is guided by the simulated annealing 
        algorithm, which makes probabilistic decisions to accept new solutions 
        based on the current temperature.
    
        Examples
        --------
        ::
    
            from sklearn.datasets import load_iris
            from sklearn.svm import SVC
            from gofast.models._selection import AnnealingBaseSearch
            X, y = load_iris(return_X_y=True)
            param_space = {'C': [1, 10, 100], 'gamma': [0.001, 0.0001]}
    
            search = AnnealingBaseSearch(estimator=SVC(), param_space=param_space)
            search.fit(X, y)
            print("Best Parameters:", search.best_params_)
    
        After fitting, `search.best_params_` will contain the best set of 
        parameters found during the annealing process.
        """
        self.X=X.copy() ; self.y=y.copy() 
        super().fit(X, y, groups=groups, **fit_params)
        return self

    def _random_hyperparameters(self):
        """
        Randomly select hyperparameters from the param_space.
    
        Returns
        -------
        hyperparameters : dict
            A dictionary containing a set of randomly selected 
            hyperparameters.
        """
        hyperparameters = {}
        for param, values in self.param_space.items():
            if isinstance(values, list):
                # For categorical hyperparameters
                hyperparameters[param] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # For continuous hyperparameters (defined by a range)
                hyperparameters[param] = random.uniform(values[0], values[1])
            else:
                raise ValueError(f"Invalid parameter space for {param}.")
        return hyperparameters
    
    def _evaluate_fitness(self, hyperparameters):
        """
        Evaluate the fitness of the given hyperparameters.
    
        Parameters
        ----------
        hyperparameters : dict
            Hyperparameters of the estimator.
    
        X : array-like of shape (n_samples, n_features)
            Training vectors.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        score : float
            The mean cross-validation score of the estimator with the given 
            hyperparameters.
        """
        hyperparameters = apply_param_types(self.estimator, hyperparameters)
        estimator = clone(self.estimator).set_params(**hyperparameters)
        score = np.mean(cross_val_score(
            estimator, self.X, self.y, cv=self.cv, scoring=self.scoring, 
            n_jobs= self.n_jobs)
            )
        return score

    def _acceptance_criterion(self, current_score, next_score, temperature):
        """
        Determine whether to accept the new solution based on simulated annealing.
    
        Parameters
        ----------
        current_score : float
            The score of the current solution.
    
        next_score : float
            The score of the new solution.
    
        temperature : float
            The current temperature in simulated annealing.
    
        Returns
        -------
        accept : bool
            True if the new solution is accepted, False otherwise.
        """
        if next_score > current_score:
            return True
        else:
            # Probability of accepting worse solution depends on the temperature
            # and difference between scores
            accept_prob = np.exp((next_score - current_score) / temperature)
            return accept_prob > random.random()
        
class GeneticBaseSearch(BaseSearchCV):
    """
    Abstract base class for Genetic hyperparameter search with 
    cross-validation.

    This abstract base class defines the common interface and parameters for
    implementing hyperparameter optimization algorithms based on genetic
    search strategies. Genetic algorithms are inspired by natural selection
    and evolution, where a population of candidate solutions (hyperparameter
    sets) evolves over multiple generations to find the best combination of
    hyperparameters for a machine learning model.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn-compatible estimator. The object used to fit the data.

    scoring : str, callable, list/tuple, dict or None, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set. Can be a string, a callable, or None.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel for `cross_val_score`. -1 means
        using all available processors. This can speed up the fitness
        evaluation, especially for computationally intensive models.

    cv : int, cross-validation generator, or an iterable, default=None
        Determines the cross-validation splitting strategy. Could be an integer,
        a CV splitter, or an iterable.

    verbose : int, default=0
        Controls the verbosity of output during the optimization process.
        Higher values increase the verbosity. A value of 0 (default) will
        suppress output, while higher values will provide more detailed
        information about the progress of the algorithm.

    n_population : int, default=10
        The number of individuals (hyperparameter sets) in the population at
        each generation.

    n_generations : int, default=10
        The number of generations (iterations) the genetic algorithm will run.

    crossover_prob : float, default=0.5
        The probability of performing crossover (recombination) between two
        parent individuals during reproduction.

    mutation_prob : float, default=0.2
        The probability of applying mutation to an individual's hyperparameters.

    selection_method : {'tournament', 'roulette'}, default='tournament'
        The method used for parent selection. 'tournament' uses tournament
        selection, and 'roulette' uses roulette wheel selection.

    tournament_size : int, default=3
        The size of the tournament when using tournament selection.

    random_state : int, RandomState instance, or None, default=None
        Controls the randomness of the estimator.

    refit : bool, default=True
        If True, refit an estimator with the best hyperparameters found during
        the genetic search process.

    pre_dispatch : int or str, default="2*n_jobs"
        Controls the number of jobs that get dispatched during parallel
        execution. Can be an integer or one of the predefined strings like
        "all" or "2*n_jobs".

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs during hyperparameter
        search. If set to 'raise', an error will be raised. If a numeric value,
        that value will be used as the score.

    return_train_score : bool, default=True
        If True, the training scores for each set of hyperparameters will be
        returned in the `cv_results_` attribute.

    """
    @abstractmethod
    def __init__(
        self,
        estimator,
        *,
        scoring=None,
        n_jobs=None,
        cv=None,
        verbose=0,
        n_population=10, 
        n_generations=10, 
        crossover_prob=0.5, 
        mutation_prob=0.2,
        selection_method='tournament', 
        tournament_size=3, 
        random_state=None, 
        refit=True,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=True,
    ):
        super().__init__(
            estimator=estimator, 
            scoring= scoring, 
            n_jobs= n_jobs, 
            refit=refit, 
            cv= cv, 
            verbose=verbose, 
            error_score=error_score, 
            return_train_score= return_train_score, 
            pre_dispatch= pre_dispatch, 
            ) 
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.random_state=random_state 
 
    def fit(self, X, y=None, groups=None, **fit_params):
        """
        Run fit with all sets of parameters.

        This fit method should delegate the evaluation of candidates to 
        the _run_search method, which is implemented in the subclasses.
        
        Run the genetic algorithm for hyperparameter optimization.
    
        This method applies a genetic algorithm to optimize the hyperparameters
        of the given estimator. The process includes generating an initial 
        population, evaluating their fitness using cross-validation, and 
        iteratively producing
        new generations through selection, crossover, and mutation.
        
        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
            
        """
        super().fit(X, y, groups=groups, **fit_params)
        return self

    def _generate_population(self):
        """
        Generate the initial population of hyperparameter sets.

        This method creates a population of individuals, each representing a 
        set of hyperparameters. The individuals are randomly generated based 
        on the specified parameter grid.

        Returns
        -------
        population : list
            A list of individuals, where each individual is a dictionary of 
            hyperparameters.
        """
        population = []
        for _ in range(self.n_population):
            individual = {}
            for param, values in self.param_space.items():
                individual[param] = np.random.choice(values)
            population.append(individual)
        return population
    
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
    
    def _select(self, population, scores):
        """
        Select individuals for crossover based on the specified selection method.

        Parameters
        ----------
        population : list
            The current population of individuals.
        scores : list
            The scores for each individual in the population.

        Returns
        -------
        selected : list
            A list of selected individuals for crossover.
        """
        if self.selection_method == 'tournament':
            return self._tournament_selection(population, scores)
        elif self.selection_method == 'roulette':
            return self._roulette_wheel_selection(population, scores)
        else:
            raise ValueError("Unknown selection method: {}".format(self.selection_method))
            
    def _tournament_selection(self, population, scores):
        """
        Tournament selection implementation.

        Repeatedly selects `tournament_size` individuals randomly and chooses 
        the best among them until the desired number of parents is selected.

        """
        selected = []
        for _ in range(len(population)):
            contenders = np.random.choice(len(population), 
                                          self.tournament_size, replace=False)
            best = np.argmax(np.array(scores)[contenders])
            selected.append(population[contenders[best]])
        return selected

    def _roulette_wheel_selection(self, population, scores):
        """
        Roulette wheel selection implementation.

        Assigns a roulette wheel slice to each individual proportional to its
        score and selects individuals by spinning the wheel.
        """
        selected = []
        max_score = max(scores)
        # Inverse scoring for minimization
        score_sum = sum(max_score - score for score in scores)  
        wheel = [(max_score - score) / score_sum for score in scores]
        cumulative_wheel = np.cumsum(wheel)

        for _ in range(len(population)):
            r = np.random.rand()
            selected_idx = np.searchsorted(cumulative_wheel, r)
            selected.append(population[selected_idx])
        return selected
    
    def _run_search(self, evaluate_candidates):
        """Repeatedly calls `evaluate_candidates` to conduct a search.

        This method, implemented in sub-classes, makes it possible to
        customize the scheduling of evaluations: GridSearchCV and
        RandomizedSearchCV schedule evaluations for their whole parameter
        search space at once but other more sequential approaches are also
        possible: for instance is possible to iteratively schedule evaluations
        for new regions of the parameter search space based on previously
        collected evaluation results. This makes it possible to implement
        Bayesian optimization or more generally sequential model-based
        optimization by deriving from the BaseSearchCV abstract base class.
        For example, Successive Halving is implemented by calling
        `evaluate_candidates` multiples times (once per iteration of the SH
        process), each time passing a different set of candidates with `X`
        and `y` of increasing sizes.

        Parameters
        ----------
        evaluate_candidates : callable
            This callback accepts:
                - a list of candidates, where each candidate is a dict of
                  parameter settings.
                - an optional `cv` parameter which can be used to e.g.
                  evaluate candidates on different dataset splits, or
                  evaluate candidates on subsampled data (as done in the
                  SucessiveHaling estimators). By default, the original `cv`
                  parameter is used, and it is available as a private
                  `_checked_cv_orig` attribute.
                - an optional `more_results` dict. Each key will be added to
                  the `cv_results_` attribute. Values should be lists of
                  length `n_candidates`

            It returns a dict of all results so far, formatted like
            ``cv_results_``.

            Important note (relevant whether the default cv is used or not):
            in randomized splitters, and unless the random_state parameter of
            cv was set to an int, calling cv.split() multiple times will
            yield different splits. Since cv.split() is called in
            evaluate_candidates, this means that candidates will be evaluated
            on different splits each time evaluate_candidates is called. This
            might be a methodological issue depending on the search strategy
            that you're implementing. To prevent randomized splitters from
            being used, you may use _split._yields_constant_splits()

        Examples
        --------

        ::

            def _run_search(self, evaluate_candidates):
                'Try C=0.1 only if C=1 is better than C=10'
                all_results = evaluate_candidates([{'C': 1}, {'C': 10}])
                score = all_results['mean_test_score']
                if score[0] < score[1]:
                    evaluate_candidates([{'C': 0.1}])
        """
        raise NotImplementedError("_run_search not implemented.")
    
    def __getstate__(self):
       """
       Return the state of the object as a dictionary for pickling.
       """
       state = self.__dict__.copy()

       # Exclude non-picklable attributes if necessary
       # For example, if 'estimator' is a complex object, handle it accordingly
       if 'estimator' in state and hasattr(state['estimator'], '__getstate__'):
           state['estimator'] = state['estimator'].__getstate__()

       # Exclude other non-picklable attributes if there are any
       # e.g., state.pop('non_picklable_attribute', None)

       return state
   
    def __setstate__(self, state):
        """
        Restore the state of the object from the pickle.
        """
        # If 'estimator' was handled in __getstate__, restore it accordingly
        if 'estimator' in state and hasattr(state['estimator'], '__setstate__'):
            restored_estimator = state['estimator']
            self.estimator = restored_estimator.__class__()
            self.estimator.__setstate__(restored_estimator)
    
        # Restore other attributes
        self.__dict__.update(state)
         
def _is_categorical(values):
    """Check if the parameter is categorical."""
    return not _is_numeric_dtype(values, to_array=True)

def _choose_categorical(values):
    """Randomly choose a value from a categorical parameter space."""
    return random.choice(values)

def _is_continuous_range(values):
    """Check if the parameter represents a continuous range."""
    return isinstance(values, (tuple, list)) and len(values) >= 2

def _choose_continuous(values):
    """Randomly choose a value from a continuous range."""
    return random.uniform(min(values), max(values))

def _is_single_numeric(values):
    """Check if the parameter is a single-value numeric."""
    return _is_numeric_dtype(values, to_array=True) and len(values) <= 1

def _choose_single_numeric(values):
    """Return the single numeric value or randomly choose from a list of one element."""
    return values[0] if len(values) == 1 else random.choice(values)       
        

