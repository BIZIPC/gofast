# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np 

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval, StrOptions, Real, Integral

class BaseAdalineStochastic(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Adaline Stochastic Gradient Descent.

    This base class implements the core functionalities of the Adaline 
    Stochastic Gradient Descent algorithm. It serves as a foundation for 
    both regression and classification tasks by updating model weights 
    incrementally for each training instance.

    Parameters
    ----------
    eta0 : float, default=0.001
        Initial learning rate. It controls the step size in the weight 
        update. Must be between 0.0 and 1.0.

    max_iter : int, default=1000
        Maximum number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when the validation error is not 
        improving. If `True`, it will terminate training when validation 
        error does not improve for `tol` for two consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as a validation set 
        for early stopping. Only used if `early_stopping` is `True`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to `True`, reuse the solution of the previous call to 
        `fit` and add more iterations to the estimator, otherwise, just 
        erase the previous solution.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning rate schedule:
        - 'constant': The learning rate remains the same.
        - 'adaptive': The learning rate decreases by `eta0_decay` factor 
          after each epoch.

    eta0_decay : float, default=0.99
        Factor by which the learning rate `eta0` is multiplied at each 
        epoch if `learning_rate` is set to 'adaptive'.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch to prevent 
        cycles.

    random_state : int or None, default=None
        Seed used by the random number generator for shuffling and 
        initializing weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1,) or (n_features + 1, n_outputs)
        Weights after fitting.

    cost_ : list of float
        Average cost (mean squared error) per epoch.

    Notes
    -----
    Adaline (ADAptive LInear NEuron) is a single-layer artificial neural 
    network and one of the earliest forms of a neural network. The 
    algorithm is sensitive to feature scaling, so it is often beneficial 
    to standardize the features before training.

    The weight update rule in stochastic gradient descent for Adaline is 
    given by:

    .. math::
        \Delta w = \eta_0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`\Delta w` is the change in weights.
    - :math:`\eta_0` is the learning rate.
    - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
    - :math:`\phi(z^{(i)})` is the predicted value using the linear 
      activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th training 
      instance.

    This method ensures that the model is incrementally adjusted to 
    reduce the mean squared error (MSE) across training instances, 
    promoting convergence to an optimal set of weights.

    Examples
    --------
    >>> from gofast.estimators.tree import AdalineStochasticClassifier
    >>> import numpy as np
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Load the California Housing dataset
    >>> housing = fetch_california_housing()
    >>> X, y = housing.data, housing.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>     X, y, test_size=0.3, random_state=0)

    >>> # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> # Initialize and fit the regressor
    >>> ada_sgd_reg = AdalineStochasticClassifier(
    >>>     eta0=0.0001, max_iter=1000, early_stopping=True, 
    >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
    >>> ada_sgd_reg.fit(X_train_std, y_train)
    >>> y_pred = ada_sgd_reg.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    See Also
    --------
    SGDRegressor : Linear regression model fitted by SGD with a variety of 
        loss functions.
    LinearRegression : Ordinary least squares Linear Regression.
    Ridge : Ridge regression.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE 
           WESCON Convention Record, New York, 96-104.

    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    _parameter_constraints: dict ={
            "eta0": [Interval(Real, 0., 1., closed="neither")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "early_stopping": [bool],
            "validation_fraction": [Interval(Real, 0., 1., closed="neither")],
            "tol": [Interval(Real, 0, None, closed="neither")],
            "warm_start": [bool],
            "learning_rate": [StrOptions({"constant", "adaptive"})],
            "eta0_decay": [Interval(Real, 0., 1., closed="neither")],
            "shuffle": [bool],
            "random_state": [Integral, None],
            "verbose": [bool]
        }
    
    @abstractmethod
    def __init__(
        self, 
        eta0=0.001, 
        max_iter=1000, 
        early_stopping=False, 
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False, 
        learning_rate='constant', 
        eta0_decay=0.99, 
        shuffle=True,
        random_state=None, 
        verbose=False
        ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.eta0_decay = eta0_decay
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        
    def _initialize_weights(self, n_features, n_outputs=1):
        """
        Initialize the weights for the model.
    
        Parameters
        ----------
        n_features : int
            Number of features in the input data.
    
        n_outputs : int, default=1
            Number of outputs. This is typically 1 for regression and the 
            number of classes for classification.
    
        Notes
        -----
        The weights are initialized using a normal distribution with mean 0 
        and standard deviation 0.01. This initialization includes an extra 
        weight for the bias term.
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)
    
    
    def _update_weights(self, xi, target, idx):
        """
        Update the weights using a single training example.
    
        Parameters
        ----------
        xi : array-like, shape (n_features,)
            Training vector of the current example.
    
        target : float
            Target value of the current example.
    
        idx : int
            Index of the output. Used for multi-output tasks.
    
        Returns
        -------
        error : float
            The error of the prediction for the current example.
        
        Notes
        -----
        The weights are updated according to the Adaline rule:
    
        .. math::
            \Delta w = \eta_0 (y - \phi(z)) x
    
        where :math:`\eta_0` is the learning rate, :math:`y` is the target 
        value, :math:`\phi(z)` is the predicted value, and :math:`x` is the 
        input feature vector.
        """
        error = target - self.activation(xi, idx)
        self.weights_[1:] += self.eta0 * xi * error
        self.weights_[0] += self.eta0 * error
        return error
    
    
    def net_input(self, X, idx=None):
        """
        Calculate the net input.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
    
        idx : int, optional
            Index of the output. Used for multi-output tasks in classification.
    
        Returns
        -------
        net_input : array-like, shape (n_samples,)
            Net input values.
        
        Notes
        -----
        The net input is calculated as the dot product of the input features 
        and the weights, plus the bias term.
        """
        if self._is_classifier():
            return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]
        else:
            return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    
    def net_activation(self, X, idx):
        """
        Compute the linear activation.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
    
        idx : int
            Index of the output. Used for multi-output tasks.
    
        Returns
        -------
        activation : array-like, shape (n_samples,)
            Activation values.
        
        Notes
        -----
        This method computes the linear activation, which is the same as the 
        net input for Adaline.
        """
        return self.net_input(X, idx)
    
    
    @abstractmethod
    def _is_classifier(self):
        """
        Check if the estimator is a classifier.
    
        Returns
        -------
        is_classifier : bool
            `True` if the estimator is a classifier, `False` otherwise.
        
        Notes
        -----
        This method is abstract and should be implemented by subclasses to 
        indicate whether the estimator is a classifier.
        """
        pass
        
    
    
    
    