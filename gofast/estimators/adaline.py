# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `gofast.estimators.adaline` module implements various versions of the 
Adaptive Linear Neuron (Adaline) algorithm for both regression and 
classification tasks. The Adaline algorithm is a foundational model in machine 
learning that uses a linear activation function and is trained using gradient 
descent. This module includes different variants of Adaline to handle 
stochastic and batch training as well as mixed tasks that can perform both 
regression and classification. 
"""

from __future__ import annotations 
import numpy as np
from tqdm import tqdm 

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

try:from sklearn.utils import type_of_target
except: from ..tools.coreutils import type_of_target 
from ..tools.validator import check_X_y, check_array 
from ..tools.validator import check_is_fitted, parameter_validator 
from ..tools._param_validation import validate_params
from ..tools._param_validation import Interval, StrOptions, Real, Integral
from ._adaline import BaseAdalineStochastic 
from .util import activator 
  
__all__= [
        "AdalineClassifier","AdalineMixte","AdalineRegressor",
        "AdalineStochasticRegressor","AdalineStochasticClassifier",
    ]


class AdalineStochasticRegressor(BaseAdalineStochastic, RegressorMixin):
    """
    Adaline Stochastic Gradient Descent Regressor.

    This regressor implements the Adaptive Linear Neuron (Adaline) algorithm 
    using Stochastic Gradient Descent (SGD) for linear regression tasks. It is 
    particularly well-suited for large datasets due to its stochastic nature, 
    updating the model weights incrementally for each training instance.

    Parameters
    ----------
    eta0 : float, default=0.0001
        Learning rate (between 0.0 and 1.0). Controls the step size in the 
        weight update.

    max_iter : int, default=1000
        Number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and 
        add more iterations to the estimator, otherwise, just erase the previous 
        solution.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning rate schedule:
        - 'constant': The learning rate remains the same.
        - 'adaptive': The learning rate decreases by `eta0_decay` factor after 
          each epoch.

    eta0_decay : float, default=0.99
        Factor by which the learning rate `eta0` is multiplied at each epoch if 
        `learning_rate` is set to 'adaptive'.

    shuffle : bool, default=True
        Whether to shuffle training data before each epoch to prevent cycles.

    random_state : int or None, default=None
        Seed used by the random number generator for shuffling and initializing 
        weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : 1d-array
        Weights after fitting.

    cost_ : list
        Average cost (mean squared error) per epoch.

    Notes
    -----
    Adaline SGD is sensitive to feature scaling and it is often beneficial to 
    standardize the features before training. Since the algorithm uses a random 
    shuffle, setting a `random_state` ensures reproducibility.

    The weight update rule in SGD for Adaline is given by:

    .. math::
        w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` is the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
    - :math:`\phi(z^{(i)})` is the predicted value using the linear activation 
      function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.

    This method ensures that the model is incrementally adjusted to reduce the 
    MSE across training instances, promoting convergence to an optimal set of 
    weights.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineStochasticRegressor
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
    >>> ada_sgd_reg = AdalineStochasticRegressor(
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
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """
    @validate_params(
        {
            "eta0": [Interval(Real, 0, 1, closed="neither")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "early_stopping": [bool],
            "validation_fraction": [Interval(Real, 0, 1, closed="neither")],
            "tol": [Interval(Real, 0, None, closed="neither")],
            "warm_start": [bool],
            "learning_rate": [StrOptions({"constant", "adaptive"})],
            "eta0_decay": [Interval(Real, 0, 1, closed="neither")],
            "shuffle": [bool],
            "random_state": [Integral, None],
            "verbose": [bool]
        }
    )
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
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            learning_rate=learning_rate, 
            eta0_decay=eta0_decay, 
            shuffle=shuffle, 
            random_state=random_state, 
            verbose=verbose
            )
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Stochastic Gradient Descent Regressor to the 
        training data `X` and target values `y`. The model weights are updated 
        incrementally for each training instance based on the learning rate `eta0`.
    
        The weight update rule in SGD for Adaline is given by:
    
        .. math::
            w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
        - :math:`\phi(z^{(i)})` is the predicted value using the linear activation 
          function.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        The learning rate can be adjusted adaptively by setting `learning_rate` to 
        'adaptive'. In this case, the learning rate `eta0` is multiplied by `eta0_decay`
        after each epoch.
    
        Examples
        --------
        >>> from gofast.estimators._base import AdalineStochasticRegressor
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
        >>> ada_sgd_reg = AdalineStochasticRegressor(
        >>>     eta0=0.0001, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_sgd_reg.fit(X_train_std, y_train)
        >>> y_pred = ada_sgd_reg.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : Linear regression model fitted by SGD 
            with a variety of loss functions.
        sklearn.linear_model.LinearRegression : Ordinary least squares Linear 
            Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        """
        X, y = check_X_y(X, y, estimator=self)
        
        self.learning_rate = parameter_validator(
            "learning_rate", target_strs={"adaptive", "constant"})(
                self.learning_rate)
                
        if not self.warm_start or not hasattr(self, 'weights_'):
            self._initialize_weights(X.shape[1])
       
        self.cost_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state)
        
        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                error = target - self.predict(xi.reshape(1, -1))
                self.weights_[1:] += self.eta0 * xi * error
                self.weights_[0] += self.eta0 * error
                cost.append(error ** 2 / 2.0)
            self.cost_.append(np.mean(cost))
            
            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = mean_squared_error(y_val, y_val_pred)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.learning_rate == 'adaptive':
                self.eta0 *= self.eta0_decay
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
        
        return self

    def predict(self, X):
        """
        Predict continuous output.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features. Can be a dense array
            or sparse matrix.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted continuous values.
    
        Notes
        -----
        This method predicts the continuous output values for the input data `X` 
        using the weights learned during training. The prediction is based on 
        calculating the net input:
    
        .. math::
            z = X \cdot w + b
    
        where:
        - :math:`X` is the input feature matrix.
        - :math:`w` is the weight vector.
        - :math:`b` is the bias term.
    
        The predicted value :math:`z` is the linear combination of the input 
        features and the learned weights.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineStochasticRegressor
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
        >>> ada_sgd_reg = AdalineStochasticRegressor(
        >>>     eta0=0.0001, max_iter=1000)
        >>> ada_sgd_reg.fit(X_train_std, y_train)
    
        >>> # Predict using the trained model
        >>> y_pred = ada_sgd_reg.predict(X_test_std)
        >>> print('Predicted values:', y_pred)
    
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        return self.net_input(X)
    
    def _is_classifier (self): 
        """ Flag to indicate that regressor is not a classifier."""
        return False  

class AdalineStochasticClassifier(BaseAdalineStochastic, ClassifierMixin):
    """
    Adaptive Linear Neuron Classifier with Stochastic Gradient Descent.

    This classifier implements a stochastic gradient descent algorithm for 
    adaptive linear neurons. Stochastic Gradient Descent (SGD) is an efficient 
    approach to discriminative learning of linear classifiers under convex loss 
    functions such as (linear) Support Vector Machines and Logistic Regression. 
    SGD has been successfully applied to large-scale and sparse machine learning 
    problems often encountered in text classification and natural language 
    processing.

    The principle behind SGD is to update the model parameters (weights) 
    incrementally for each training example. In the context of this classifier, 
    the weight update is performed as follows:

    .. math::
        \Delta w = \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here, :math:`\Delta w` represents the change in weights, :math:`y^{(i)}` is 
    the true label, :math:`\phi(z^{(i)})` is the predicted label, and :math:`x^{(i)}` 
    is the input feature vector.

    The weights are updated incrementally for each training example:

    .. math::
        w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where :math:`\eta0` is the learning rate. This incremental update approach 
    helps in adapting the classifier more robustly to large and varying datasets, 
    ensuring that each training instance directly influences the model's learning.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while 
        moving toward a minimum of a loss function. The value must be between 
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training data (aka epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and 
        add more iterations to the estimator, otherwise, just erase the previous 
        solution.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning rate schedule:
        - 'constant': The learning rate remains the same.
        - 'adaptive': The learning rate decreases by `eta0_decay` factor after 
          each epoch.

    eta0_decay : float, default=0.99
        Factor by which the learning rate `eta0` is multiplied at each epoch if 
        `learning_rate` is set to 'adaptive'.
        
    activation : str or callable, default='sigmoid'
        The activation function to apply. Supported activation functions are:
        'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 'tanh', 'softmax'.
        If a callable is provided, it should take `z` as input and return the
        transformed output.
        
        - Sigmoid: :math:`\sigma(z) = \frac{1}{1 + \exp(-z)}`
        - ReLU: :math:`\text{ReLU}(z) = \max(0, z)`
        - Leaky ReLU: :math:`\text{Leaky ReLU}(z) = \max(0.01z, z)`
        - Identity: :math:`\text{Identity}(z) = z`
        - ELU: :math:`\text{ELU}(z) = \begin{cases}
                      z & \text{if } z > 0 \\
                      \alpha (\exp(z) - 1) & \text{if } z \leq 0
                    \end{cases}`
        - Tanh: :math:`\tanh(z) = \frac{\exp(z) - \exp(-z)}{\exp(z) + \exp(-z)}`
        - Softmax: :math:`\text{Softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}`
        
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps 
        in preventing cycles and ensures that individual samples are encountered 
        in different orders.

    random_state : int, default=None
        The seed of the pseudo random number generator to use when shuffling the 
        data and initializing the weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    cost_ : list
        The sum of squared errors (cost) accumulated over the training epochs. 
        This can be used to evaluate how the model's performance has improved 
        over time.

    Notes
    -----
    Stochastic Gradient Descent is sensitive to feature scaling, so it is 
    highly recommended to scale your data. For example, use 
    `sklearn.preprocessing.StandardScaler` for standardization.

    The weight update rule in SGD for Adaline is given by:

    .. math::
        w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` is the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
    - :math:`\phi(z^{(i)})` is the predicted value using the linear activation 
      function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.

    This method ensures that the model is incrementally adjusted to reduce the 
    cost across training instances, promoting convergence to an optimal set of 
    weights.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineStochasticClassifier
    >>> import numpy as np
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Load the breast cancer dataset
    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>     X, y, test_size=0.3, random_state=0)

    >>> # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> # Initialize and fit the classifier
    >>> ada_sgd_clf = AdalineStochasticClassifier(
    >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
    >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
    >>> ada_sgd_clf.fit(X_train_std, y_train)

    >>> # Predict class labels
    >>> y_pred = ada_sgd_clf.predict(X_test_std)
    >>> print('Accuracy:', np.mean(y_pred == y_test))

    >>> # Predict class probabilities
    >>> y_proba = ada_sgd_clf.predict_proba(X_test_std)
    >>> print('Class probabilities:', y_proba)

    See Also
    --------
    AdalineGradientDescent : Gradient Descent variant of Adaline.
    SGDClassifier : Scikit-learn's SGD classifier.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
           Convention Record, New York, 96-104.
    """

    @validate_params(
        {
            "eta0": [Interval(Real, 0, 1, closed="neither")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "early_stopping": [bool],
            "validation_fraction": [Interval(Real, 0, 1, closed="neither")],
            "tol": [Interval(Real, 0, None, closed="neither")],
            "warm_start": [bool],
            "learning_rate": [StrOptions({"constant", "adaptive"})],
            "eta0_decay": [Interval(Real, 0, 1, closed="neither")],
            "activation": [ StrOptions(
                {'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 'tanh', 'softmax'}),
                callable], 
            "shuffle": [bool],
            "random_state": [Integral, None],
            "verbose": [bool]
        }
    )
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
            activation="sigmoid", 
            shuffle=True, 
            random_state=None, 
            verbose=False
            ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            learning_rate=learning_rate, 
            eta0_decay=eta0_decay, 
            shuffle=shuffle, 
            random_state=random_state, 
            verbose=verbose
            )
        self.activation=activation 
        

    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Stochastic Gradient Descent Classifier to the 
        training data `X` and target values `y`. The model weights are updated 
        incrementally for each training instance based on the learning rate `eta0`.
    
        The weight update rule in SGD for Adaline is given by:
    
        .. math::
            w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the true label for the \(i\)-th training instance.
        - :math:`\phi(z^{(i)})` is the predicted label using the linear activation 
          function.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        The learning rate can be adjusted adaptively by setting `learning_rate` to 
        'adaptive'. In this case, the learning rate `eta0` is multiplied by `eta0_decay`
        after each epoch.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineStochasticClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the breast cancer dataset
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_sgd_clf = AdalineStochasticClassifier(
        >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_sgd_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_sgd_clf.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        sklearn.linear_model.SGDClassifier : Linear classifier trained with SGD.
        sklearn.linear_model.Perceptron : Perceptron classifier.
        sklearn.linear_model.LogisticRegression : Logistic regression classifier.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        X, y = check_X_y(X, y, estimator=self, ensure_2d=True, multi_output=True)
        self.label_binarizer_ = LabelBinarizer()
        y = self.label_binarizer_.fit_transform(y)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)
        
        self.learning_rate = parameter_validator(
            "learning_rate", target_strs={"adaptive", "constant"})(
                self.learning_rate)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1] + 1, y.shape[1]))
        
        self.cost_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state)
         
        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                for idx in range(self.weights_.shape[1]):
                    error = target[idx] - self.activation(xi, idx)
                    self.weights_[1:, idx] += self.eta0 * xi * error
                    self.weights_[0, idx] += self.eta0 * error
                    cost.append(error ** 2 / 2.0) 
            self.cost_.append(np.mean(cost))
            if self.early_stopping:
                y_val_pred = self.predict(X_val).reshape (-1, 1)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.n = self.max_iter  # Force the progress bar to complete
                        progress_bar.last_print_n = self.max_iter
                        progress_bar.update(0)  # Refresh the progress bar display
                    break

            if self.learning_rate == 'adaptive':
                self.eta0 *= self.eta0_decay
            
            if self.verbose:
                progress_bar.update(1)
                
        if self.verbose:
            progress_bar.close()

        return self
    
    def predict(self, X):
        """
        Return class label after unit step.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and 
            `n_features` is the number of features. Can be a dense array 
            or sparse matrix.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class labels.
    
        Notes
        -----
        This method predicts the class labels for the input data `X` by 
        calculating the net input and applying the unit step function. The 
        unit step function maps the net input to class labels:
    
        .. math::
            y^{\text{pred}} = 
            \begin{cases}
            1 & \text{if } \phi(z) \geq 0 \\
            -1 & \text{if } \phi(z) < 0
            \end{cases}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineStochasticClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the breast cancer dataset
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_sgd_clf = AdalineStochasticClassifier(
        >>>     eta0=0.01, max_iter=1000)
        >>> ada_sgd_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_sgd_clf.predict(X_test_std)
        >>> print('Predicted class labels:', y_pred)
    
        """

        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
    
        if len(self.label_binarizer_.classes_) == 2:
            return self.label_binarizer_.inverse_transform(
                np.where(self.activation(X, 0) >= 0.0, 1, 0))
        else:
            # Calculate the activations for all classes
            activations = np.column_stack(
                [self.activation(X, idx) for idx in range(self.weights_.shape[1])])
    
            # Find the indices of the maximum values along each row
            indices = np.argmax(activations, axis=1)
            
            # Create a binary array with ones at the positions of the maximum values
            binary_arr = np.zeros_like(activations, dtype=int)
            binary_arr[np.arange(activations.shape[0]), indices] = 1
            
            # Return the inverse transform of the binary array
            return self.label_binarizer_.inverse_transform(binary_arr)
        
    
    def predict_proba(self, X):
        """
        Probability estimates.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and 
            `n_features` is the number of features. Can be a dense array 
            or sparse matrix.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples. The order of the 
            classes corresponds to that in `self.classes_`.
    
        Notes
        -----
        This method predicts the probability of each class for the input data `X` 
        using the sigmoid function. The sigmoid function maps the net input to 
        probabilities:
    
        .. math::
            P(y=1|x) = \frac{1}{1 + \exp(-\phi(z))}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        The method returns a 2D array where each row corresponds to a sample and 
        each column corresponds to a class probability. The first column contains 
        the probabilities of the negative class and the second column contains the 
        probabilities of the positive class.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineStochasticClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the breast cancer dataset
        >>> data = load_breast_cancer()
        >>> X, y = data.data, data.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_sgd_clf = AdalineStochasticClassifier(
        >>>     eta0=0.01, max_iter=1000)
        >>> ada_sgd_clf.fit(X_train_std, y_train)
    
        >>> # Predict class probabilities
        >>> y_proba = ada_sgd_clf.predict_proba(X_test_std)
        >>> print('Class probabilities:', y_proba)

        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        if len(self.label_binarizer_.classes_) == 2:
            proba = self._activator(self.net_activation(X, 0))
            return np.vstack([1 - proba, proba]).T
        else:
            proba = [self._activator(self.net_activation(X, idx)) for idx in range(
                self.weights_.shape[1])]
            return np.column_stack(proba)
    
    def _activator (self, z): 
        """Compute the activation function, defayult is sigmoid."""
        return activator (z, self.activation  )

    def _is_classifier (self):
        "Flag to indicate the type of problem."
        return True 
    
class AdalineRegressor(BaseEstimator, RegressorMixin):
    r"""
    Adaline Gradient Descent Regressor.

    `AdalineRegressor` is based on the principles of Adaptive Linear Neurons (Adaline),
    employing the batch gradient descent optimization algorithm for regression tasks.
    The AdalineRegressor fits a linear model to the data by minimizing the sum
    of squared errors between the observed targets in the dataset and the targets
    predicted by the linear approximation.

    The weight update in the batch gradient descent step is performed using 
    the following rule:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` represents the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the actual observed value for the \(i\)-th sample.
    - :math:`\phi(z^{(i)})` is the predicted value obtained from the linear model.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    This weight update mechanism allows the model to iteratively adjust the weights
    to minimize the overall prediction error, thereby refining the model's accuracy
    over training iterations.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and 
        add more iterations to the estimator, otherwise, just erase the previous 
        solution.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps 
        in preventing cycles and ensures that individual samples are encountered 
        in different orders.

    random_state : int, default=None
        The seed of the pseudo random number generator for shuffling the data 
        and initializing the weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The sum of squared errors (residuals) after each epoch. It can be used to
        analyze the convergence behavior of the algorithm during training.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.adaline import AdalineRegressor
    >>> diabetes = load_diabetes()
    >>> X = diabetes.data
    >>> y = diabetes.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineRegressor(eta0=0.01, max_iter=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    Notes
    -----
    This implementation is suitable for learning linear relationships but might
    not be optimal for non-linear datasets. Feature scaling (e.g., using
    `StandardScaler` from sklearn.preprocessing) is recommended with gradient 
    descent algorithms.

    See Also
    --------
    LinearRegression : Ordinary least squares Linear Regression from Scikit-Learn.
    SGDRegressor : Linear model fitted by minimizing a regularized empirical loss 
                   with SGD from Scikit-Learn.
    """
    def __init__(
        self, 
        eta0=0.01, 
        max_iter=1000,
        early_stopping=False, 
        validation_fraction=0.1,
        tol=1e-4, 
        warm_start=False, 
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
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Gradient Descent Regressor to the training 
        data `X` and target values `y`. The model weights are updated incrementally 
        for each training instance based on the learning rate `eta0`.
    
        The weight update rule in batch gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the actual observed value for the \(i\)-th sample.
        - :math:`\phi(z^{(i)})` is the predicted value obtained from the linear model.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the diabetes dataset
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_reg = AdalineRegressor(
        >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_reg.fit(X_train_std, y_train)
    
        >>> # Predict continuous values
        >>> y_pred = ada_reg.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        sklearn.linear_model.LinearRegression : 
            Ordinary least squares Linear Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        X, y = check_X_y(X, y, estimator=self)
        rgen = np.random.RandomState(self.random_state)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        self.errors_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state)

        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            
            net_input = self.net_input(X)
            errors = y - net_input
            self.weights_[1:] += self.eta0 * X.T.dot(errors)
            self.weights_[0] += self.eta0 * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.errors_.append(cost)

            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'\nEarly stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    def predict(self, X):
        """
        Return continuous output.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features. Can be a dense array 
            or sparse matrix.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted continuous values.
    
        Notes
        -----
        This method predicts the continuous output values for the input data `X` 
        using the weights learned during training. The prediction is based on 
        calculating the net input:
    
        .. math::
            z = X \cdot w + b
    
        where:
        - :math:`X` is the input feature matrix.
        - :math:`w` is the weight vector.
        - :math:`b` is the bias term.
    
        The predicted value :math:`z` is the linear combination of the input 
        features and the learned weights.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the diabetes dataset
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_reg = AdalineRegressor(eta0=0.01, max_iter=1000)
        >>> ada_reg.fit(X_train_std, y_train)
    
        >>> # Predict continuous values
        >>> y_pred = ada_reg.predict(X_test_std)
        >>> print('Predicted values:', y_pred)
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        sklearn.linear_model.LinearRegression : 
            Ordinary least squares Linear Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        return self.net_input(X)
     
class AdalineClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaline Gradient Descent Classifier.

    This classifier is based on the principles of Adaptive Linear Neurons 
    (Adaline), employing the gradient descent optimization algorithm for 
    binary classification tasks. The AdalineClassifier fits a linear decision 
    boundary to the data by minimizing the sum of squared errors between 
    the observed labels in the dataset and the labels predicted by the 
    linear approximation.

    The weight update in the gradient descent step is performed using the 
    following rule:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` represents the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the actual label for the \(i\)-th sample.
    - :math:`\phi(z^{(i)})` is the predicted label, obtained from the
      linear model activation.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    This update mechanism incrementally adjusts the weights to minimize 
    the prediction error, refining the model's ability to classify new 
    samples accurately.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and 
        add more iterations to the estimator, otherwise, just erase the previous 
        solution.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps 
        in preventing cycles and ensures that individual samples are encountered 
        in different orders.

    random_state : int, default=None
        The seed of the pseudo random number generator for shuffling the data 
        and initializing the weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The number of misclassifications in each epoch. This can be used to 
        analyze the convergence behavior of the algorithm during training.

    Notes
    -----
    This implementation is intended for binary classification tasks. For 
    multi-class classification, one could use strategies such as One-vs-Rest 
    (OvR). Feature scaling (e.g., using `StandardScaler` from sklearn.preprocessing) 
    is recommended with gradient descent algorithms.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineClassifier
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Load the iris dataset
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]  # we only take the first two features.
    >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>     X, y, test_size=0.3, random_state=0)

    >>> # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> # Initialize and fit the classifier
    >>> ada_clf = AdalineClassifier(
    >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
    >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
    >>> ada_clf.fit(X_train_std, y_train)

    >>> # Predict class labels
    >>> y_pred = ada_clf.predict(X_test_std)
    >>> print('Accuracy:', np.mean(y_pred == y_test))

    >>> # Predict class probabilities
    >>> y_proba = ada_clf.predict_proba(X_test_std)
    >>> print('Class probabilities:', y_proba)

    See Also
    --------
    LogisticRegression : Logistic Regression classifier from Scikit-Learn.
    SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                    loss with SGD from Scikit-Learn.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
           Convention Record, New York, 96-104.
    """

    def __init__(self, eta0=0.01, max_iter=1000, early_stopping=False,
                 validation_fraction=0.1, tol=1e-4, warm_start=False,
                 shuffle=True, random_state=None, verbose=False):
        
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Gradient Descent Classifier to the training 
        data `X` and target values `y`. The model weights are updated incrementally 
        for each training instance based on the learning rate `eta0`.
    
        The weight update rule in gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the actual label for the \(i\)-th sample.
        - :math:`\phi(z^{(i)})` is the predicted label, obtained from the linear 
          model activation.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(
        >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_clf.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        X, y = check_X_y(X, y, estimator=self, ensure_2d=True, multi_output=True)
        self.label_binarizer_ = LabelBinarizer()
        y = self.label_binarizer_.fit_transform(y)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=(X.shape[1] + 1, y.shape[1]))
        
        self.errors_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state)

        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                for idx in range(self.weights_.shape[1]):
                    error = target[idx] - self.activation(xi, idx)
                    self.weights_[1:, idx] += self.eta0 * xi * error
                    self.weights_[0, idx] += self.eta0 * error
                    cost.append(error ** 2 / 2.0)
            self.errors_.append(np.mean(cost))
            
            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self
    
    def net_input(self, X, idx):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]
    
    def activation(self, X, idx):
        """ Compute the activation"""
        return self.net_input(X, idx)

    def predict(self, X):
        """
        Predict class label after unit step.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels for the input samples.
    
        Notes
        -----
        This method predicts the class labels for the input data `X` by 
        calculating the net input and applying the unit step function. The 
        unit step function maps the net input to class labels:
    
        .. math::
            y^{\text{pred}} = 
            \begin{cases}
            1 & \text{if } \phi(z) \geq 0 \\
            -1 & \text{if } \phi(z) < 0
            \end{cases}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(eta0=0.01, max_iter=1000)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_clf.predict(X_test_std)
        >>> print('Predicted class labels:', y_pred)
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)

        if len(self.label_binarizer_.classes_) == 2:
            return self.label_binarizer_.inverse_transform(
                np.where(self.activation(X, 0) >= 0.0, 1, 0))
        else:
            activations = np.column_stack(
                [self.activation(X, idx) for idx in range(self.weights_.shape[1])])
            indices = np.argmax(activations, axis=1)
            binary_arr = np.zeros_like(activations, dtype=int)
            binary_arr[np.arange(activations.shape[0]), indices] = 1
            return self.label_binarizer_.inverse_transform(binary_arr)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
    
        Notes
        -----
        This method predicts the probability of each class for the input data `X` 
        using the logistic sigmoid function. The sigmoid function maps the net 
        input to probabilities:
    
        .. math::
            P(y=1|x) = \frac{1}{1 + \exp(-\phi(z))}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        The method returns a 2D array where each row corresponds to a sample and 
        each column corresponds to a class probability. The first column contains 
        the probabilities of the negative class and the second column contains the 
        probabilities of the positive class.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(eta0=0.01, max_iter=1000)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class probabilities
        >>> y_proba = ada_clf.predict_proba(X_test_std)
        >>> print('Class probabilities:', y_proba)
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
 
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        # Apply the linear model
        if len(self.label_binarizer_.classes_) == 2:
            proba = self._sigmoid(self.activation(X, 0))
            return np.vstack([1 - proba, proba]).T
        else:
            proba = [self._sigmoid(self.activation(X, idx)) for idx in range(
                self.weights_.shape[1])]
            return np.column_stack(proba)

    def _sigmoid(self, z):
        # Use the logistic sigmoid function to estimate probabilities
        return 1 / (1 + np.exp(-z))
   
class AdalineMixte(BaseEstimator, ClassifierMixin, RegressorMixin):
    r"""
    Adaline Mixte for Dual Regression and Classification Tasks.

    The ADAptive LInear NEuron (Adaline) Mixte model is capable of performing
    both regression and classification tasks. This model extends the classical
    Adaline by incorporating both tasks, allowing it to handle various types
    of targets including binary, multiclass, and continuous values.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and
        add more iterations to the estimator, otherwise, just erase the previous
        solution.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps
        in preventing cycles and ensures that individual samples are encountered
        in different orders.

    random_state : int, default=None
        The seed of the pseudo random number generator for shuffling the data
        and initializing the weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : array-like of shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The number of misclassifications (for classification tasks) or the
        sum of squared errors (for regression tasks) in each epoch. This can be
        used to analyze the convergence behavior of the algorithm during training.

    Notes
    -----
    This implementation is suitable for both regression and classification
    tasks. For multi-class classification, the model uses a one-vs-rest strategy.
    Feature scaling (e.g., using `StandardScaler` from `sklearn.preprocessing`)
    is recommended with gradient descent algorithms.
    
    Adaline is critical in demonstrating the importance of defining and 
    minimizing a continuous cost function, paving the way for the development 
    of advanced machine learning algorithms such as Logistic Regression, 
    Support Vector Machines, and various regression models.

    The Widrow-Hoff learning rule, or the Adaline rule, differs from Rosenblatt's 
    perceptron by updating weights based on a linear activation function rather 
    than a binary step function. This approach allows for the use of gradient 
    descent  optimization methods due to the differentiability of the cost 
    function.

    Key Contributions and Innovations:
    - Adaline's primary innovation is its cost function, which is continuous 
      and differentiable. This allows for the application of gradient descent,
      leading to more efficient and robust optimization.
    - The introduction of the Widrow-Hoff learning rule with Adaline minimizes
      the mean squared error between actual outputs and predictions, fostering
      a stable and convergent learning process.


    The weight update rule in the gradient descent step for Adaline is:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}

    where:
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value or label for the \(i\)-th sample.
    - :math:`\phi(w^T x^{(i)})` is the predicted value, where :math:`\phi`
      is the identity function (i.e., :math:`\phi(z) = z`), reflecting the
      continuous nature of the linear activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineMixte
    >>> from sklearn.datasets import load_boston, load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Example for regression
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)
    >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
    >>> model.fit(X_train_std, y_train)
    >>> y_pred = model.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    >>> # Example for classification
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)
    >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
    >>> model.fit(X_train_std, y_train)
    >>> y_pred = model.predict(X_test_std)
    >>> print('Accuracy:', np.mean(y_pred == y_test))

    See Also
    --------
    LogisticRegression : Logistic Regression classifier for binary
        classification tasks.
    LinearRegression : Ordinary least squares Linear Regression.
    SGDClassifier : Linear models (SVM, logistic regression, etc.) fitted
        by SGD.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
           "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
           October 1960.
    """

    def __init__(self, eta0=0.01, max_iter=1000, early_stopping=False,
                 validation_fraction=0.1, tol=1e-4, warm_start=False,
                 shuffle=True, random_state=None, verbose=False):
        
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values. For classification tasks, these should be discrete class
            labels. For regression tasks, these should be continuous values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Mixte model to the training data `X` and
        target values `y`. The model can handle both regression and classification
        tasks, and the type of task is determined based on the target values.
    
        For classification tasks, the target values are binarized. The weight update
        rule in gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}
    
        where:
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the true value or label for the \(i\)-th sample.
        - :math:`\phi(w^T x^{(i)})` is the predicted value, where :math:`\phi`
          is the identity function (i.e., :math:`\phi(z) = z`), reflecting the
          continuous nature of the linear activation function.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside
        as a validation set. Training stops if the validation error does not improve
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> from sklearn.datasets import load_boston, load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Example for regression
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        >>> # Example for classification
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier for binary
            classification tasks.
        LinearRegression : Ordinary least squares Linear Regression.
        SGDClassifier : Linear models (SVM, logistic regression, etc.) fitted
            by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """

        X, y = check_X_y(X, y, estimator=self, ensure_2d=True, multi_output=True)
        self.task_type = type_of_target(y)
        
        if self.task_type not in [
                "binary", "multiclass", "continuous", "multilabel-indicator"]:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        if self.task_type in ["binary", "multiclass"]:
            self.label_binarizer_ = LabelBinarizer()
            y = self.label_binarizer_.fit_transform(y)
            if y.ndim == 1:
                y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=(
                X.shape[1] + 1, y.shape[1] if self.task_type != "continuous" else 1))
        
        self.errors_ = []

        if self.early_stopping and self.task_type != "continuous":
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state)

        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                if self.task_type == "continuous":
                    error = target - self.activation(xi, 0)
                    self.weights_[1:, 0] += self.eta0 * xi * error
                    self.weights_[0, 0] += self.eta0 * error
                    cost.append(error ** 2 / 2.0)
                else:
                    for idx in range(self.weights_.shape[1]):
                        error = target[idx] - self.activation(xi, idx)
                        self.weights_[1:, idx] += self.eta0 * xi * error
                        self.weights_[0, idx] += self.eta0 * error
                        cost.append(error ** 2 / 2.0)
            self.errors_.append(np.mean(cost))
            if self.early_stopping and self.task_type != "continuous":
                y_val_pred = self.predict(X_val)
                if y_val_pred.ndim == 1:
                    y_val_pred = y_val_pred[:, np.newaxis]
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self

    def net_input(self, X, idx=0):
        return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]

    def activation(self, X, idx=0):
        return self.net_input(X, idx)
    
    def predict(self, X):
        """
        Predict class labels or continuous values for the input samples.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for classification tasks or continuous values
            for regression tasks.
    
        Notes
        -----
        This method predicts the class labels for classification tasks or continuous
        values for regression tasks based on the input samples `X`. For classification,
        the model uses the net input to determine the class labels. For regression,
        the continuous values are predicted directly from the net input.
    
        For classification tasks:
        - If the task is binary classification, the method uses a threshold of 0.0
          on the activation function to determine class labels.
        - If the task is multi-class classification, the method determines the class
          with the highest activation value.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris, load_boston
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Example for regression
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        >>> # Example for classification
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        sklearn.linear_model.LogisticRegression : Logistic regression classifier.
        sklearn.linear_model.LinearRegression : Linear regression model.
        sklearn.linear_model.SGDClassifier : Linear models (SVM, logistic regression, etc.)
            fitted by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
    
        if self.task_type == "continuous":
            return self.net_input(X)
        else:
            if len(self.label_binarizer_.classes_) == 2:
                return self.label_binarizer_.inverse_transform(
                    np.where(self.activation(X, 0) >= 0.0, 1, 0))
            else:
                activations = np.column_stack(
                    [self.activation(X, idx) for idx in range(self.weights_.shape[1])])
                indices = np.argmax(activations, axis=1)
                binary_arr = np.zeros_like(activations, dtype=int)
                binary_arr[np.arange(activations.shape[0]), indices] = 1
                return self.label_binarizer_.inverse_transform(binary_arr)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Mixte model for classification tasks.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
    
        Raises
        ------
        ValueError
            If the model is used for regression tasks.
    
        Notes
        -----
        This method predicts the probability of each class for the input data `X`
        using the logistic sigmoid function. The sigmoid function maps the net
        input to probabilities:
    
        .. math::
            P(y=1|x) = \frac{1}{1 + \exp(-\phi(z))}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        The method returns a 2D array where each row corresponds to a sample and
        each column corresponds to a class probability. The first column contains
        the probabilities of the negative class and the second column contains the
        probabilities of the positive class.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_proba = model.predict_proba(X_test_std)
        >>> print('Class probabilities:', y_proba)
    
        See Also
        --------
        sklearn.linear_model.LogisticRegression :
            Logistic regression classifier.
        sklearn.linear_model.SGDClassifier : 
            Linear models (SVM, logistic regression, etc.)
            fitted by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        if self.task_type == "continuous":
            raise ValueError("predict_proba is not supported for regression tasks"
                             " with AdalineMixte. Use AdalineRegresssor instead.")
        net_input = self.net_input(X)
        proba_positive_class = 1 / (1 + np.exp(-net_input))
        proba_negative_class = 1 - proba_positive_class
    
        return np.vstack((proba_negative_class, proba_positive_class)).T

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels for 
        classification tasks, and the coefficient of determination R^2 for 
        regression tasks.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
    
        y : array-like of shape (n_samples,)
            True labels for classification tasks, or continuous target values 
            for regression tasks.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
    
        Returns
        -------
        score : float
            Mean accuracy for classification tasks, and R^2 score for 
            regression tasks.
        """
    
        check_is_fitted(self, 'weights_')
        X, y = check_X_y(X, y, estimator= self ) 
        y_pred = self.predict(X)
        
        if self.task_type != "continuous":
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return r2_score(y, y_pred, sample_weight=sample_weight)
        
    def __repr__(self):
        """
        Provide a string representation of the Perceptron object, displaying 
        its parameters in a formatted manner for better readability.
    
        Returns
        -------
        repr : str
            String representation of the Perceptron object with its parameters.
        """
        params = ",\n    ".join(f"{key}={val}" for key, val in self.get_params().items())
        return f"{self.__class__.__name__}(\n    {params}\n)"
    

