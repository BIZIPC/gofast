# -*- coding: utf-8 -*-

"""
:mod:`gofast.estimators.dynamic_system` implements various dynamic system 
models for classification and regression tasks within the gofast library. 
These models are designed to handle complex, time-dependent data by combining 
dynamic system theory with machine learning techniques.
"""
from __future__ import annotations 
import numpy as np
from tqdm import tqdm 

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer # noqa 
from ..tools.validator import check_X_y, check_array 
from ..tools.validator import check_is_fitted
from ..tools.baseutils import normalizer 
from .util import select_default_estimator, validate_memory_depth 

__all__= [
    "HammersteinWienerClassifier","HammersteinWienerRegressor",
    "EnsembleHWClassifier", "EnsembleHWRegressor",
    ]

class HammersteinWienerClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Hammerstein-Wiener Classifier for Dynamic Classification Tasks.

    The Hammerstein-Wiener Classifier is designed for modeling and predicting 
    outcomes in dynamic systems where the response depends on past inputs. It 
    is particularly useful in scenarios where the system's behavior exhibits 
    both linear and nonlinear characteristics. This model uniquely combines 
    the properties of Hammerstein and Wiener systems to effectively capture 
    complex input-output relationships.

    The Hammerstein model component consists of a nonlinear input function 
    followed by a linear dynamic block, whereas the Wiener model component 
    features a linear dynamic block followed by a nonlinear output function. 
    This configuration allows the classifier to capture both linear dynamics 
    and nonlinear transformations inherent in the data.

    The Hammerstein-Wiener model for classification is mathematically 
    expressed as:

    .. math::
        y(t) = g_2\left( \text{{classifier}}\left( \sum_{i=1}^{n} g_1(X_{t-i}) \right) \right)

    where:
    - :math:`g_1` is the nonlinear function applied to inputs.
    - :math:`g_2` is the nonlinear output function, typically a logistic 
      function for classification.
    - 'classifier' denotes a linear model, such as logistic regression, 
      applied within the construct.
    - :math:`X_{t-i}` represents the input features at time step :math:`t-i`, 
      highlighting the memory effect essential for dynamic systems.

    The classifier is especially suited for time-series classification, 
    signal processing, and systems identification tasks where current outputs 
    are significantly influenced by historical inputs. It finds extensive 
    applications in fields like telecommunications, control systems, and 
    financial modeling, where understanding dynamic behaviors is crucial.

    Parameters
    ----------
    nonlinearity_in : str or callable, default='tanh'
        Nonlinear function applied to inputs. This can be a string 
        ('tanh', 'sigmoid', 'relu', 'leaky_relu') to select a predefined 
        function or a callable for a custom function. It transforms the input 
        data before feeding it into the linear dynamic block.

    nonlinearity_out : str or callable, default='sigmoid'
        Nonlinear function applied to the output of the classifier. This can 
        be a string ('sigmoid', 'softmax') to select a predefined function or 
        a callable for a custom function. It models the nonlinear transformation 
        at the output stage.

    memory_depth : int, default=5
        The number of past time steps to consider in the model. This parameter 
        defines the 'memory' of the system, enabling the model to use past 
        information for current predictions.
        
    classifier : object, Optional, default=LogisticRegression()
        Linear classifier model for the dynamic block. Should support fit 
        and predict methods. For multi-class classification, it can be set to 
        use softmax regression (e.g., 
        LogisticRegression with multi_class='multinomial').

    verbose : int, default=False
        Controls the verbosity when fitting.
        
    Attributes
    ----------
    fitted_ : bool
        Indicates whether the classifier has been fitted to the data.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> hw = HammersteinWienerClassifier(
    ...     classifier=LogisticRegression(),
    ...     nonlinearity_in='tanh',
    ...     nonlinearity_out='sigmoid',
    ...     memory_depth=5
    ... )
    >>> X, y = np.random.rand(100, 1), np.random.randint(0, 3, 100)
    >>> hw.fit(X, y)
    >>> y_pred = hw.predict(X)

    Notes
    -----
    The choice of nonlinear functions (:math:`g_1` and :math:`g_2`) and the 
    memory depth are crucial in capturing the dynamics of the system accurately. 
    They should be chosen based on the specific characteristics of the data and 
    the underlying system behavior.

    References
    ----------
    - Hammerstein, A. (1930). Nichtlineare Systeme und Regelkreise.
    - Wiener, N. (1958). Nonlinear Problems in Random Theory.

    See Also
    --------
    LogisticRegression : Standard logistic regression classifier from 
       Scikit-Learn.
    TimeSeriesSplit : Time series cross-validator for Scikit-Learn.
    """

    def __init__(
        self, 
        nonlinearity_in='tanh', 
        nonlinearity_out='sigmoid', 
        memory_depth=5, 
        classifier=None, 
        verbose=False 
        ):
        self.classifier = classifier
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.memory_depth = memory_depth
        self.verbose = verbose 
        
    def _validate_parameters(self):
        """
        Validate and initialize the parameters of the Hammerstein-Wiener Classifier.

        This method ensures that the nonlinearities and classifier model are 
        correctly specified and initializes them if necessary. It also validates 
        the memory depth parameter.

        Nonlinearity functions can be specified either as strings (for predefined 
        functions) or as callables. The following nonlinear functions are supported:
        - 'tanh': Hyperbolic tangent function.
        - 'sigmoid': Sigmoid function, defined as :math:`1 / (1 + e^{-x})`.
        - 'relu': Rectified Linear Unit function, defined as :math:`\max(0, x)`.
        - 'leaky_relu': Leaky Rectified Linear Unit function, defined as 
          :math:`x \text{ if } x > 0 \text{ else } 0.01 \times x`.

        Raises
        ------
        ValueError
            If `nonlinearity_in` or `nonlinearity_out` is not a supported string 
            or a callable function.
            If `memory_depth` is not a positive integer.
            If `classifier` is neither "LogisticRegression" nor an estimator with 
            `fit` and `predict` methods.

        Notes
        -----
        This method is called during the initialization of the estimator to ensure 
        that all parameters are set correctly before fitting the model to data.

        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> hw = HammersteinWienerClassifier(
        ...     nonlinearity_in='sigmoid',
        ...     nonlinearity_out='relu',
        ...     memory_depth=10
        ... )
        >>> hw._validate_parameters()  # This will initialize and validate parameters
        """
        func_dict = {
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x)
        }
        if isinstance(self.nonlinearity_in, str):
            if self.nonlinearity_in not in func_dict:
                raise ValueError(f"nonlinearity_in '{self.nonlinearity_in}'"
                                 f" is not supported. Choose from {list(func_dict.keys())}")
            self.nonlinearity_in = func_dict[self.nonlinearity_in]
        elif not callable(self.nonlinearity_in):
            raise ValueError("nonlinearity_in must be a callable function")

        if isinstance(self.nonlinearity_out, str):
            if self.nonlinearity_out not in func_dict:
                raise ValueError(f"nonlinearity_out '{self.nonlinearity_out}'"
                                 f" is not supported. Choose from {list(func_dict.keys())}")
            self.nonlinearity_out = func_dict[self.nonlinearity_out]
        elif not callable(self.nonlinearity_out):
            raise ValueError("nonlinearity_out must be a callable function")

        self.classifier = select_default_estimator (
            self.classifier or "logit", problem="classification")
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Hammerstein-Wiener model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target classification labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples after adjusting for 
            memory depth.
    
        Returns
        -------
        self : object
            Returns self.
    
        Raises
        ------
        ValueError
            If the length of `sample_weight` does not match the length of the 
            adjusted target array.
    
        Notes
        -----
        This method is responsible for training the Hammerstein-Wiener model. It 
        ensures that the classifier model is fitted with the appropriately transformed 
        input data, taking into account past time steps up to the specified memory 
        depth.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> hw = HammersteinWienerClassifier(
        ...     classifier=LogisticRegression(),
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='sigmoid',
        ...     memory_depth=5
        ... )
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw.fit(X, y)
        >>> print(hw.fitted_)
        True
    
        See Also
        --------
        HammersteinWienerClassifier._preprocess_data : 
            Preprocesses the input data by applying nonlinearity and 
            incorporating memory depth.
        """
        X, y = check_X_y(X, y, estimator=self)
        if self.verbose: 
            print("Fitting Hammerstein Wiener Classifier....")
        X_lagged = self._preprocess_data(X)
        self.memory_depth = validate_memory_depth(
            X, self.memory_depth,default_depth="auto" )
        
        y_adjusted = y[self.memory_depth:]
    
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            if sample_weight.shape[0] != y_adjusted.shape[0]:
                raise ValueError("Sample weights array length must match the"
                                 " adjusted target array length.")
            self.classifier.fit(
                X_lagged, y_adjusted, sample_weight=sample_weight[self.memory_depth:])
        else:
            self.classifier.fit(X_lagged, y_adjusted)
    
        self.fitted_ = True
        if self.verbose: 
            print("Fitting Hammerstein Wiener Classifier completed.")
            
        return self
    
    def _preprocess_data(self, X):
        """
        Preprocess the input data by applying the input nonlinearity and
        incorporating memory depth.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
    
        Returns
        -------
        X_transformed : array-like
            The transformed input data.
    
        Raises
        ------
        ValueError
            If the number of samples in `X` is less than or equal to the memory 
            depth, indicating insufficient data to create lagged features.
    
        Notes
        -----
        This method is essential for preparing the data to be used in the 
        Hammerstein-Wiener model, as it ensures that past information is 
        incorporated into the model, allowing it to capture dynamic behavior 
        effectively.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> hw = HammersteinWienerClassifier(
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='tanh',
        ...     memory_depth=5
        ... )
        >>> X = np.random.rand(10, 3)
        >>> X_lagged = hw._preprocess_data(X)
        >>> print(X_lagged.shape)
        (5, 15)
    
        See Also
        --------
        HammersteinWienerClassifier._validate_parameters : 
            Validates and initializes the parameters.
        """
        self._validate_parameters()
        if self.verbose: 
            print("Start preprocessing X  and control Memory Depth...")
        X_transformed = self.nonlinearity_in(X)
        n_samples, n_features = X_transformed.shape
        if n_samples <= self.memory_depth:
            raise ValueError("Not enough samples to match the memory depth")
        X_lagged = np.zeros((n_samples - self.memory_depth, self.memory_depth * n_features))
        for i in range(self.memory_depth, n_samples):
            X_lagged[i - self.memory_depth, :] = ( 
                X_transformed[i - self.memory_depth:i, :].flatten()
                )
        if self.verbose: 
            print("Preprocess X and Memory depth control completed.")
            
        return X_lagged
    
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted classification labels.
    
        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.
    
        ValueError
            If the input data is not in the correct shape or type.
    
        Notes
        -----
        This method first checks if the model is fitted. It then preprocesses the 
        input data to include past information based on memory depth, applies the 
        classifier model, and finally applies the output nonlinearity to produce the 
        final predictions.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> hw = HammersteinWienerClassifier(
        ...     classifier=LogisticRegression(),
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='sigmoid',
        ...     memory_depth=5
        ... )
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw.fit(X, y)
        >>> y_pred = hw.predict(X)
        >>> print(y_pred.shape)
        (100,)
    
        See Also
        --------
        HammersteinWienerClassifier.fit :
            Fits the Hammerstein-Wiener model to the data.
        HammersteinWienerClassifier._preprocess_data :
            Preprocesses the input data by applying nonlinearity and incorporating memory depth.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X)
        X_lagged = self._preprocess_data(X)
        y_linear = self.classifier.predict(X_lagged)
        y_pred = self.nonlinearity_out(y_linear)
        
        # Adjust for truncated samples
        # Generate default predictions for the first 'memory_depth' samples
        default_prediction = np.array([self.classifier.classes_[0]] * self.memory_depth)
        
        # Concatenate the default predictions with the actual predictions
        y_pred_full = np.concatenate((default_prediction, y_pred))
        
        # Get the predicted class based on the probability threshold of 0.5
        y_pred_full = np.where(y_pred_full >= 0.5, 1, 0)
        
        return y_pred_full
    
    def predict_proba(self, X):
        """
        Probability estimates for the Hammerstein-Wiener model.
    
        The returned estimates for all classes are ordered by the
        label of classes.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
    
        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.
    
        ValueError
            If the input data is not in the correct shape or type.
    
        Notes
        -----
        This method first checks if the model is fitted. It then preprocesses the 
        input data to include past information based on memory depth, applies the 
        classifier model, and finally applies the output nonlinearity to produce the 
        final probability estimates.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> hw = HammersteinWienerClassifier(
        ...     classifier=LogisticRegression(),
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='sigmoid',
        ...     memory_depth=5
        ... )
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw.fit(X, y)
        >>> proba = hw.predict_proba(X)
        >>> print(proba.shape)
        (100, 2)
    
        See Also
        --------
        HammersteinWienerClassifier.fit : 
            Fits the Hammerstein-Wiener model to the data.
        HammersteinWienerClassifier._preprocess_data :
            Preprocesses the input data by applying nonlinearity and incorporating memory depth.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X)
        X_lagged = self._preprocess_data(X)
        proba_linear = self.classifier.predict_proba(X_lagged)
        return normalizer (np.apply_along_axis(self.nonlinearity_out, 1, proba_linear))
    
class HammersteinWienerRegressor(BaseEstimator, RegressorMixin):
    """
    Hammerstein-Wiener Estimator for Nonlinear Dynamic System Identification.

    The Hammerstein-Wiener Estimator models dynamic systems where outputs are 
    nonlinear functions of both past inputs and outputs. It combines a 
    Hammerstein model (nonlinear input followed by linear dynamics) with a 
    Wiener model (linear dynamics followed by nonlinear output), making it 
    adept at capturing the complexities in nonlinear dynamic systems. This 
    estimator is particularly valuable in fields such as control systems, 
    signal processing, and physiological data analysis, where systems often 
    exhibit both linear and nonlinear dynamics over time.

    The estimator's operation is mathematically represented as:

    .. math::
        y(t) = g_2\left( \sum_{i} a_i y(t-i) + \sum_{j} b_j g_1(u(t-j)) \right)

    where:
    - :math:`g_1` and :math:`g_2` are the nonlinear functions applied to the 
      input and output, respectively.
    - :math:`a_i` and :math:`b_j` are the coefficients representing the linear 
      dynamic components of the system.
    - :math:`u(t-j)` and :math:`y(t-i)` denote the system inputs and outputs at 
      various time steps, capturing the past influence on the current output.

    Parameters
    ----------
    nonlinearity_in : str or callable, default='tanh'
        Nonlinear function applied to system inputs. This can be a string 
        ('tanh', 'sigmoid', 'relu', 'leaky_relu') to select a predefined 
        function or a callable for a custom function. The function should be 
        chosen based on the expected nonlinear behavior of the input data.

    nonlinearity_out : str or callable, default='tanh'
        Nonlinear function applied to the output of the linear dynamic block. 
        This can be a string ('tanh', 'sigmoid', 'relu', 'leaky_relu') to 
        select a predefined function or a callable for a custom function. It 
        shapes the final output of the system and should reflect the expected 
        output nonlinearity.

    memory_depth : int, default=5
        The number of past time steps considered in the model. This parameter 
        is crucial for capturing the memory effect in dynamic systems. A higher 
        value means more past data points are used, which can enhance model 
        accuracy but increase computational complexity.

    linear_model : object or str, Optional,  default="LinearRegression"
        A linear model for the dynamic block. This should be an estimator 
        with fit and predict methods (e.g., LinearRegression from scikit-learn). 
        If a string is provided, it must be "LinearRegression". The choice of 
        linear model influences how the system's linear dynamics are captured.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible 
        output across multiple function calls.
        
    verbose : int, default=False
        Controls the verbosity when fitting.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the estimator has been fitted to data.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> import numpy as np
    >>> hw = HammersteinWienerRegressor(
    ...     nonlinearity_in='tanh',
    ...     nonlinearity_out='tanh',
    ...     linear_model=LinearRegression(),
    ...     memory_depth=5
    ... )
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> hw.fit(X, y)
    >>> y_pred = hw.predict(X)

    Notes
    -----
    Selecting appropriate nonlinear functions and memory depth is key to 
    effectively modeling the system. The estimator's performance can be 
    significantly impacted by these choices, and they should be tailored to the 
    specific characteristics of the data and the system being modeled.

    `HammersteinWienerRegressor` is especially suited for applications that 
    require detailed analysis and prediction of systems where the output 
    behavior is influenced by historical input and output data. Its ability to 
    model both linear and nonlinear dynamics makes it indispensable in advanced 
    fields like adaptive control, nonlinear system analysis, and complex signal 
    processing, providing insights and predictive capabilities critical for 
    effective system management.

    References
    ----------
    - Hammerstein, A. (1930). Nichtlineare Systeme und Regelkreise.
    - Wiener, N. (1958). Nonlinear Problems in Random Theory.
    - Narendra, K.S., and Parthasarathy, K. (1990). Identification and Control 
      of Dynamical Systems Using Neural Networks. IEEE Transactions on Neural 
      Networks.

    See Also
    --------
    LinearRegression : Ordinary least squares Linear Regression.
    ARIMA : AutoRegressive Integrated Moving Average model for time-series 
        forecasting.
    ARIMA : AutoRegressive Integrated Moving Average model for time-series 
         forecasting.
    """

    def __init__(
        self,
        nonlinearity_in='tanh', 
        nonlinearity_out='tanh', 
        memory_depth=5, 
        linear_model=None, 
        random_state=None, 
        verbose=False 
        ):
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.linear_model = linear_model
        self.memory_depth = memory_depth
        self.random_state = random_state
        self.verbose = verbose 
        
    def _validate_parameters(self):
        """
        Validate and initialize the parameters of the Hammerstein-Wiener Regressor.
    
        This method ensures that the nonlinearities and linear model are correctly 
        specified and initializes them if necessary. It also validates the memory 
        depth parameter.
    
        Nonlinearity functions can be specified either as strings (for predefined 
        functions) or as callables. The following nonlinear functions are supported:
        - 'tanh': Hyperbolic tangent function.
        - 'sigmoid': Sigmoid function, defined as :math:`1 / (1 + e^{-x})`.
        - 'relu': Rectified Linear Unit function, defined as :math:`\max(0, x)`.
        - 'leaky_relu': Leaky Rectified Linear Unit function, defined as 
          :math:`x \text{ if } x > 0 \text{ else } 0.01 \times x`.
        - 'identity': Identity function, defined as :math:`x`.
    
        Raises
        ------
        ValueError
            If `nonlinearity_in` or `nonlinearity_out` is not a supported string 
            or a callable function.
            If `memory_depth` is not a positive integer.
            If `linear_model` is neither "LinearRegression" nor an estimator with 
            `fit` and `predict` methods.
    
        Notes
        -----
        This method is called during the initialization of the estimator to ensure 
        that all parameters are set correctly before fitting the model to data.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> hw = HammersteinWienerRegressor(
        ...     nonlinearity_in='sigmoid',
        ...     nonlinearity_out='relu',
        ...     memory_depth=10
        ... )
        >>> hw._validate_parameters()  # This will initialize and validate parameters
        """
        func_dict = {
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
            'identity': lambda x: x
        }
        if isinstance(self.nonlinearity_in, str):
            if self.nonlinearity_in not in func_dict:
                raise ValueError(
                    f"nonlinearity_in '{self.nonlinearity_in}' is not"
                    f" supported. Choose from {list(func_dict.keys())}")
            self.nonlinearity_in = func_dict[self.nonlinearity_in]
        elif not callable(self.nonlinearity_in):
            raise ValueError("nonlinearity_in must be a callable function")
    
        if isinstance(self.nonlinearity_out, str):
            if self.nonlinearity_out not in func_dict:
                raise ValueError(
                    f"nonlinearity_out '{self.nonlinearity_out}' "
                    f"is not supported. Choose from {list(func_dict.keys())}")
            self.nonlinearity_out = func_dict[self.nonlinearity_out]
        elif not callable(self.nonlinearity_out):
            raise ValueError("nonlinearity_out must be a callable function")
            
        self.linear_model = select_default_estimator (
            self.linear_model or "lreg")

    def _preprocess_data(self, X):
        """
        Preprocess the input data by applying the input nonlinearity and 
        incorporating memory depth.
    
        This method transforms the input data by first applying the specified 
        nonlinearity function to each feature and then creating lagged versions of 
        the transformed data to capture the memory effects up to the specified 
        memory depth.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
    
        Returns
        -------
        X_lagged : array-like of shape (n_samples - memory_depth, memory_depth * n_features)
            The transformed and lagged input data, ready to be used for fitting 
            the linear model.
    
        Raises
        ------
        ValueError
            If the number of samples in `X` is less than or equal to the memory 
            depth, indicating insufficient data to create lagged features.
    
        Notes
        -----
        This method is essential for preparing the data to be used in the 
        Hammerstein-Wiener model, as it ensures that past information is 
        incorporated into the model, allowing it to capture dynamic behavior 
        effectively.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> hw = HammersteinWienerRegressor(
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='tanh',
        ...     memory_depth=5
        ... )
        >>> X = np.random.rand(10, 3)
        >>> X_lagged = hw._preprocess_data(X)
        >>> print(X_lagged.shape)
        (5, 15)
    
        See Also
        --------
        HammersteinWienerRegressor._validate_parameters :
            Validates and initializes the parameters.
        """
        self._validate_parameters()
        if self.verbose :
            print("Starting Memory depth control... ")
        
        self.memory_depth = validate_memory_depth(
            X, self.memory_depth,default_depth="auto" )
        
        X_transformed = self.nonlinearity_in(X)
        n_samples, n_features = X_transformed.shape
        if n_samples <= self.memory_depth:
            raise ValueError("Not enough samples to match the memory depth")
        X_lagged = np.zeros(
            (n_samples - self.memory_depth, self.memory_depth * n_features))
        for i in range(self.memory_depth, n_samples):
            X_lagged[i - self.memory_depth, :] = ( 
                X_transformed[i - self.memory_depth:i, :].flatten()
                )
        if self.verbose :
            print(" Preprocess X and Memory depth control completed.")
            
        return X_lagged

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Hammerstein-Wiener model to the data.
    
        This method fits the Hammerstein-Wiener model to the provided training 
        data. It preprocesses the data to include past information based on 
        memory depth and fits the linear model to the transformed data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples after adjusting for 
            memory depth.
    
        Returns
        -------
        self : object
            Returns self, an instance of HammersteinWienerRegressor.
    
        Raises
        ------
        ValueError
            If the length of `sample_weight` does not match the length of the 
            adjusted target array.
    
        Notes
        -----
        This method is responsible for training the Hammerstein-Wiener model. It 
        ensures that the linear model is fitted with the appropriately transformed 
        input data, taking into account past time steps up to the specified memory 
        depth.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> hw = HammersteinWienerRegressor(
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='tanh',
        ...     linear_model=LinearRegression(),
        ...     memory_depth=5
        ... )
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> hw.fit(X, y)
        >>> print(hw.fitted_)
        True
    
        See Also
        --------
        HammersteinWienerRegressor._preprocess_data : 
            Preprocesses the input data by applying nonlinearity and incorporating memory depth.
        HammersteinWienerRegressor.predict : 
            Predicts using the Hammerstein-Wiener model.
        """
        X, y = check_X_y(X, y, estimator=self)
        if self.verbose :
            print(" Fitting Hammerstein Wiener Regressor... ")
        X_lagged = self._preprocess_data(X)
        y_adjusted = y[self.memory_depth:]
        
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
            if sample_weight.shape[0] != y_adjusted.shape[0]:
                raise ValueError("Sample weights array length must match"
                                 " the adjusted target array length.")
            self.linear_model.fit(
                X_lagged, y_adjusted, 
                sample_weight=sample_weight[self.memory_depth:])
        else:
            self.linear_model.fit(X_lagged, y_adjusted)

        self.fitted_ = True
        if self.verbose :
            print(" Fitting Hammerstein Wiener Regressor completed. ")
            
        return self
    
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener model.
    
        This method predicts the output for the given input data using the fitted 
        Hammerstein-Wiener model. It preprocesses the input data, applies the 
        linear model to the transformed data, and then applies the output 
        nonlinearity function to the predictions.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
    
        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.
    
        ValueError
            If the input data is not in the correct shape or type.
    
        Notes
        -----
        This method first checks if the model is fitted. It then preprocesses the 
        input data to include past information based on memory depth, applies the 
        linear model, and finally applies the output nonlinearity to produce the 
        final predictions.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> hw = HammersteinWienerRegressor(
        ...     nonlinearity_in='tanh',
        ...     nonlinearity_out='tanh',
        ...     linear_model=LinearRegression(),
        ...     memory_depth=5
        ... )
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> hw.fit(X, y)
        >>> y_pred = hw.predict(X)
        >>> print(y_pred.shape)
        (100,)
    
        See Also
        --------
        HammersteinWienerRegressor.fit : 
            Fits the Hammerstein-Wiener model to the data.
        HammersteinWienerRegressor._preprocess_data : 
            Preprocesses the input data by applying nonlinearity and 
            incorporating memory depth.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(X)
        X_lagged = self._preprocess_data(X)
        y_linear = self.linear_model.predict(X_lagged)
        y_pred_transformed = self.nonlinearity_out(y_linear)
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        y_pred[self.memory_depth:] = y_pred_transformed
        if self.memory_depth > 0:
            default_prediction = np.mean(y_pred_transformed) if len(
                y_pred_transformed) > 0 else 0
            y_pred[:self.memory_depth] = default_prediction
        return y_pred

class EnsembleHWClassifier(BaseEstimator, ClassifierMixin):
    """
    Hammerstein-Wiener Ensemble Classifier.

    `EnsembleHWClassifier` combines the Hammerstein-Wiener model with ensemble 
    learning, effectively managing both linear and nonlinear dynamics within data. 
    It is particularly suited for dynamic systems where outputs depend on 
    historical inputs and outputs, making it ideal for applications in time-series
    forecasting, control systems, and complex scenarios in signal processing 
    or economics.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of base classifiers in the ensemble.
    eta0 : float, default=0.1
        The learning rate for gradient boosting, influencing how base classifier 
        weights are adjusted.
    nonlinearity_in : str or callable, default='tanh'
        Nonlinear function applied to inputs. This can be a string ('tanh', 
        'sigmoid', 'relu', 'leaky_relu') to select a predefined function or 
        a callable for a custom function. It transforms the input data before 
        feeding it into the linear dynamic block.
    nonlinearity_out : str or callable, default='sigmoid'
        Nonlinear function applied to the output of the classifier. This can 
        be a string ('sigmoid', 'softmax') to select a predefined function or 
        a callable for a custom function. It models the nonlinear transformation 
        at the output stage.
    memory_depth : int, default=5
        The number of past time steps to consider in the model. This parameter 
        defines the 'memory' of the system, enabling the model to use past 
        information for current predictions.
    classifier : object or str, Optional, default="LogisticRegression"
        The base classifier to be used in the ensemble. If a string is provided, 
        it must be "LogisticRegression". The classifier should have fit and 
        predict methods.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible 
        output across multiple function calls.
    verbose : int, default=False
        Controls the verbosity when fitting.

    Attributes
    ----------
    base_classifiers_ : list
        List of the instantiated base classifiers.
    weights_ : list
        Weights of each base classifier, determining their influence on the 
        final outcome.

    The Hammerstein-Wiener Ensemble Classifier utilizes the following models and 
    computations:

    1. Hammerstein-Wiener Model:
       .. math::
           y(t) = g_2\left( \sum_{i} a_i y(t-i) + \sum_{j} b_j g_1(u(t-j)) \right)

       where:
       - :math:`g_1` and :math:`g_2` are nonlinear functions applied to the 
         inputs and outputs, respectively.
       - :math:`a_i` and :math:`b_j` are the coefficients of the linear dynamic
         blocks.
       - :math:`u(t-j)` represents the input at time \(t-j\).
       - :math:`y(t-i)` denotes the output at time \(t-i\).

    2. Weight Calculation for Base Classifiers:
       The influence of each classifier is determined by its performance, using:
       
       .. math::
           \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}}

       where :math:`\text{Weighted Error}` is typically evaluated based on the 
       mean squared error or a similar metric.

    Usage and Applications:
    This classifier excels in scenarios requiring modeling of dynamics involving 
    delays or historical dependencies. It is especially effective in time-series 
    forecasting, control systems, and complex signal processing applications.

    See Also 
    ----------
    gofast.estimators.dynamic_system.HammersteinWienerClassifier: 
        Hammerstein-Wiener Classifier for Dynamic Classification Tasks. 
        
    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.dynamic_system import EnsembleHWClassifier
    >>> # Define your own data and labels
    >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)

    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    >>> # Create a Hammerstein-Wiener Ensemble Classifier with default parameters
    >>> hammerstein_wiener_classifier = EnsembleHWClassifier()
    >>> hammerstein_wiener_classifier.fit(X_train, y_train)
    >>> y_pred = hammerstein_wiener_classifier.predict(X_test)
    >>> accuracy = np.mean(y_pred == y_test)
    >>> print('Accuracy:', accuracy)

    Notes
    -----
    - The Hammerstein-Wiener Ensemble Classifier combines the power of the
      Hammerstein-Wiener model with ensemble learning to make accurate
      predictions.
    - The number of base classifiers and the learning rate can be adjusted
      to control the ensemble's behavior.
    - This ensemble is particularly effective when dealing with complex,
      dynamic systems where individual classifiers may struggle.
    - It provides a powerful tool for classification tasks with challenging
      dynamics and nonlinearities.
    """

    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
        nonlinearity_in='tanh', 
        nonlinearity_out='sigmoid', 
        memory_depth=5, 
        classifier=None,
        random_state=None, 
        verbose=False 
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.memory_depth = memory_depth
        self.classifier = classifier
        self.random_state = random_state
        self.verbose=verbose 

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Ensemble Hammerstein-Wiener model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target classification labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples.
    
        Returns
        -------
        self : object
            Returns self.
    
        Raises
        ------
        ValueError
            If the length of `sample_weight` does not match the length of `y`.
    
        Notes
        -----
        This method trains the ensemble Hammerstein-Wiener model using the provided 
        training data and optional sample weights. It employs bootstrapping to 
        create multiple resampled datasets, fits a base classifier on each, and 
        combines their predictions to form the final model.
    
        The procedure involves:
        1. Checking and validating the input data `X` and target labels `y`.
        2. Initializing lists to store the base classifiers and their weights.
        3. For each of the `n_estimators`:
           - Resampling the training data (and sample weights if provided) using 
             bootstrapping.
           - Initializing and training a `HammersteinWienerClassifier` on the 
             resampled data.
           - Calculating the weighted error of the base classifier on the original 
             training data.
           - Computing the weight for the base classifier based on its error.
           - Updating the cumulative predictions with the weighted predictions of 
             the base classifier.
           - Storing the base classifier and its weight.
        4. Returning the fitted estimator.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import EnsembleHWClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw_ensemble = EnsembleHWClassifier(
        ...     n_estimators=50, eta0=0.1, nonlinearity_in='tanh', 
        ...     nonlinearity_out='sigmoid', memory_depth=5, 
        ...     random_state=42)
        >>> hw_ensemble.fit(X, y)
        >>> print(hw_ensemble.fitted_)
        True
        """
        X, y = check_X_y(X, y, estimator=self)
        self.base_classifiers_ = []
        self.weights_ = []
        y_pred = np.zeros(len(y))
        
        if self.verbose: 
            progress_bar = tqdm(
                total=len(self.n_estimators), 
                ascii=True, 
                desc=f'Fitting {self.__class__.__name__}',
                ncols=100
            )
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(
                X, y, n_samples=len(y), random_state=self.random_state, 
                stratify=y, replace=True)
            if sample_weight is not None:
                sample_weight_resampled = resample(
                    sample_weight, n_samples=len(y), 
                    random_state=self.random_state, replace=True)
            else:
                sample_weight_resampled = None
    
            base_classifier = HammersteinWienerClassifier(
                classifier=self.classifier, 
                nonlinearity_in=self.nonlinearity_in, 
                nonlinearity_out=self.nonlinearity_out, 
                memory_depth=self.memory_depth)
            base_classifier.fit(
                X_resampled, y_resampled, sample_weight=sample_weight_resampled)
    
            y_pred_single = base_classifier.predict(X)
            weighted_error = np.sum((y - y_pred_single) ** 2) / len(y)
    
            weight = self.eta0 / (1 + weighted_error)
            y_pred += weight * y_pred_single
    
            self.base_classifiers_.append(base_classifier)
            self.weights_.append(weight)
            
            if self.verbose: 
                progress_bar.update (1)
        
        if self.verbose: 
            progress_bar.close () 
    
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
    
        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.
    
        ValueError
            If the input data is not in the correct shape or type.
    
        Notes
        -----
        This method predicts class labels by aggregating the weighted predictions 
        of all base classifiers in the ensemble. The final prediction is the 
        weighted sum of individual predictions, thresholded at 0.5 to determine 
        the class label.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import EnsembleHWClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw_ensemble = EnsembleHWClassifier(
        ...     n_estimators=50, eta0=0.1, nonlinearity_in='tanh', 
        ...     nonlinearity_out='sigmoid', memory_depth=5, 
        ...     random_state=42)
        >>> hw_ensemble.fit(X, y)
        >>> y_pred = hw_ensemble.predict(X)
        >>> print(y_pred.shape)
        (100,)
        """
        check_is_fitted(self, 'base_classifiers_')
        X = check_array(X)
        y_pred = np.zeros(X.shape[0])
    
        for weight, base_classifier in zip(self.weights_, self.base_classifiers_):
            y_pred += weight * base_classifier.predict(X)
    
        return np.where(y_pred >= 0.5, 1, 0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
    
        The returned estimates for all classes are ordered by the label of classes.
    
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
        NotFittedError
            If the estimator is not fitted yet.
    
        ValueError
            If the input data is not in the correct shape or type.
    
        Notes
        -----
        This method predicts class probabilities by aggregating the weighted 
        predictions of all base classifiers in the ensemble. The cumulative 
        prediction is transformed into probabilities using the sigmoid function.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import EnsembleHWClassifier
        >>> from sklearn.linear_model import LogisticRegression
        >>> import numpy as np
        >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
        >>> hw_ensemble = EnsembleHWClassifier(
        ...     n_estimators=50, eta0=0.1, nonlinearity_in='tanh', 
        ...     nonlinearity_out='sigmoid', memory_depth=5, 
        ...     random_state=42)
        >>> hw_ensemble.fit(X, y)
        >>> proba = hw_ensemble.predict_proba(X)
        >>> print(proba.shape)
        (100, 2)
        """
        check_is_fitted(self, 'base_classifiers_')
        X = check_array(X)
        cumulative_prediction = np.zeros(X.shape[0])
    
        for weight, classifier in zip(self.weights_, self.base_classifiers_):
            cumulative_prediction += weight * classifier.predict(X)
    
        proba_positive_class = 1 / (1 + np.exp(-cumulative_prediction))
        proba_negative_class = 1 - proba_positive_class
    
        return np.vstack((proba_negative_class, proba_positive_class)).T

class EnsembleHWRegressor(BaseEstimator, RegressorMixin):
    """
    Hammerstein-Wiener Ensemble Regressor.

    `EnsembleHWRegressor` combines the Hammerstein-Wiener model with ensemble 
    learning, effectively managing both linear and nonlinear dynamics within data. 
    It is particularly suited for dynamic systems where outputs depend on 
    historical inputs and outputs, making it ideal for applications in time-series
    forecasting, control systems, and complex scenarios in signal processing 
    or economics.
    Hammerstein-Wiener Ensemble (HWE) Regressor.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of base regressors in the ensemble.
    eta0 : float, default=0.1
        The learning rate for gradient boosting, influencing how base regressor 
        weights are adjusted.
    nonlinearity_in : str or callable, default='tanh'
        Nonlinear function applied to inputs. This can be a string ('tanh', 
        'sigmoid', 'relu', 'leaky_relu') to select a predefined function or 
        a callable for a custom function. It transforms the input data before 
        feeding it into the linear dynamic block.
    nonlinearity_out : str or callable, default='identity'
        Nonlinear function applied to the output of the regressor. This can 
        be a string ('identity') to select a predefined function or a callable 
        for a custom function. It models the nonlinear transformation at the 
        output stage.
    regressor : object or str, optional, default="LinearRegression"
        The base regressor to be used in the ensemble. If a string is provided, 
        it must be "LinearRegression". The regressor should have fit and 
        predict methods.
    memory_depth : int, default=5
        The number of past time steps to consider in the model. This parameter 
        defines the 'memory' of the system, enabling the model to use past 
        information for current predictions.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible 
        output across multiple function calls.

    Attributes
    ----------
    base_regressors_ : list
        List of the instantiated base regressors.
    weights_ : list
        Weights of each base regressor, determining their influence on the 
        final outcome.

    Notes 
    ------
    The :class:`EnsembleHWRegressor` assumes that each component model 
    (HammersteinWienerEstimator) is pre-implemented with its fit and predict 
    methods. It fits each individual model to the training data and then 
    averages their predictions to produce the final output.
    
    This approach can potentially improve the performance by capturing different 
    aspects or dynamics of the data with each Hammerstein-Wiener model, and then 
    combining these to form a more robust overall prediction. However, the 
    effectiveness of this ensemble would heavily depend on the diversity and 
    individual accuracy of the included Hammerstein-Wiener models.
    
    The Hammerstein-Wiener Ensemble Regressor utilizes the following models and 
    computations:

    1. Hammerstein-Wiener Model:
       .. math::
           y(t) = g_2\left( \sum_{i} a_i y(t-i) + \sum_{j} b_j g_1(u(t-j)) \right)

       where:
       - :math:`g_1` and :math:`g_2` are nonlinear functions applied to the 
         inputs and outputs, respectively.
       - :math:`a_i` and :math:`b_j` are the coefficients of the linear dynamic
         blocks.
       - :math:`u(t-j)` represents the input at time \(t-j\).
       - :math:`y(t-i)` denotes the output at time \(t-i\).

    2. Weight Calculation for Base Regressors:
       The influence of each regressor is determined by its performance, using:
       
       .. math::
           \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}}

       where :math:`\text{Weighted Error}` is typically evaluated based on the 
       mean squared error or a similar metric.

    Usage and Applications:
    This regressor excels in scenarios requiring modeling of dynamics involving 
    delays or historical dependencies. It is especially effective in time-series 
    forecasting, control systems, and complex signal processing applications.

    See Also 
    ----------
    gofast.estimators.dynamic_system.HammersteinWienerRegressor: 
        Hammerstein-Wiener Regressor for Dynamic Regression Tasks. 
        
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.dynamic_system import EnsembleHWRegressor
    >>> # Define your own data and labels
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)

    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    >>> # Create a Hammerstein-Wiener Ensemble Regressor with default parameters
    >>> hammerstein_wiener_regressor = EnsembleHWRegressor()
    >>> hammerstein_wiener_regressor.fit(X_train, y_train)
    >>> y_pred = hammerstein_wiener_regressor.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test)**2))

    Notes
    -----
    - The Hammerstein-Wiener Ensemble Regressor combines the power of the
      Hammerstein-Wiener model with ensemble learning to make accurate
      predictions.
    - The number of base regressors and the learning rate can be adjusted
      to control the ensemble's behavior.
    - This ensemble is particularly effective when dealing with complex,
      dynamic systems where individual regressors may struggle.
    - It provides a powerful tool for regression tasks with challenging
      dynamics and nonlinearities.
    """

    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
        nonlinearity_in='tanh', 
        nonlinearity_out='identity', 
        memory_depth=5, 
        regressor=None,
        random_state=None, 
        verbose=False 
        
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.regressor = regressor
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.memory_depth = memory_depth
        self.random_state = random_state
        self.verbose=verbose 

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Ensemble Hammerstein-Wiener model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target regression values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If the length of `sample_weight` does not match the length of `y`.

        Notes
        -----
        This method trains the ensemble Hammerstein-Wiener model using the provided 
        training data and optional sample weights. It employs bootstrapping to 
        create multiple resampled datasets, fits a base regressor on each, and 
        combines their predictions to form the final model.
        """
        X, y = check_X_y(X, y, estimator=self)
        self.base_regressors_ = []
        self.weights_ = []
        y_pred = np.zeros(len(y))
    
        if self.verbose: 
            progress_bar = tqdm(
                total=len(self.n_estimators), 
                ascii=True, 
                desc=f'Fitting {self.__class__.__name__}',
                ncols=100
                )
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(
                X, y, n_samples=len(y), random_state=self.random_state, 
                stratify=y if sample_weight is None else None, replace=True)
            if sample_weight is not None:
                sample_weight_resampled = resample(
                    sample_weight, n_samples=len(y), 
                    random_state=self.random_state, replace=True)
            else:
                sample_weight_resampled = None

            base_regressor = HammersteinWienerRegressor(
                linear_model=self.regressor,
                nonlinearity_in=self.nonlinearity_in, 
                nonlinearity_out=self.nonlinearity_out, 
                memory_depth=self.memory_depth
            )
            base_regressor.fit(
                X_resampled, y_resampled, sample_weight=sample_weight_resampled)

            y_pred_single = base_regressor.predict(X)
            weighted_error = np.sum((y - y_pred_single) ** 2) / len(y)

            weight = self.eta0 / (1 + weighted_error)
            y_pred += weight * y_pred_single

            self.base_regressors_.append(base_regressor)
            self.weights_.append(weight)
            
            if self.verbose: 
                progress_bar.update (1)
        
        if self.verbose: 
            progress_bar.close () 

        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted regression values.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.

        ValueError
            If the input data is not in the correct shape or type.

        Notes
        -----
        This method predicts regression values by aggregating the weighted predictions 
        of all base regressors in the ensemble. The final prediction is the weighted 
        sum of individual predictions.

        Examples
        --------
        >>> from gofast.estimators.dynamic_system import EnsembleHWRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> hw_ensemble = EnsembleHWRegressor(
        ...     n_estimators=50, eta0=0.1, regressor='LinearRegression', 
        ...     nonlinearity_in='tanh', nonlinearity_out='identity', 
        ...     memory_depth=5, random_state=42)
        >>> hw_ensemble.fit(X, y)
        >>> y_pred = hw_ensemble.predict(X)
        >>> print(y_pred.shape)
        (100,)
        """
        check_is_fitted(self, 'base_regressors_')
        X = check_array(X)
        y_pred = np.zeros(X.shape[0])

        for weight, base_regressor in zip(self.weights_, self.base_regressors_):
            y_pred += weight * base_regressor.predict(X)

        return y_pred

    def decision_function(self, X):
        """
        Calculate the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values for the samples.

        Raises
        ------
        NotFittedError
            If the estimator is not fitted yet.

        ValueError
            If the input data is not in the correct shape or type.

        Notes
        -----
        This method calculates the decision function by aggregating the weighted 
        outputs of all base regressors in the ensemble. The decision function 
        values indicate the aggregated predictions before applying any final 
        transformation (e.g., nonlinearity_out).

        Examples
        --------
        >>> from gofast.estimators.dynamic_system import EnsembleHWRegressor
        >>> from sklearn.linear_model import LinearRegression
        >>> import numpy as np
        >>> X, y = np.random.rand(100, 1), np.random.rand(100)
        >>> hw_ensemble = EnsembleHWRegressor(
        ...     n_estimators=50, eta0=0.1, regressor='LinearRegression', 
        ...     nonlinearity_in='tanh', nonlinearity_out='identity', 
        ...     memory_depth=5, random_state=42)
        >>> hw_ensemble.fit(X, y)
        >>> decision = hw_ensemble.decision_function(X)
        >>> print(decision.shape)
        (100,)
        """
        check_is_fitted(self, 'base_regressors_')
        X = check_array(X)
        decision = np.zeros(X.shape[0])

        for weight, base_regressor in zip(self.weights_, self.base_regressors_):
            decision += weight * base_regressor.predict(X)

        return decision




