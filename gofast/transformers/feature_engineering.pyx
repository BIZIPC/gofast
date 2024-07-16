# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides a comprehensive collection of classes for feature engineering, 
including feature selection, transformation, scaling, and encoding to enhance 
machine learning model inputs."""

from __future__ import division

import itertools 
import warnings 
import numpy as np 
import pandas as pd 
from scipy import sparse

from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from .._gofastlog import gofastlog 
from ..api.types import _F, Union, Optional
from ..exceptions import EstimatorError, NotFittedError 
from ..tools.coreutils import  parse_attrs, assert_ratio, validate_feature
from ..tools.coreutils import  ellipsis2false, to_numeric_dtypes, is_iterable
from ..tools.coreutils import exist_features, type_of_target
from ..tools.validator import  get_estimator_name, check_X_y, is_frame
from ..tools.validator import _is_arraylike_1d, build_data_if, check_array 
from ..tools.validator import check_is_fitted, check_consistent_length 

__all__= [
    'FeatureImportanceSelector', 
    'SequentialBackwardSelector',
    'FloatCategoricalToInt', 
    'KMeansFeaturizer',
    'AttributesCombinator', 
    'StratifyFromBaseFeature',
    'CategoryBaseStratifier', 
    'CategorizeFeatures', 
    'FrameUnion', 
    'FrameUnionFlex', 
    'DataFrameSelector',
    'BaseColumnSelector', 
    'BaseCategoricalEncoder', 
    'BaseFeatureScaler', 
    'CombinedAttributesAdder', 
    'FeaturizeX', 
    'FeatureSelectorByModel', 
    'PolynomialFeatureCombiner', 
    'DimensionalityReducer', 
    'CategoricalEncoder', 
    'CategoricalEncoder2',
    'FeatureScaler', 
    'MissingValueImputer', 
    'ColumnSelector', 
    'LogTransformer', 
    'CategoryFrequencyEncoder', 
  ]

class FeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses importance scores from a fitted model
    to select features exceeding a specified importance threshold.

    This transformer integrates with scikit-learn and can use any model
    that exposes "feature_importances_" or similar attributes after fitting.
    By default, it uses a RandomForest model to determine importance but can
    be configured to use any other similar estimator.

    Parameters
    ----------
    model : estimator object implementing 'fit', default=None
        The model to use for assessing feature importance. If None, a default
        RandomForest model is used based on the classification or regression
        nature of the target variable specified in the fit method.
    threshold : float, default=0.5
        The threshold value for feature importances. Features with importance
        scores above this threshold will be kept.
    use_classifier : bool or 'auto', default=True
        Determines whether to use a classifier (True) or a regressor (False).
        If 'auto', the choice is made based on the target variable's type
        (continuous for regressors and categorical for classifiers).
    max_depth : int, optional
        The maximum depth of the tree, applicable if the default model is used.
    n_estimators : int, default=100
        The number of trees in the forest, applicable if the default model is used.
    rf_kwargs : dict, optional
        Additional keyword arguments to pass to the RandomForest constructor.

    Attributes
    ----------
    important_indices_ : ndarray
        Indices of features considered important based on the importance threshold.
    feature_names_ : ndarray
        Feature names extracted from the input DataFrame, if provided.

    Examples
    --------
    >>> from gofast.transformers import FeatureImportanceSelector
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> selector = FeatureImportanceSelector(threshold=0.1, use_classifier=True)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(selector.get_feature_names_out())

    Notes
    -----
    This selector is particularly useful in scenarios where dimensionality
    reduction based on feature importance is required to improve model
    performance or interpretability.
    """

    def __init__(
        self, 
        model=None, 
        threshold=0.5, 
        use_classifier=True, 
        max_depth=None, 
        n_estimators=100, 
        rf_kwargs=None
    ):
        self.model = model
        self.threshold = threshold
        self.use_classifier = use_classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.rf_kwargs = rf_kwargs or {}
        self.important_indices_ = None
        self.feature_names_ = None 
        
    def _select_model_if(self, y):
        """
        Selects and initializes the model based on the type of target variable
        and user preference.
    
        This method is responsible for dynamically selecting either a 
        regression or classification model based on the type of the target 
        data (`y`). If `use_classifier` is 'auto', the method automatically 
        detects the type of task (regression or classification) based on the 
        target variable type.
    
        Parameters
        ----------
        y : array-like
            Target variable array that is used to determine the model type if 
            `use_classifier` is set to 'auto'.
    
        Returns
        -------
        model : object
            An instance of either RandomForestRegressor or RandomForestClassifier
            depending on the detected type of task or user preference.
    
        Raises
        ------
        ValueError
            If `y` is None and `use_classifier` is set to 'auto', a ValueError
            is raised indicating that the target variable must be provided.
    
        Notes
        -----
        This method supports customization through the `rf_kwargs` attribute 
        which allows passing additional parameters to the RandomForest model.
        """
        # Mapping the selector based on the user input
        _model = {
            "regression_selector": RandomForestRegressor, 
            "classification_selector": RandomForestClassifier
        }
        if self.use_classifier == 'auto':
            # Automatically detect based on the type of target
            if y is None:
                raise ValueError(
                    "Target vector 'y' must not be None when use_classifier='auto'.")
            target_type = type_of_target(y)
            selector_key = ( 'regression_selector' if target_type == 'continuous' 
                            else 'classification_selector' ) 
            self.use_classifier = False if target_type == 'continuous' else True
        else:
            selector_key = ( 
                'classification_selector' if self.use_classifier else 'regression_selector'
                )

        return _model[selector_key](
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            **self.rf_kwargs
            )
        
    def fit(self, X, y=None):
        """
        Fits the model to the input data `X` and target `y`.
    
        This method fits the feature importance selector to the data by first 
        determining the appropriate model to use (either automatically or 
        based on user preference), then fitting this model to the data. It 
        also handles checking and warning about potentialmismatches between 
        the model type and the target variable type.
    
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape [n_samples], optional
            Target values. Required unless the model explicitly provided does 
            not need a target variable.
    
        Returns
        -------
        self : object
            Returns self.
    
        Raises
        ------
        ValueError
            If `y` is None when no explicit model is provided.
        TypeError
            If the model does not support the fit method.
        ValueError
            If the model lacks the `feature_importances_` attribute necessary 
            for feature selection.
    
        Notes
        -----
        It stores the feature names if `X` is a DataFrame to use later in 
        `get_feature_names_out`.
        """
        if self.model is None:
            if y is None:
                raise ValueError("Target vector 'y' must not be None when no"
                                 " model is explicitly provided.")
    
            self.model = self._select_model_if(y)
            target_type = type_of_target(y)
            model_type = (
                'Regressor' if 'Regressor' in self.model.__class__.__name__ 
                else 'Classifier'
            )
    
            if model_type == 'Regressor' and target_type != 'continuous':
                warnings.warn(
                    "Regressor is selected while the task seems to be classification.",
                    UserWarning)
            elif model_type == 'Classifier' and target_type == 'continuous':
                warnings.warn(
                    "Classifier is selected while the task seems to be regression.",
                    UserWarning)
    
        # Fit the model and check for feature_importances_ attribute
        if not hasattr(self.model, 'feature_importances_'):
            if not hasattr(self.model, 'fit'):
                raise TypeError(
                    f"The model {self.model.__class__.__name__} does not"
                    " support the fit method.")
            try:
                self.model.fit(X, y)
            except Exception as e:
                raise ValueError(
                    "The model used does not have feature_importances_ attribute."
                ) from e
        
        # Store feature names if X is a DataFrame
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns

        # Fetch important features based on the threshold
        self.important_indices_ = np.where(
            self.model.feature_importances_ > self.threshold)[0]
    
        return self
    
    def transform(self, X):
        """
        Transforms the dataset to include only the most important features as
        determined during fitting.
    
        This method reduces the dimensionality of the data to only include 
        features that are considered important based on the threshold set 
        during the initialization of the selector.
    
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The input samples to transform.
    
        Returns
        -------
        X_transformed : array-like, shape [n_samples, n_selected_features]
            The array of input samples but only including the features that 
            were deemed important.
    
        Raises
        ------
        RuntimeError
            If this method is called before the model is fitted.
        """
        
        if self.important_indices_ is None:
            raise RuntimeError("The fit method must be called before transform.")
        if len(self.important_indices_) == 0:
            warnings.warn(
                "No important features were detected based on the provided "
                f"threshold({self.threshold}). Consider reducing the threshold"
                " or revisiting feature engineering.", UserWarning
            )
        #     return X
        
        if isinstance (X, pd.DataFrame ): 
            return X.iloc[:, self.important_indices_]
        
        return X[:, self.important_indices_]
    

    def get_feature_names_out(self):
        """
        Returns the names of the features deemed important by the model.
    
        This method returns the names of the features that have been selected 
        as important. This can either be the original feature names from a 
        DataFrame or constructed names if the input was an array.
    
        Returns
        -------
        feature_names : list of str
            The names of the features that have been selected as important.
    
        Raises
        ------
        RuntimeError
            If this method is called before the model is fitted.
        """

        if self.important_indices_ is None:
            raise RuntimeError(
                "The fit method must be called before fetching feature names.")

        if self.feature_names_ is not None:
            # Use column names if X was a DataFrame
            return self.feature_names_[self.important_indices_].tolist()
        else:
            # Default feature names if X was an array
            return [f"feature_{i}" for i in self.important_indices_]

class FloatCategoricalToInt(BaseEstimator, TransformerMixin):
    """
    A transformer that detects floating-point columns in a DataFrame 
    representing categorical variables and converts them to integers.

    This transformer is useful when dealing with datasets where categorical 
    variables are represented as floating-point numbers but essentially 
    contain integer values, for example, [0.0, 1.0, 2.0] representing different 
    categories.

    Parameters
    ----------
    dtype : {'auto', dtype}, default='auto'
        If `dtype` is 'auto' and `as_frame` is False, the transformed array 
        dtype is coerced to object. For other values of `dtype` when `as_frame`
        is False, the array is returned as it is, and the transformation may 
        be meaningless if not converting all array elements to integers or 
        keeping them as floats.
    as_frame : bool, default=True
        If `as_frame` is True, the data is kept as a DataFrame. If not passed 
        as a frame, a default `col_prefix` is created.
    col_prefix : str, default='feature'
        The prefix to use for column names if a numpy array is passed. This can 
        be changed if it is not intuitive enough.

    Attributes
    ----------
    columns_to_transform_ : list
        List of column names in the DataFrame that are identified as 
        floating-point columns to be transformed to integers.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from gofast.transformers.feature_engineering import FloatCategoricalToInt
    >>> data = {'category': [0.0, 1.0, 2.0, 1.0], 'value': [23.5, 12.6, 15.0, 22.1]}
    >>> df = pd.DataFrame(data)
    >>> transformer = FloatCategoricalToInt()
    >>> transformer.fit(df)
    >>> transformed = transformer.transform(df)
    >>> print(transformed)
       category  value
    0         0   23.5
    1         1   12.6
    2         2   15.0
    3         1   22.1

    Notes
    -----
    The fit method determines which columns are to be transformed by checking 
    if the unique values in each floating-point column are integers ending 
    with .0. During the transform phase, these columns are then cast to the 
    integer type, preserving their categorical nature but in a more 
    memory-efficient format.

    The transformer does not modify the input DataFrame directly; instead, it 
    returns a transformed copy.

    The mathematical formulation for determining if a floating-point column 
    contains integer values is based on checking the modulo operation:
    
    .. math::
        \text{if } \forall x \in \text{column}, \ x \mod 1 == 0,\\
            \ \text{then the column is converted to integers}

    See Also
    --------
    pandas.DataFrame.astype : Cast a pandas object to a specified dtype.
    numpy.mod : Return element-wise remainder of division.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """

    def __init__(self, dtype='auto', as_frame=True, col_prefix='feature'): 
        self.dtype = dtype  
        self.as_frame = as_frame  
        self.col_prefix = col_prefix  

    def fit(self, X, y=None):
        """
        Fit the transformer to the DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self
    
    def transform(self, X):
        """
        Transform the DataFrame by converting identified floating-point columns
        to integers.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            The input data to transform.

        Returns
        -------
        X_transformed : pandas.DataFrame or numpy.ndarray
            The transformed data with floating-point columns converted to integers.
        """
        original_type = 'pd.DataFrame'
        if isinstance(X, np.ndarray): 
            original_type = 'np.ndarray'
        X = build_data_if(X, input_name=self.col_prefix, raise_exception=True,
                          force=True)

        # Identify columns to transform based on their unique values
        self.columns_to_transform_ = [
            col for col in X.columns if X[col].dtype == float and all(np.mod(X[col], 1) == 0)
        ]

        # Copy DataFrame to avoid modifying the original data
        X_transformed = X.copy()

        # Convert identified columns to integer type
        for col in self.columns_to_transform_:
            X_transformed[col] = X_transformed[col].astype(int)

        # Revert to original type if input was numpy array and `as_frame` is False
        if original_type == 'np.ndarray' and not self.as_frame:
            if self.dtype == 'auto': 
                X_transformed = X_transformed.to_numpy()
            else:
                X_transformed = X_transformed.astype(self.dtype).values

        return X_transformed

class SequentialBackwardSelector(BaseEstimator, TransformerMixin):
    r"""
    Sequential Backward Selection (SBS)
    
    SBS is a feature selection algorithm aimed at reducing the dimensionality
    of the initial feature subspace with minimal performance decay in the
    classifier, thereby enhancing computational efficiency. In certain cases,
    SBS can even improve the predictive power of a model, particularly in
    scenarios of overfitting.
    
    The core concept of SBS is to sequentially remove features from the full
    feature subset until the remaining feature subspace contains the desired
    number of features. To determine which feature to remove at each stage,
    a criterion function :math:`J` is minimized [1]_. Essentially, the criterion
    is the difference in classifier performance before and after removing a
    particular feature. The feature removed at each stage is the one that
    maximizes this criterion, meaning it has the least impact on performance
    when removed. The SBS algorithm can be outlined in the following steps:
    
    - Initialize with :math:`k=d`, where :math:`d` is the dimensionality
      of the full feature space, :math:`X_d`.
    - Identify the feature :math:`x^{-}` that maximizes the criterion:
      :math:`x^{-} = argmax J(X_k - x)`, where :math:`x \in X_k`.
    - Remove the feature :math:`x^{-}` from the set, updating
      :math:`X_{k+1} = X_k - x^{-}; k = k - 1`.
    - Terminate if :math:`k` equals the desired number of features;
      otherwise, repeat from step 2. [2]_
    
    Parameters
    ----------
    estimator : callable or instantiated object
        A callable or an instance of a classifier/regressor with a `fit` method.
    k_features : int, default=1
        The starting number of features for selection. Must be less than the
        total number of features in the training set.
    scoring : callable or str, default='accuracy'
        Metric for scoring. Available metrics are 'precision', 'recall',
        'roc_auc', or 'accuracy'. Other metrics will raise an error.
    test_size : float or int, default=None
        If float, represents the proportion of the dataset for the test split.
        If int, represents the absolute number of test samples. Defaults to the
        complement of the train size if None. If `train_size` is also None, it
        defaults to 0.25.
    random_state : int, RandomState instance, or None, default=None
        Controls the shuffling applied to the data before the split.
        An integer value ensures reproducible results across multiple function calls.
    
    Attributes
    ----------
    feature_names_ : ndarray of shape (n_features_in,)
        This attribute stores the names of the features that the model has been 
        trained on. It is particularly useful for understanding which features 
        were present in the input dataset `X` during the `fit` method call. This 
        attribute is only defined if the input `X` has string type feature names.

    selected_indices_ : tuple
        Contains the indices of the features in the final selected subset after 
        the model fitting process. These indices correspond to the features in 
        `feature_names_` that the algorithm has identified as the most 
        significant or relevant for the model.

    feature_subsets_ : list of tuples
        A list where each tuple represents a subset of features selected at each 
        step of the Sequential Backward Selection process. It tracks the 
        evolution of feature selection over the course of the algorithm's 
        execution, showing the progressive elimination of features.

    model_scores_ : list of floats
        This list contains the scores of the model corresponding to each of the 
        feature subsets stored in `feature_subsets_`. The scores are calculated 
        during the cross-validation process within the algorithm. They provide 
        insight into the performance of the model as different features are 
        removed.

    optimal_score_ : float
        Represents the highest score achieved by the model using the feature 
        subset specified by the `k_features` parameter. It signifies the 
        performance of the model with the optimally selected features, giving 
        an idea of how well the model can perform after the feature selection 
        process.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.base import SequentialBackwardSelector
    >>> X, y = fetch_data('bagoue analysed') # data already standardized
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs = SequentialBackwardSelector(knn)
    >>> sbs.fit(X_train, y_train)
    
    References
    ----------
    .. [1] Raschka, S., Mirjalili, V., Python Machine Learning, 3rd ed., Packt, 2019.
    .. [2] Ferri F., Pudil P., Hatef M., Kittler J., Comparative study of
           techniques for large-scale feature selection, pages 403-413, 1994.
           
    """

    _scorers = {
        'accuracy': accuracy_score,
        'recall': recall_score,
        'precision': precision_score,
        'roc_auc': roc_auc_score
    }

    def __init__(
        self, 
        estimator, 
        k_features=1, 
        scoring='accuracy', 
        test_size=0.25, 
        random_state=42
        ):
        self.estimator = estimator
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the Sequential Backward Selection (SBS) model to the training data.

        This method involves splitting the dataset into training and validation 
        subsets within the fit function itself. The SBS algorithm uses this 
        internal validation set, distinct from any external test set, to 
        evaluate feature importance and make selection decisions. 
        This approach ensures that the original test set remains untouched 
        and is not used inadvertently during the training process.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data, where each sample is represented as a feature vector. 
            `X` can be a matrix of features, or it might require preprocessing  
            (e.g., using a feature extractor or a pairwise metric) before being 
            used. n_samples is the number of samples, and n_features is the  
            number of features.
        y : array-like of shape (n_samples,)
            Target values corresponding to the training data. These are the 
            dependent variables that the model is trained to predict.

        Returns
        -------
        self : object
            The fitted `SequentialBackwardSelection` instance, allowing for 
            method chaining.
        """
        X, y = check_X_y(X, y, estimator=self, multi_output=True, to_frame=True)
        self._validate_params(X)
        
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
            X = X.values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        self.indices_ = tuple(range(X_train.shape[1]))
        self.subsets_ = [self.indices_]
        self.scores_ = [self._compute_score(X_train, X_test, y_train, y_test, self.indices_)]

        while len(self.indices_) > self.k_features:
            scores, subsets = [], []
            for subset in itertools.combinations(self.indices_, r=len(self.indices_)-1):
                score = self._compute_score(X_train, X_test, y_train, y_test, subset)
                scores.append(score)
                subsets.append(subset)

            best_score_index = np.argmax(scores)
            self.indices_ = subsets[best_score_index]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best_score_index])

        self.k_score_ = self.scores_[-1]
        
        return self

    def transform(self, X):
        """
        Transform the dataset to contain only the selected features.
        
        This method reduces the feature set to the ones selected during fitting.
        It handles both pandas DataFrames and numpy arrays appropriately,
        indexing into the data structure to select the specified features.
        
        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            The input samples to transform.
        
        Returns
        -------
        X_transformed : DataFrame or array-like of shape (n_samples, k_features)
            The dataset with only the selected features.
        
        Raises
        ------
        NotFittedError
            If this method is called before the instance is fitted.
        """
        if not hasattr(self, 'indices_'):
            raise NotFittedError(
                "This SequentialBackwardSelector instance is not fitted yet. "
               "Call 'fit' with appropriate arguments before using this estimator.")
        
        # Depending on the input type, use the appropriate indexing method
        if isinstance(X, pd.DataFrame):
            X_transformed = X.iloc[:, list(self.indices_)]
        else:
            X_transformed = X[:, list(self.indices_)]
        
        return X_transformed

    def _compute_score(self, X_train, X_test, y_train, y_test, indices):
        """
        Compute the score of a subset of features.

        Internally used to evaluate the performance of the model on a given
        subset of features.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        X_test : array-like of shape (n_samples, n_features)
            The testing input samples.
        y_train : array-like of shape (n_samples,)
            The training target values.
        y_test : array-like of shape (n_samples,)
            The testing target values.
        indices : array-like of shape (n_features,)
            The indices of the features to be used.

        Returns
        -------
        score : float
            The score of the estimator on the provided feature subset.
        """
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        return self.scoring(y_test, y_pred)

    def _validate_params(self, X):
        """
        Validate the parameters of the estimator.

        This method checks the compatibility of the parameters with the input data
        and raises appropriate errors if invalid parameters are detected.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples used for validation.

        Raises
        ------
        TypeError
            If the estimator does not have a 'fit' method.
        ValueError
            If `k_features` is greater than the number of features in X.
        """
        if not hasattr(self.estimator, 'fit'):
            raise TypeError("Estimator must have a 'fit' method.")
        
        self.k_features = int(self.k_features)
        if self.k_features > X.shape[1]:
            raise ValueError(
                f"k_features must be <= number of features in X ({X.shape[1]}).")

        if callable(self.scoring) or hasattr(self.scoring, '__call__'):
            self.scoring = self.scoring.__name__.replace('_score', '')
        
        if self.scoring not in self._scorers:
            valid_scorers = ", ".join(self._scorers.keys())
            raise ValueError(
                f"Invalid scoring method. Valid options are: {valid_scorers}")

        self.scoring = self._scorers[self.scoring]

    def __repr__(self):
        """
        Provide a string representation of the SequentialBackwardSelection 
        instance.

        This method is useful for debugging and provides an easy-to-read summary
        of the configuration of the SequentialBackwardSelection instance.

        Returns
        -------
        representation : str
            The string representation of the instance.
        """
        class_name = self.__class__.__name__
        params = self.get_params()
        params_str = ", ".join(f"{key}={value!r}" for key, value in params.items())
        return f"{class_name}({params_str})"

class KMeansFeaturizer(BaseEstimator, TransformerMixin):
    """Transforms numeric data into k-means cluster memberships.
     
    This transformer runs k-means on the input data and converts each data point
    into the ID of the closest cluster. If a target variable is present, it is 
    scaled and included as input to k-means in order to derive clusters that
    obey the classification boundary as well as group similar points together.
    
    Parameters 
    -------------
    n_clusters: int, default=7
       Number of initial clusters
    target_scale: float, default=5.0 
       Apply appropriate scaling and include it in the input data to k-means.
    n_components: int, optional
       Number of components for reducted down the predictor. It uses the PCA 
       to reduce down dimension to the importance components. 
    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centroids using sampling based on
        an empirical probability distribution of the points' contribution to the
        overall inertia. This technique speeds up convergence. The algorithm
        implemented is "greedy k-means++". It differs from the vanilla k-means++
        by making several trials at each sampling step and choosing the best centroid
        among them.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : 'auto' or int, default=10
        Number of times the k-means algorithm is run with different centroid
        seeds. The final results is the best output of `n_init` consecutive runs
        in terms of inertia. Several runs are recommended for sparse
        high-dimensional problems (see :ref:`kmeans_sparse_high_dim`).

        When `n_init='auto'`, the number of runs will be 10 if using
        `init='random'`, and 1 if using `init='kmeans++'`.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    algorithm : {"lloyd", "elkan", "auto", "full"}, default="lloyd"
        K-means algorithm to use. The classical EM-style algorithm is `"lloyd"`.
        The `"elkan"` variation can be more efficient on some datasets with
        well-defined clusters, by using the triangle inequality. However it's
        more memory intensive due to the allocation of an extra array of shape
        `(n_samples, n_clusters)`.

        `"auto"` and `"full"` are deprecated and they will be removed in
        Scikit-Learn 1.3. They are both aliases for `"lloyd"`.

    to_sparse : bool, default=False
            If True, the input data `X` will be converted to a sparse matrix
            before applying the transformation. This is useful for handling
            large datasets more efficiently. If False, the data format of `X`
            is preserved.

    encoding : {'onehot', 'bin-counting', 'label', 'frequency', 'mean_target'},\
        default='onehot'
        Encoding strategy for cluster labels:
        - 'onehot': One-hot encoding of the cluster assignments.
        - 'bin-counting': Probabilistic bin-counting encoding.
        - 'label': Label encoding of the cluster assignments.
        - 'frequency': Frequency encoding of the cluster assignments.
        - 'mean_target': Mean target encoding based on target values provided during fit.
    
    Attributes 
    -----------
    km_model: KMeans featurization model used to transform

    Examples 
    --------
    >>> # (1) Use a common dataset 
    >>> import matplotlib.pyplot as plt 
    >>> from sklearn.datasets import make_moons
    >>> from gofast.plot.utils import plot_voronoi 
    >>> from gofast.datasets import load_mxs 
    >>> X, y = make_moons(n_samples=5000, noise=0.2)
    >>> kmf_hint = KMeansFeaturizer(n_clusters=50, target_scale=10).fit(X,y)
    >>> kmf_no_hint = KMeansFeaturizer(n_clusters=50, target_scale=0).fit(X, y)
    >>> fig, ax = plt.subplots(2,1, figsize =(7, 7)) 
    >>> plot_voronoi ( X, y ,cluster_centers=kmf_hint.cluster_centers_, 
                      fig_title ='KMeans with hint', ax = ax [0] )
    >>> plot_voronoi ( X, y ,cluster_centers=kmf_no_hint.cluster_centers_, 
                      fig_title ='KMeans No hint' , ax = ax[1])
    <AxesSubplot:title={'center':'KMeans No hint'}>
    >>> # (2)  Use a concrete data set 
    >>> X, y = load_mxs ( return_X_y =True, key ='numeric' ) 
    >>> # get the most principal components 
    >>> from gofast.analysis import nPCA 
    >>> Xpca =nPCA (X, n_components = 2  ) # veronoi plot expect two dimensional data 
    >>> kmf_hint = KMeansFeaturizer(n_clusters=7, target_scale=10).fit(Xpca,y)
    >>> kmf_no_hint = KMeansFeaturizer(n_clusters=7, target_scale=0).fit(Xpca, y)
    >>> fig, ax = plt.subplots(2,1, figsize =(7, 7)) 
    >>> plot_voronoi ( Xpca, y ,cluster_centers=kmf_hint.cluster_centers_, 
                      fig_title ='KMeans with hint', ax = ax [0] )
    >>> plot_voronoi ( Xpca, y ,cluster_centers=kmf_no_hint.cluster_centers_, 
                      fig_title ='KMeans No hint' , ax = ax[1])
    
    References 
    ------------
    .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
          K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
          Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3

    """
    def __init__(
        self, 
        n_clusters=2, 
        target_scale=1.0, 
        random_state=None, 
        n_components=None, 
        init='k-means++', 
        n_init="auto", 
        max_iter=300, 
        tol=1e-4, 
        copy_x=True,
        verbose=0, 
        algorithm='lloyd', 
        to_sparse=False,
        encoding='onehot' 
        ):
        self.n_clusters = n_clusters
        self.target_scale = target_scale
        self.random_state = random_state
        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.copy_x = copy_x
        self.verbose = verbose
        self.algorithm = algorithm
        self.to_sparse = to_sparse
        self.encoding = encoding

    def fit(self, X, y=None):
        """
        Fit the KMeansFeaturizer to the data.
    
        The `fit` method applies PCA if `n_components` is specified, scales
        the target variable if provided, and fits the KMeans model to the data.
    
        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Training instances to cluster. It must be a dense array or a sparse matrix.
        y : array-like of shape (n_samples,), default=None
            Target values (class labels in classification, real numbers in regression).
            If provided, the target values are scaled and included in the clustering 
            process to guide the clustering.
    
        Returns
        -------
        self : object
            Returns the instance itself.
    
        Notes
        -----
        If `n_components` is specified, PCA is applied to reduce the dimensionality
        of the input data `X`. This is particularly useful for high-dimensional data.
    
        If `y` is provided, it is scaled by the `target_scale` parameter and
        concatenated with `X` to influence the clustering. This helps in creating 
        clusters that are more informative with respect to the target variable.
    
        The KMeans algorithm is applied to the data, and the cluster centers are
        adjusted if `y` is provided during fitting [2]_.
    
        The mathematical formulation of the KMeans algorithm is as follows:
    
        .. math::
            \min_{C} \sum_{i=1}^{n} \min_{\mu_j \in C} \|x_i - \mu_j\|^2
    
        where :math:`\mu_j` are the cluster centers.
    
        Examples
        --------
        >>> from gofast.transformers.feature_engineering import KMeansFeaturizer
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        >>> featurizer = KMeansFeaturizer(n_clusters=3)
        >>> featurizer.fit(X, y)
        >>> print(featurizer.cluster_centers_)
    
        See Also
        --------
        sklearn.cluster.KMeans : 
            The underlying KMeans implementation used by this transformer.
        sklearn.decomposition.PCA : 
            Principal Component Analysis used for dimensionality reduction.
    
        References
        ----------
        .. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024. 
              K-Means Featurizer: A booster for intricate datasets. Earth Sci. 
              Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
              
        .. [2] MacQueen, J. (1967). "Some Methods for classification and Analysis
               of Multivariate Observations". Proceedings of 5th Berkeley 
               Symposium on Mathematical Statistics and Probability. 
               University of California Press. pp. 281–297.
        """
        # Validate input array X
        X = check_array(X, accept_sparse=True)
        if y is not None:
            # Validate target array y
            y= np.asarray (y).ravel() # for consistency
            X, y  = check_X_y(X, y, estimator =self )

        # Apply PCA if n_components is specified
        if self.n_components is not None:
            pca = PCA(n_components=self.n_components)
            X = pca.fit_transform(X)

        # Scale target and concatenate with X if y is provided
        if y is not None:
            self.y_ = np.asarray(y).copy()
            y_scaled = y[:, np.newaxis] * self.target_scale
            data_for_clustering = np.hstack((X, y_scaled))
        else:
            data_for_clustering = X

        # Fit the KMeans model on the data
        self.km_model_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm=self.algorithm,
            copy_x=self.copy_x,
            tol=self.tol,
            random_state=self.random_state, 
            verbose=self.verbose, 
        ).fit(data_for_clustering)

        # Adjust centroids if y was used during fit
        if y is not None:
            self.km_model_ = KMeans(
                n_clusters=self.n_clusters,
                init=self.km_model_.cluster_centers_[:, :-1],
                n_init=1,
                max_iter=1
            )
            self.km_model_.fit(X)
            
        # Store cluster centers
        self.cluster_centers_ = self.km_model_.cluster_centers_
        return self
    
    def transform(self, X):
        """
        Transform the data by encoding the closest cluster ID for each sample.
    
        The `transform` method predicts the closest cluster for each sample in the 
        provided dataset `X` and encodes the cluster assignments according to the 
        specified encoding strategy. The transformed data can include cluster assignments 
        as features using various encoding methods such as one-hot, bin-counting, label, 
        frequency, and mean target encoding.
    
        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            New data to transform. It can be a dense array or a sparse matrix.
    
        Returns
        -------
        X_transformed : array-like or sparse matrix of shape \
            (n_samples, n_features + n_encoded_features)
            The transformed data with additional features indicating the encoded 
            cluster assignments. The number of additional features depends on the 
            encoding strategy used.
    
        Notes
        -----
        The method checks if the KMeans model is fitted. It validates the input data
        and predicts the closest cluster for each sample. Based on the `encoding` 
        strategy, it transforms the cluster assignments into new features.
    
        Encoding strategies include:
        - `onehot`: One-hot encoding of the cluster assignments.
        - `bin-counting`: Probabilistic bin-counting encoding.
        - `label`: Label encoding of the cluster assignments.
        - `frequency`: Frequency encoding of the cluster assignments.
        - `mean_target`: Mean target encoding based on target values provided during fit.
    
        The mathematical formulation of the transformation is based on the cluster 
        assignments predicted by the KMeans algorithm:
    
        .. math::
            \text{Cluster}_i = \arg\min_{\mu_j \in C} \|x_i - \mu_j\|^2
    
        Examples
        --------
        >>> from gofast.transformers.feature_engineering import KMeansFeaturizer
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        >>> featurizer = KMeansFeaturizer(n_clusters=3, encoding='onehot')
        >>> featurizer.fit(X, y)
        >>> X_transformed = featurizer.transform(X)
        >>> print(X_transformed)
    
        See Also
        --------
        sklearn.cluster.KMeans : The underlying KMeans implementation used by this transformer.

        """
        # Check if the model is fitted
        check_is_fitted(self, 'km_model_')
        
        # Validate input array X
        X = check_array(X, accept_sparse=True)
        
        # Predict the closest cluster for each sample
        clusters = self.km_model_.predict(X)
    
        if self.encoding == 'bin-counting':
            # Bin-counting encoding
            n_clusters = len(set(clusters))
            cluster_counts = np.bincount(clusters, minlength=n_clusters)
            cluster_probabilities = cluster_counts / cluster_counts.sum()
            clusters_bin_count = np.zeros((X.shape[0], n_clusters))
            for idx, cluster_id in enumerate(clusters):
                clusters_bin_count[idx, cluster_id] = cluster_probabilities[cluster_id]
            if self.to_sparse:
                clusters_sparse = sparse.csr_matrix(clusters_bin_count)
                X_transformed = sparse.hstack((X, clusters_sparse), format='csr')
            else:
                X_transformed = np.hstack((X, clusters_bin_count))
        elif self.encoding == 'onehot':
            # One-hot encoding
            n_clusters = len(set(clusters))
            clusters_onehot = np.zeros((X.shape[0], n_clusters))
            for idx, cluster_id in enumerate(clusters):
                clusters_onehot[idx, cluster_id] = 1
            if self.to_sparse:
                clusters_sparse = sparse.csr_matrix(clusters_onehot)
                X_transformed = sparse.hstack((X, clusters_sparse), format='csr')
            else:
                X_transformed = np.hstack((X, clusters_onehot))
        elif self.encoding == 'frequency':
            # Frequency encoding
            n_clusters = len(set(clusters))
            cluster_counts = np.bincount(clusters, minlength=n_clusters)
            cluster_frequencies = cluster_counts / len(clusters)
            clusters_frequency = cluster_frequencies[clusters].reshape(-1, 1)
            if self.to_sparse:
                clusters_sparse = sparse.csr_matrix(clusters_frequency)
                X_transformed = sparse.hstack((X, clusters_sparse), format='csr')
            else:
                X_transformed = np.hstack((X, clusters_frequency))
        elif self.encoding == 'mean_target':
            # Mean target encoding
            if hasattr(self, 'y_'):
                n_clusters = len(set(clusters))
                target_means = np.zeros(n_clusters)
                for cluster in range(n_clusters):
                    target_means[cluster] = self.y_[clusters == cluster].mean()
                clusters_mean_target = target_means[clusters].reshape(-1, 1)
                if self.to_sparse:
                    clusters_sparse = sparse.csr_matrix(clusters_mean_target)
                    X_transformed = sparse.hstack((X, clusters_sparse), format='csr')
                else:
                    X_transformed = np.hstack((X, clusters_mean_target))
            else:
                raise ValueError(
                    "Mean target encoding requires target values provided during fit.")
        else:
            # Label encoding
            # Default strategy: just add the cluster labels as a new feature
            clusters_reshaped = clusters.reshape(-1, 1)
            if self.to_sparse:
                clusters_sparse = sparse.csr_matrix(clusters_reshaped)
                X_transformed = sparse.hstack((X, clusters_sparse), format='csr')
            else:
                X_transformed = np.hstack((X, clusters_reshaped))
    
        return X_transformed

class CategoricalEncoder2(BaseEstimator, TransformerMixin):
    """
    CategoricalEncoder is a transformer that encodes categorical variables in
    a DataFrame or numpy array using various encoding strategies such as label
    encoding, one-hot encoding, bin-counting, frequency encoding, and mean-target
    encoding.

    Parameters
    ----------
    encoding : {'label', 'onehot', 'bin-counting', 'frequency', 'mean_target'}, default='label'
        Encoding strategy for categorical variables:
        - 'label': Label encoding of the categorical variables.
        - 'onehot': One-hot encoding of the categorical variables.
        - 'bin-counting': Probabilistic bin-counting encoding.
        - 'frequency': Frequency encoding of the categorical variables.
        - 'mean_target': Mean target encoding based on target values provided during fit.
    target : array-like of shape (n_samples,), default=None
        Target values (class labels in classification, real numbers in regression).
        Required if `encoding='mean_target'`.

    Attributes
    ----------
    label_encoders_ : dict
        Dictionary of label encoders for each categorical variable.

    Examples
    --------
    >>> from gofast.transformers.feature_engineering import CategoricalEncoder
    >>> import pandas as pd
    >>> data = {'category': ['a', 'b', 'a', 'c'], 'value': [1, 2, 3, 4]}
    >>> df = pd.DataFrame(data)
    >>> encoder = CategoricalEncoder(encoding='onehot')
    >>> encoder.fit(df)
    >>> df_encoded = encoder.transform(df)
    >>> print(df_encoded)

    Notes
    -----
    The `fit` method detects categorical variables in the input data. If the input
    is a DataFrame, it identifies columns of type `object` or `category`. If the
    input is a numpy array, it identifies floating-point columns that should be
    considered as categorical.

    The mathematical formulation of encoding strategies are as follows:

    - Label Encoding:
        Assigns a unique integer to each category.
    - One-hot Encoding:
        .. math:: X_{ij} = \begin{cases} 1 & \text{if category is present}\\
            \\ 0 & \text{otherwise} \end{cases}
    - Bin-counting:
        Converts categories into probabilities based on their frequency.
    - Frequency Encoding:
        .. math:: X_{ij} = \frac{\text{count of category j}}{\text{total count}}
    - Mean-target Encoding:
        .. math:: X_{ij} = \frac{\sum y_j}{\text{count of category j}}

    See Also
    --------
    sklearn.preprocessing.LabelEncoder : Encode labels with value between 0 and n_classes-1.
    pandas.get_dummies : Convert categorical variable(s) into dummy/indicator variables.

    References
    ----------
    .. [1] Micci-Barreca, D. (2001). "A preprocessing scheme for high-cardinality
          categorical attributes in classification and prediction problems". 
          ACM SIGKDD Explorations Newsletter, 3(1), 27-32.
    """

    def __init__(self, encoding='label', target=None):
        self.encoding = encoding
        self.target = target
        
    def fit(self, X, y=None):
        """
        Fit the CategoricalEncoder to the data.
    
        The `fit` method detects categorical variables in the input data, creates 
        label encoders for each categorical variable, and prepares for encoding 
        based on the specified strategy. If the target variable `y` is provided, 
        it is stored for use in mean-target encoding.
    
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data containing categorical variables. If a DataFrame is passed,
            it identifies columns of type `object` or `category` for encoding. If a 
            numpy array is passed, it identifies floating-point columns that should 
            be considered as categorical.
        y : array-like of shape (n_samples,), default=None
            Target values (class labels in classification, real numbers in regression).
            Required if `encoding='mean_target'`.
    
        Returns
        -------
        self : object
            Returns the instance itself.
    
        Notes
        -----
        If the input `X` is a DataFrame, the method identifies columns with data types 
        `object` or `category` as categorical variables. It then creates a `LabelEncoder` 
        for each categorical variable to transform string or object categories into 
        integer codes.
    
        If the input `X` is a numpy array, it identifies columns with floating-point 
        data types as potential categorical variables and creates `LabelEncoder` 
        instances for these columns.
    
        If `y` is provided and `encoding='mean_target'`, the method stores `y` for 
        calculating the mean target values for each category during the transform step.
    
        The mathematical formulation for the encoding strategies include:
    
        - Label Encoding:
            Assigns a unique integer to each category.
        - One-hot Encoding:
            .. math:: X_{ij} = \begin{cases} 1 & \text{if category is present}\\
                \\ 0 & \text{otherwise} \end{cases}
        - Bin-counting:
            Converts categories into probabilities based on their frequency.
        - Frequency Encoding:
            .. math:: X_{ij} = \frac{\text{count of category j}}{\text{total count}}
        - Mean-target Encoding:
            .. math:: X_{ij} = \frac{\sum y_j}{\text{count of category j}}
    
        Examples
        --------
        >>> from gofast.transformers.feature_engineering import CategoricalEncoder2
        >>> import pandas as pd
        >>> data = {'category': ['a', 'b', 'a', 'c'], 'value': [1, 2, 3, 4]}
        >>> df = pd.DataFrame(data)
        >>> encoder = CategoricalEncoder2(encoding='mean_target')
        >>> target = [10, 20, 10, 30]
        >>> encoder.fit(df, target)
        >>> print(encoder.label_encoders_)
    
        See Also
        --------
        sklearn.preprocessing.LabelEncoder : 
            Encode labels with value between 0 and n_classes-1.
        pandas.get_dummies : 
            Convert categorical variable(s) into dummy/indicator variables.
 
        """
        self.label_encoders_ = {}
        if isinstance(X, pd.DataFrame):
            self._fit_dataframe(X, y)
        else:
            X= np.asarray (X)
            if y is not None: 
                y= np.asarray (y ) 
                check_consistent_length(X, y )
            self._fit_array(X, y)
        return self

    def _fit_dataframe(self, X, y=None):
        for col in X.select_dtypes(['object', 'category']).columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders_[col] = le

        if y is not None:
            self.target = y

    def _fit_array(self, X, y=None):
        float_columns = self._get_float_columns(X)
        for col in float_columns:
            le = LabelEncoder()
            le.fit(X[:, col])
            self.label_encoders_[col] = le

        if y is not None:
            self.target = y
            
    def transform(self, X):
        """
        Transform the data by encoding the categorical variables.
    
        The `transform` method encodes the categorical variables in the input data `X`
        based on the encoding strategy specified during the initialization of the 
        `CategoricalEncoder`. The transformed data includes encoded categorical variables
        and numerical data concatenated together.
    
        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            New data to transform. It can be a dense array or a DataFrame. If a DataFrame
            is passed, it applies encoding to the identified categorical columns. If a numpy
            array is passed, it applies encoding to the identified floating-point columns
            considered as categorical.
    
        Returns
        -------
        X_transformed : array-like or DataFrame of shape \
            (n_samples, n_features + n_encoded_features)
            The transformed data with additional features indicating the encoded 
            categorical variables. The number of additional features depends on the 
            encoding strategy used.
    
        Notes
        -----
        The method checks if the `CategoricalEncoder` has been fitted by ensuring that
        `label_encoders_` have been created. It validates the input data and applies
        the encoding strategy to the identified categorical variables.
    
        Encoding strategies include:
        - `label`: Label encoding of the categorical variables.
        - `onehot`: One-hot encoding of the categorical variables.
        - `bin-counting`: Probabilistic bin-counting encoding.
        - `frequency`: Frequency encoding of the categorical variables.
        - `mean_target`: Mean target encoding based on target values provided during fit.
    
        The mathematical formulation of the transformation is based on the encoding 
        strategy:
    
        - Label Encoding:
            Assigns a unique integer to each category.
        - One-hot Encoding:
            .. math:: X_{ij} = \begin{cases} 1 & \text{if category is present}\\
                \\ 0 & \text{otherwise} \end{cases}
        - Bin-counting:
            Converts categories into probabilities based on their frequency.
        - Frequency Encoding:
            .. math:: X_{ij} = \frac{\text{count of category j}}{\text{total count}}
        - Mean-target Encoding:
            .. math:: X_{ij} = \frac{\sum y_j}{\text{count of category j}}
    
        Examples
        --------
        >>> from gofast.transformers.feature_engineering import CategoricalEncoder
        >>> import pandas as pd
        >>> data = {'category': ['a', 'b', 'a', 'c'], 'value': [1, 2, 3, 4]}
        >>> df = pd.DataFrame(data)
        >>> encoder = CategoricalEncoder(encoding='onehot')
        >>> encoder.fit(df)
        >>> df_encoded = encoder.transform(df)
        >>> print(df_encoded)
    
        See Also
        --------
        sklearn.preprocessing.LabelEncoder : Encode labels with value between 0 and n_classes-1.
        pandas.get_dummies : Convert categorical variable(s) into dummy/indicator variables.
    
        """
        check_is_fitted(self, 'label_encoders_')
        if isinstance(X, pd.DataFrame):
            return self._transform_dataframe(X)
        else:
            X = np.asarray (X )
            return self._transform_array(X)

    def _transform_dataframe(self, X):
        X_encoded = X.copy()

        for col, le in self.label_encoders_.items():
            X_encoded[col] = le.transform(X_encoded[col])

        if self.encoding == 'onehot':
            X_encoded = pd.get_dummies(X_encoded, columns=self.label_encoders_.keys())
        elif self.encoding == 'bin-counting':
            X_encoded = self._bin_counting_encode(X_encoded)
        elif self.encoding == 'frequency':
            X_encoded = self._frequency_encode(X_encoded)
        elif self.encoding == 'mean_target':
            if self.target is not None:
                X_encoded = self._mean_target_encode(X_encoded)
            else:
                raise ValueError(
                    "Mean target encoding requires target values provided"
                    " during fit.")

        return X_encoded

    def _transform_array(self, X):
        float_columns = self._get_float_columns(X)
        X_encoded = X.copy()

        for col in float_columns:
            X_encoded[:, col] = self.label_encoders_[col].transform(X[:, col])

        if self.encoding == 'onehot':
            X_encoded = self._onehot_encode_array(X_encoded, float_columns)
        elif self.encoding == 'bin-counting':
            X_encoded = self._bin_counting_encode_array(X_encoded, float_columns)
        elif self.encoding == 'frequency':
            X_encoded = self._frequency_encode_array(X_encoded, float_columns)
        elif self.encoding == 'mean_target':
            if self.target is not None:
                X_encoded = self._mean_target_encode_array(X_encoded, float_columns)
            else:
                raise ValueError("Mean target encoding requires"
                                 " target values provided during fit.")

        return X_encoded

    def _get_float_columns(self, X):
        return [i for i in range(X.shape[1]) if np.issubdtype(X[:, i].dtype, np.floating)]

    def _onehot_encode_array(self, X, float_columns):
        return pd.get_dummies(pd.DataFrame(X), columns=float_columns).values

    def _bin_counting_encode(self, X):
        for col in self.label_encoders_.keys():
            col_dummies = pd.get_dummies(X[col], prefix=col)
            col_prob = col_dummies.div(col_dummies.sum(axis=1), axis=0)
            X = pd.concat([X.drop(columns=[col]), col_prob], axis=1)
        return X

    def _bin_counting_encode_array(self, X, float_columns):
        X_df = pd.DataFrame(X)
        for col in float_columns:
            col_dummies = pd.get_dummies(X_df[col], prefix=col)
            col_prob = col_dummies.div(col_dummies.sum(axis=1), axis=0)
            X_df = pd.concat([X_df.drop(columns=[col]), col_prob], axis=1)
        return X_df.values

    def _frequency_encode(self, X):
        for col in self.label_encoders_.keys():
            freq = X[col].value_counts(normalize=True)
            X[col] = X[col].map(freq)
        return X

    def _frequency_encode_array(self, X, float_columns):
        X_df = pd.DataFrame(X)
        for col in float_columns:
            freq = X_df[col].value_counts(normalize=True)
            X_df[col] = X_df[col].map(freq)
        return X_df.values

    def _mean_target_encode(self, X):
        for col in self.label_encoders_.keys():
            means = X.groupby(col)[self.target.name].mean()
            X[col] = X[col].map(means)
        return X

    def _mean_target_encode_array(self, X, float_columns):
        X_df = pd.DataFrame(X)
        for col in float_columns:
            means = X_df.groupby(col)[self.target].mean()
            X_df[col] = X_df[col].map(means)
        return X_df.values

class StratifyFromBaseFeature(BaseEstimator, TransformerMixin):
    """
    Stratifies a dataset by categorizing a numerical attribute and returns 
    stratified training and testing sets.
    
    Useful for datasets with limited data.

    Parameters:
    ----------
    base_feature : str, optional
        Numerical feature to be categorized for stratification.

    threshold_operator : float, default=1.0
        Coefficient to normalize the numerical feature for 
        categorization.

    max_category : int, default=3
        Maximum category value. Values greater than this are grouped 
        into max_category.

    return_train : bool, default=False
        If True, returns the whole stratified training set.

    n_splits : int, default=1
        Number of re-shuffling & splitting iterations.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.

    random_state : int, default=42
        Controls the randomness of the dataset splitting.

    Attributes:
    ----------
    statistics_ : DataFrame or None
        Statistics about the categorization and stratification process.
    base_class_: list 
       List composed of the base feature class labels. 
       
    Example
    --------
    >>> from gofast.transformers import StratifyFromBaseFeature
    >>> st= StratifyFromBaseFeature('flow') 
    >>> a, b = st.fit_transform(data)
    >>> st.statistics_
    Out[17]: 
                 Overall (total)    Random  ...  Rand. error (%)  Strat. error (%)
    class_label                             ...                                   
    1.0                 0.320186  0.310345  ...        -3.073463          0.516408
    0.0                 0.354988  0.356322  ...         0.375629          0.375629
    3.0                 0.141531  0.183908  ...        29.941587         -2.543810
    2.0                 0.183295  0.149425  ...       -18.478103          0.334643

    [4 rows x 5 columns]
    Notes:
    ------
    The `statistics_` attribute helps evaluate the distribution of the newly
    added category in different splits and assess the effectiveness of 
    stratification.
    """

    def __init__(self, base_feature=None, threshold_operator=1.0,
                 max_category=3, return_train=False, n_splits=1,
                 test_size=0.2, random_state=42):
        self.logger = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.base_feature = base_feature
        self.threshold_operator = threshold_operator
        self.max_category = max_category
        self.return_train = return_train
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fits the transformer to X for sklearn's Transformer API compatibility.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present only for compatibility with sklearn's transformer
            interface.

        Returns:
        --------
        self : object
            Returns self for method chaining.
            """
        return self

    def transform(self, X):
        """
        Transforms the dataset and stratifies based on the categorized 
        attribute.

        Parameters:
        ----------
        X : DataFrame
            The input DataFrame to be stratified.

        Returns:
        -------
        tuple of DataFrames
            Stratified training and testing sets.
        """
        train_set, test_set = train_test_split(X, test_size=self.test_size,
                                               random_state=self.random_state)
        if self.base_feature is None:
            self.logger.info('Base numerical feature not provided. '
                             'Using random sampling.')
            return train_test_split(X, test_size=self.test_size,
                                    random_state=self.random_state)

        X = self._categorize_feature(X)
        # split after categorizing the base feature 
        train_set, test_set = train_test_split(
            X, test_size=self.test_size,random_state=self.random_state)
        strat_train_set, strat_test_set = self._stratify_dataset(X)
        self._calculate_statistics(X, test_set, strat_test_set) 
        self._cleanup_temp_columns(X, strat_train_set, strat_test_set)

        return (strat_train_set, strat_test_set
                ) if not self.return_train else (X, strat_test_set)

    def _categorize_feature(self, X):
        """Categorizes the numerical feature."""
        # Implement logic to categorize 'base_feature' here
        from ..tools.mlutils import discretize_categories
        X = discretize_categories(X, in_cat=self.base_feature, 
            new_cat="class_label", divby =self.threshold_operator,
            higherclass = self.max_category
             )
        self.base_class_=  list(X["class_label"].value_counts().index.values)
        return X

    def _stratify_dataset(self, X):
        """Performs stratification of the dataset."""
        split = StratifiedShuffleSplit(n_splits=self.n_splits,
                                       test_size=self.test_size,
                                       random_state=self.random_state)
        for train_index, test_index in split.split(X, X['class_label']):
            return X.loc[train_index], X.loc[test_index]

    def _calculate_statistics(self, X, *test_dataframes ):
        """Calculates statistics for the stratification process.
        
        Parameters 
        -----------
        X : DataFrame
            The input DataFrame to be stratified.
        *test_dataframes: DataFrames 
           Test data before and after the stratification.  
        """
        test_set, strat_test_set = test_dataframes 
        # Implement logic to calculate statistics here
        random_stats= test_set["class_label"].value_counts()/ len(test_set)
        stratified_stats = strat_test_set["class_label"].value_counts()/len(
            strat_test_set) 
        total = X["class_label"].value_counts() /len(X)
        stats = {
             "class_label":  np.array (self.base_class_), 
             "Overall (total)":total, 
             "Random": random_stats, 
             "Stratified": stratified_stats,  
             "Rand. error (%)":  (random_stats /total -1) *100, 
             "Strat. error (%)":  (stratified_stats /total -1) *100, 
         }
        self.statistics_ = pd.DataFrame ( stats )
        # set a pandas dataframe for inspections attributes `statistics`.
        self.statistics_.set_index("class_label", inplace=True)
        
    def _cleanup_temp_columns(self, X, *dataframes):
        """Removes temporary columns used for stratification."""
        temp_columns = ["class_label"]  # Add any temporary columns used
        for df in dataframes:
            df.drop(temp_columns, axis=1, inplace=True)
        X.drop(temp_columns, axis=1, inplace=True)
        
class CategoryBaseStratifier(BaseEstimator, TransformerMixin):
    """
    Stratifies a dataset based on a specified base category 
    
    It is more representative splits into training and testing sets,
    especially useful when data is limited.

    Parameters:
    -----------
    base_column : str or int, optional
        The name or index of the column to be used for stratification. If None, 
        returns purely random sampling.

    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.

    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split.

    Attributes:
    ----------
    statistics_ : DataFrame or None
        Statistics about the stratification process, including the distribution
        of the base category in the full dataset, the random test set, and the 
        stratified test set.

    Examples:
    --------
    >>> from gofast.datasets import load_bagoue 
    >>> from gofast.transformers import CategoryBaseStratifier
    >>> X = load_bagoue ( as_frame=True) 
    >>> cobj = CategoryBaseStratifier(base_column='shape')
    >>> Xtrain_stratified, X_test_stratified= cobj.fit_transform ( X ) 
    >>> X_test_stratified.head(2) 
    Out[67]: 
           num  name      east  ...    lwi                    geol  flow
    121  122.0  b122  756473.0  ...  32.54  VOLCANO-SEDIM. SCHISTS   1.0
    207  208.0  b208  792785.0  ...  29.80  VOLCANO-SEDIM. SCHISTS   1.0

    [2 rows x 13 columns]
    Notes:
    ------
    The `statistics_` attribute provides insights into the distribution of the 
    base category across different splits, helping evaluate the effectiveness 
    of the stratification process.
    """
    
    def __init__(self, base_column=None, test_size=0.2, random_state=42):
        self.logger = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.base_column = base_column
        self.test_size = test_size
        self.random_state = random_state


    def fit(self, X, y=None):
        """Does nothing, exists for compatibility with sklearn's Transformer API."""
        return self
    
    def transform(self, X):
        """
        Splits the dataset into training and testing sets using stratification.

        Parameters:
        -----------
        X : DataFrame
            The input DataFrame to split.

        Returns:
        --------
        tuple of DataFrames
            The training and testing sets after stratification.
        """
        if self.base_column is None:
            self.logger.debug('Base column not provided. Using purely random sampling.')
            return train_test_split(X, test_size=self.test_size,
                                    random_state=self.random_state)

        if isinstance(self.base_column, int):
            self.base_column = X.columns[int(self.base_column)]
        elif isinstance(self.base_column, str) and self.base_column not in X.columns:
            self.logger.warning(
                f'Base column "{self.base_column}" not found in DataFrame columns.')
            return train_test_split(
                X, test_size=self.test_size, random_state=self.random_state)

        strat_train_set, strat_test_set = self._stratify(X)
        self._calculate_statistics(X, strat_test_set)
        return strat_train_set, strat_test_set

    def _stratify(self, X):
        """
        Internal method to perform the stratification.

        Parameters:
        -----------
        X : DataFrame
            The input DataFrame to stratify.

        Returns:
        --------
        tuple of DataFrames
            The stratified training and testing sets.
        """
        from ..tools.mlutils import stratify_categories 
        # stratification logic here (e.g., using pd.cut)
        strat_train_set, strat_test_set = stratify_categories(
            X, self.base_column, test_size = self.test_size, 
            random_state = self.random_state )
        return strat_train_set, strat_test_set

    def _calculate_statistics(self, X, strat_test_set):
        """
        Calculates statistics related to the stratification process.

        Parameters:
        -----------
        X : DataFrame
            The original input DataFrame.

        strat_test_set : DataFrame
            The stratified testing set.
        """
        overall_distribution = X[self.base_column].value_counts() / len(X)
        stratified_distribution = strat_test_set[self.base_column].value_counts(
            ) / len(strat_test_set)
        error = ((stratified_distribution / overall_distribution) - 1) * 100

        self.statistics_ = pd.DataFrame({
            'Overall': overall_distribution,
            'Stratified': stratified_distribution,
            'Strat. %error': error
        })


class CategorizeFeatures(BaseEstimator, TransformerMixin ): 
    """ Transform numerical features into categorical features and return 
    a new array transformed. 
    
    Parameters  
    ------------
    columns: list,
       List of the columns to encode the labels 
       
    func: callable, 
       Function to apply the label accordingly. Label must be included in 
       the columns values.
       
    categories: dict, Optional 
       Dictionnary of column names(`key`) and labels (`values`) to 
       map the labels.  
       
    get_dummies: bool, default=False 
      returns a new encoded DataFrame  with binary columns 
      for each category within the specified categorical columns.

    parse_cols: bool, default=False
      If `columns` parameter is listed as string, `parse_cols` can defaultly 
      constructs an iterable objects. 
    
    return_cat_codes: bool, default=False 
       return the categorical codes that used for mapping variables. 
       if `func` is applied, mapper returns an empty dict. 
    
    Examples
    --------
    >>> from gofast.datasets import make_mining_ops 
    >>> from gofast.transformers import CategorizeFeatures
    >>> X = make_mining_ops (samples =20, as_frame =True ) 
    >>> cf = CategorizeFeatures (columns =['OreType', 'EquipmentType']) 
    >>> Xtransformed = cf.fit_transform (X)
    >>> Xtransformed.head(7) 
       OreType  EquipmentType
    0        1              3
    1        0              3
    2        1              3
    3        1              3
    4        0              0
    5        1              0
    6        2              2
    >>> cf.cat_codes_
    {'OreType': {1: 'Type2', 0: 'Type1', 2: 'Type3'},
     'EquipmentType': {3: 'Truck', 0: 'Drill', 2: 'Loader', 1: 'Excavator'}}
    """
    
    def __init__(
        self, 
        columns: list =None, 
        func: _F=None, 
        categories: dict=None, 
        get_dummies:bool=..., 
        parse_cols:bool =..., 
        ): 
        self._logging= gofastlog().get_gofast_logger(self.__class__.__name__)
        self.columns =columns 
        self.func=func
        self.categories=categories 
        self.get_dummies =get_dummies 
        self.parse_cols=parse_cols 
 
    def fit(self, X, y=None):
        """ 
        Parameters 
        ----------
        
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            y is passed for API purpose. It does nothing.  
            
        Returns 
        --------
        ``Self`: Instanced object for methods chaining 
        
        """
        return self
    
    def transform(self, X) :
        """ 
        Transforms `X` by applying the specified operation.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.
        
        """
        from ..tools.mlutils import soft_encoder 
        # -------------------------------------------
        if _is_arraylike_1d(X): 
            raise ValueError ("One-dimensional or Series is not allowed."
                              " Use sklearn.preprocessing.LabelEncoder or "
                              " gofast.tools.smart_label_classier to encode"
                              " variables.")
        X = build_data_if(X, to_frame =True, force =True,
                          raise_warning="silence", input_name='col')
        X, self.num_columns_, self.cat_columns_ = to_numeric_dtypes(
            X, return_feature_types=True
            )
        X, self.cat_codes_ = soft_encoder( 
            X ,
            columns = self.columns, 
            func =self.func, 
            categories=self.categories, 
            get_dummies=self.get_dummies,
            return_cat_codes=True
            )
        
        return X 
    

class AttributesCombinator(BaseEstimator, TransformerMixin):
    """
    Combine attributes using operators, indexes, or names.

    Create a new attribute by performing operations on selected features using
    their indexes, literal string operators, or names. This transformer is useful
    for creating new features that are combinations of existing features, which
    might improve the performance of machine learning models.

    Parameters:
    -----------
    attribute_names : list of str or str, optional
        List of feature names for combination, or a string with operator symbols.
        Decides how to combine new feature values based on the `operator` parameter.
        Example: attribute_names=['feature1', 'feature2'].

    attribute_indexes : list of int, optional
        Indexes of each feature for combining. Raises an error if any index does
        not match the dataframe or array columns. Example: attribute_indexes=[0, 1].

    operator : str, default='/' 
        Type of operation to perform. Can be one of ['/','+', '-', '*', '%'].

    Attributes:
    ----------
    _operators : dict
        A mapping of string operator symbols to NumPy functions for efficient
        computation of the desired operation.

    Methods:
    --------
    fit(X, y=None):
        Fits the transformer to `X`. In this transformer, `fit` does not perform
        any operation and returns self.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            This parameter is ignored and present only for compatibility with
            the sklearn transformer interface.

    transform(X):
        Transforms `X` by applying the specified operation to the selected
        features and returns the transformed data.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.

    Examples:
    --------
    >>> import pandas as pd
    >>> from gofast.transformers import AttributesCombinator
    >>> from gofast.datasets.dload import load_bagoue
    >>> X, y = load_bagoue(as_frame=True)
    """
    _operators = {
        '/': np.divide,
        '+': np.add,
        '-': np.subtract,
        '*': np.multiply,
        '%': np.mod,
    }

    def __init__(self, attribute_names=None, attribute_indexes=None,
                 operator='/'):
        self.attribute_names = attribute_names
        self.attribute_indexes = attribute_indexes
        self.operator = operator
        self._validate_operator()

    def fit(self, X, y=None):
        """
        Fits the transformer to `X`.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : Ignored
            Not used, present only for compatibility with sklearn's transformer
            interface.

        Returns:
        --------
        self : object
            Returns self for method chaining.
        """
        return self

    def transform(self, X):
        """
        Transforms `X` by applying the specified operation.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples to transform.

        Returns:
        --------
        X_transformed : array-like of shape (n_samples, n_features + 1)
            The transformed array with the additional combined attribute.
        """
        if isinstance(X, pd.DataFrame):
            X, columns = self._validate_dataframe(X)
        else:
            columns = None

        if self.attribute_indexes:
            X = self._apply_operation(X)
            columns = columns + [self._new_feature_name(columns)] if columns else None

        return pd.DataFrame(X, columns=columns) if columns else X

    def _validate_operator(self):
        """
        Validates the operator.

        Raises:
        -------
        ValueError: If the operator is not recognized.
        """
        if self.operator not in self._operators:
            raise ValueError(f"Invalid operator '{self.operator}'. Valid"
                             f" operators are {list(self._operators.keys())}.")

    def _validate_dataframe(self, X):
        """
        Validates the DataFrame and extracts column names and indexes.

        Parameters:
        -----------
        X : DataFrame
            The DataFrame to validate.

        Returns:
        --------
        X_values : np.ndarray
            The values of the DataFrame.

        columns : list
            The column names of the DataFrame.
        """
        columns = X.columns.tolist()
        if self.attribute_names:
            if not all(name in columns for name in self.attribute_names):
                raise ValueError("Some attribute names are not in the DataFrame columns.")
            self.attribute_indexes = [columns.index(name) for name in self.attribute_names]
        return X.values, columns

    def _apply_operation(self, X):
        """
        Applies the specified operation to the selected features.

        Parameters:
        -----------
        X : np.ndarray
            The array to which the operation will be applied.

        Returns:
        --------
        X_transformed : np.ndarray
            The transformed array.
        """
        operation = self._operators[self.operator]
        result = operation(X[:, self.attribute_indexes[0]], X[:, self.attribute_indexes[1]])

        for index in self.attribute_indexes[2:]:
            result = operation(result, X[:, index])

        return np.c_[X, result]

    def _new_feature_name(self, columns):
        """
        Generates a new feature name based on the operation and involved columns.

        Parameters:
        -----------
        columns : list
            The list of column names.

        Returns:
        --------
        new_feature_name : str
            The generated feature name.
        """
        return f'{"_".join([columns[i] for i in self.attribute_indexes])}_{self.operator}'


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Combine attributes using literal string operators, indexes, or names.

    Create a new attribute by performing operations on selected features using
    either their indexes, literal string operators, or names. This class inherits
    from scikit-learn's `BaseEstimator` and `TransformerMixin` classes.

    Parameters:
    -----------
    attribute_names : list of str, optional
        List of feature names for combination. Decides how to combine new feature
        values based on the `operator` parameter. By default, it performs a ratio
        of the given attributes. For example, if `attribute_names=['lwi', 'ohmS']`,
        it will divide the 'lwi' feature by 'ohmS'.

    attribute_indexes : list of int, optional
        Indexes of each feature for combining. Raises a warning if any index does
        not match the dataframe or array columns.

    operator : str, default='/' 
        Type of operation to perform. Can be one of ['/','+', '-', '*', '%'].

    Returns:
    --------
    X : np.ndarray
        A new array containing the result of the operations specified by
        `attribute_names`, `attribute_indexes`, and `operator`. If both `attribute_names`
        and `attribute_indexes` are `None`, it will return the original array.

    Notes:
    ------
    Literal string operators can be used for operations. For example, dividing two
    numerical features can be represented as "per" separated by underscores, like
    "_per_". For instance, to create a new feature based on the division of the
    features 'lwi' and 'ohmS', you can use the `attribute_names` parameter as:
    attribute_names='lwi_per_ohmS'.

    The same literal string format is valid for other operations such as multiplication
    (_mul_), subtraction (_sub_), modulo (_mod_), and addition (_add_). Alternatively,
    you can use the indexes of features for combining by providing the `attribute_indexes`
    parameter. Multiple operations can be set by passing a list of literal string
    operators in `attribute_indexes`.

    Examples:
    --------
    >>> import pandas as pd
    >>> from gofast.transformers import CombinedAttributesAdder
    >>> from gofast.datasets.dload import load_bagoue
    >>> X, y = load_bagoue(as_frame=True)
    >>> cobj = CombinedAttributesAdder(attribute_names='lwi_per_ohmS')
    >>> Xadded = cobj.fit_transform(X)
    >>> cobj.attribute_names_
    ... ['num',
         'name',
         'east',
         'north',
         'power',
         'magnitude',
         'shape',
         'type',
         'sfi',
         'ohmS',
         'lwi',
         'geol',
         'lwi_div_ohmS']  # new attribute 'lwi_div_ohmS'
    >>> df0 = pd.DataFrame(Xadded, columns=cobj.attribute_names_)
    >>> df0['lwi_div_ohmS']
    ... 0           0.0
        1      0.000002
        2      0.000005
        3      0.000004
        4      0.000008
          
        426    0.453359
        427    0.382985
        428    0.476676
        429    0.457371
        430    0.379429
        Name: lwi_div_ohmS, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder(attribute_names=['lwi', 'ohmS', 'power'], operator='+')
    >>> df0 = pd.DataFrame(cobj.fit_transform(X), columns=cobj.attribute_names_)
    >>> df0.iloc[:, -1]
    ... 0      1777.165142
        1      1207.551531
        2         850.5625
        3      1051.943553
        4       844.095833
            
        426      1708.8585
        427      1705.5375
        428      1568.9825
        429     1570.15625
        430      1666.9185
        Name: lwi_add_ohmS_add_power, Length: 431, dtype: object
    >>> cobj = CombinedAttributesAdder(attribute_indexes=[1, 6], operator='+')
    >>> df0 = pd.DataFrame(cobj.fit_transform(X), columns=cobj.attribute_names_)
    >>> df0.iloc[:, -1]
    ... 0        b1W
        1        b2V
        2        b3V
        3        b4W
        4        b5W
         
        426    b427W
        427    b428V
        428    b429V
        429    b430V
        430    b431V
        Name: name_add_shape, Length: 431, dtype: object
    """
    _op = {
        'times': ('times', 'prod', 'mul', '*', 'x'),
        'add': ('add', '+', 'plus'),
        'div': ('quot', '/', 'div', 'per'),
        'sub': ('sub', '-', 'less'),
        'mod': ('mod', '%'),
    }

    def __init__(
            self,
            attribute_names=None,
            attribute_indexes=None,
            operator='/', ):
        self.attribute_names = attribute_names
        self.attribute_indexes = attribute_indexes
        self.operator = operator
        self.attribute_names_ = None

    def fit(self, X, y=None):
        """
        Fit the `CombinedAttributesAdder` transformer to the input data `X`.

        Parameters:
        -----------
        X : ndarray (M x N matrix where M=m-samples, N=n-features)
            Training set; denotes data that is observed at training and
            prediction time, used as independent variables in learning.

        y : array-like, shape (M,), default=None
            Train target; denotes data that may be observed at training time
            as the dependent variable in learning, but which is unavailable
            at prediction time and is usually the target of prediction.

        Returns:
        --------
        self : object
            Returns self for easy method chaining.

        """
        return self

    def transform(self, X):
        """
        Transform data and return an array with the combined attributes.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns:
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features + 1)
            Transformed array, where n_samples is the number of samples and
            n_features is the number of features, with the additional combined attribute.

        """
        columns = []
        self.operator = self._get_operator(
            self.operator or self.attribute_names)

        if self.operator is None:
            warnings.warn("None or invalid operator cannot be used for attribute combinations.")

        if isinstance(self.attribute_names, str):
            self.attribute_names_ = parse_attrs(self.attribute_names)

        elif isinstance(self.attribute_names, (list, tuple, np.ndarray)):
            self.attribute_names_ = self.attribute_names

        if isinstance(X, pd.DataFrame):
            # Check if attributes exist in the DataFrame
            if self.attribute_names_:
                validate_feature(X, self.attribute_names_)
            # Get the index of attributes from the DataFrame
            if self.attribute_names_:
                self.attribute_indexes = list(map(
                    lambda o: list(X.columns).index(o), self.attribute_names_)
                )

            elif self.attribute_indexes:
                try:
                    self.attribute_names_ = list(map(
                        lambda ix: list(X.columns)[ix], self.attribute_indexes)
                    )
                except IndexError:
                    raise IndexError("List of indexes is out of range.")

            columns = X.columns
            X = to_numeric_dtypes(X)
            X = X.values

        if self.attribute_indexes:
            X = self._operate(X)

        if self.attribute_names_ is not None:
            self.attribute_names_ = list(columns) + ([
                f'_{self.operator}_'.join([v for v in self.attribute_names_])
            ] if self._isfine else [])
            
        try: 
            X= pd.DataFrame ( X, columns = self.attribute_names_)
        except : pass 
    
        return X

    def _get_operator(self, operator):
        """ Get operator for combining attributes """
        for k, v in self._op.items():
            for o in v:
                if operator.find(o) >= 0:
                    self.operator = k
                    return self.operator
        return

    def _operate(self, X):
        """ Perform operations based on indexes """
        def weird_division(ix_):
            """ Replace 0. value with 1 in denominator for division calculations """
            return ix_ if ix_ != 0. else 1

        msg = ("Unsupported operand type(s)! Index provided {} doesn't match "
               "any numerical features. Combined attribute creation is not possible.")

        self._isfine = True
        Xc = X[:, self.attribute_indexes]
        cb = Xc[:, 0]
        Xc = Xc[:, 1:]

        for k in range(Xc.shape[1]):
            try:
                if self.operator == 'mod':
                    cb %= Xc[:, k]
                if self.operator == 'add':
                    cb += Xc[:, k]
                if self.operator == 'sub':
                    cb -= Xc[:, k]
                if self.operator == 'div':
                    # If the denominator contains NaN or 0, a weird division is triggered
                    # and replaces the denominator by 1
                    try:
                        cb /= Xc[:, k]
                    except ZeroDivisionError:
                        wv = np.array(
                            list(map(weird_division, Xc[:, k])))
                        cb /= wv

                    except (TypeError, RuntimeError, RuntimeWarning):
                        warnings.warn(msg.format(
                            self.attribute_indexes[1:][k]))
                if self.operator == 'x':
                    cb *= Xc[:, k]
            except:
                warnings.warn(msg.format(self.attribute_indexes[1:][k]))
                self._isfine = False

        X = np.c_[X, cb] if self._isfine else X

        return X


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select specific attributes from a DataFrame for column transformation.

    The `DataFrameSelector` transformer is used to select specific columns,
    either numerical or categorical, from a DataFrame for further data 
    transformation, similar to scikit-learn's `make_column_transformer`.

    Parameters:
    -----------
    columns : list or array-like, default=None
        List of column names to keep in the data. If None, no columns 
        are selected.

    select_type : {'number', 'category'}, default=None
        Automatically select numerical or categorical columns based on the
        specified type. If 'number', only numerical columns will be selected;
        if 'category', only categorical columns
        will be selected. If None, no automatic selection is performed.

    parse_cols : bool, default=False
        If True, enables column name parsing to handle special characters 
        and spaces.

    return_frame : bool, default=False
        If True, returns a DataFrame after selecting columns; if False,
        returns a numpy array.

    Attributes:
    -----------
    features_ : list
        The final list of column names to be selected after considering 
        the input 'columns' and 'select_type'.

    num_features_ : list
        List of column names that are considered numerical after selection.

    cat_features_ : list
        List of column names that are considered categorical after selection.

    Returns:
    --------
    X : ndarray or DataFrame
        A new array or DataFrame composed of data from selected `columns`.

    Examples:
    ---------
    >>> from gofast.transformers import DataFrameSelector
    >>> from gofast.datasets import make_african_demo
    >>> af_data = make_african_demo(start_year='2021', end_year='2022').frame
    >>> dfr = DataFrameSelector(columns=None, select_type='number')
    >>> af_transf = dfr.fit_transform(af_data)
    >>> af_transf[:3, :]
    Out[14]: 
    array([[2.02100000e+03, 1.12942759e+08, 4.43867835e+01, 1.98222121e+01,
            7.48855579e+01, 2.83628640e+03],
           [2.02100000e+03, 1.12834803e+08, 2.73762471e+01, 6.04337242e+00,
            5.35163931e+01, 5.93341830e+03],
           [2.02100000e+03, 1.86877893e+08, 2.53328024e+01, 5.08724291e+00,
            8.10308039e+01, 9.77191812e+03]])
    """
    def __init__(
        self, 
        columns:list=None, 
        select_type:str=None, 
        parse_cols:bool=..., 
        return_frame:bool=...
        ):
        self.columns = columns 
        self.select_type = select_type
        self.parse_cols = parse_cols
        self.return_frame = return_frame

    def fit(self, X, y=None):
        """
        Fit the `DataFrameSelector` transformer to the input data `X`.

        Parameters:
        -----------
        X : ndarray (M x N matrix where M=m-samples, N=n-features)
            Training set; denotes data that is observed at training and
            prediction time, used as independent variables in learning.

        y : array-like, shape (M,), default=None
            Train target; denotes data that may be observed at training time
            as the dependent variable in learning, but which is unavailable
            at prediction time and is usually the target of prediction.

        Returns:
        --------
        self : object
            Returns self for easy method chaining.

        """
        self.parse_cols, self.return_frame = ellipsis2false(
            self.parse_cols, self.return_frame)
        self.num_features_ = []
        self.cat_features_ = []
        return self

    def transform(self, X):
        """
        Transform data and return numerical or categorical values.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        Returns:
        --------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Transformed array, where n_samples is the number of samples and
            n_features is the number of features.

        """
        is_frame(X, df_only=True, raise_exception=True,
                 objname="DataFrameSelector")
        
        if self.columns is not None:
            self.columns = is_iterable(
                self.columns, exclude_string=True, transform=True,
                parse_string=self.parse_cols)

        if self.columns is None and self.select_type is None:
            raise TypeError(
                "Either 'columns' or 'select_type' needs to be supplied.")

        if self.columns is not None:
            validate_feature(X, self.columns)
            X = X[self.columns]

        if self.select_type is not None:
            if str(self.select_type).lower().strip(
                    ) not in "numbernumericalcategorycategories":
                emsg = ("Support only 'number' or 'category'."
                        f" Got {self.select_type!r}")
                if self.columns is None:
                     raise ValueError(emsg) 
                else: warnings.warn(emsg)
                
            _, num_feats, cat_feats = to_numeric_dtypes(
                X, return_feature_types=True)

            if str(self.select_type).lower().strip().find('num') >= 0:
                self.columns = num_feats
            elif str(self.select_type).lower().strip().find('cat') >= 0:
                self.columns = cat_feats
                
        # For consistency, shrunk X
        self.features_ = list(X.columns)
        X = X[self.columns]

        # Update the numeric and categorical features
        _, self.num_features_, self.cat_features_ = to_numeric_dtypes(
            X, return_feature_types=True)

        return X if self.return_frame else np.array(X)

class FrameUnionFlex(BaseEstimator, TransformerMixin):
    """
    FrameUnionFlex combines numerical and categorical data preprocessing
    into a single transformer. It automates the process of imputing missing
    values, scaling numerical features, and encoding categorical features,
    simplifying the creation of machine learning pipelines. This transformer
    is highly configurable, allowing for detailed specification or automatic
    detection of feature types, as well as a choice of imputation strategies,
    scaling methods, and encoding techniques.

    Parameters
    ----------
    num_attributes : list of str, optional
        Specifies the column names to be treated as numerical attributes.
        If None, numerical attributes will be automatically identified in
        the dataset based on their data type.
    cat_attributes : list of str, optional
        Specifies the column names to be treated as categorical attributes.
        If None, categorical attributes will be automatically identified in
        the dataset based on their data type.
    scale : bool, default=True
        If True, applies scaling to numerical attributes using the scaling
        method defined by `scale_mode`. This is crucial for models that are
        sensitive to feature magnitude.
    impute_data : bool, default=True
        If True, imputes missing values in numerical attributes using the
        strategy defined by `strategy`. This helps in handling datasets with
        incomplete data.
    encode : bool, default=True
        If True, encodes categorical attributes using the method defined by
        `encode_mode`. Encoding is essential for converting categorical data
        into a format that can be provided to machine learning models.
    strategy : str, default='median'
        Defines the strategy used for imputing missing values. Options include
        'mean', 'median', and 'most_frequent'.
    scale_mode : str, default='StandardScaler'
        Defines the method used for scaling numerical attributes. Options are
        'StandardScaler' for z-score normalization and 'MinMaxScaler' for
        min-max normalization. 'RobustScaler' is also available for scaling
        features using statistics that are robust to outliers.
    encode_mode : str, default='OrdinalEncoder'
        Defines the method used for encoding categorical attributes. Options
        are 'OrdinalEncoder' for ordinal encoding and 'OneHotEncoder' for
        one-hot encoding.

    Attributes
    ----------
    num_attributes_ : list
        Auto-detected or specified names of numerical attributes in the data.
    cat_attributes_ : list
        Auto-detected or specified names of categorical attributes in the data.
    attributes_ : list
        Combined list of all numerical and categorical attributes in the data.
    X_ : ndarray of shape (n_samples, n_features)
        The transformed dataset containing processed numerical and encoded
        categorical features.

    Notes
    -----
    - FrameUnionFlex is designed to be flexible and efficient, automatically
      adapting to the provided dataset while allowing for user overrides.
    - It supports handling datasets with a mix of numerical and categorical
      data, preparing them for machine learning models that require numerical
      input.
    - Users are encouraged to explicitly define `num_attributes` and
      `cat_attributes` for better control and clarity in preprocessing.

    Examples
    --------
    >>> from sklearn.datasets import fetch_california_housing
    >>> from gofast.transformers import FrameUnionFlex
    >>> data = fetch_california_housing(as_frame=True)
    >>> X = data.frame.drop('MedHouseVal', axis=1)
    >>> transformer = FrameUnionFlex(scale=True, impute_data=True,
    ...                              encode=True, scale_mode='MinMaxScaler',
    ...                              encode_mode='OneHotEncoder')
    >>> X_transformed = transformer.fit_transform(X)
    >>> print(X_transformed.shape)
    
    This example demonstrates using FrameUnionFlex to preprocess the
    California housing dataset, applying min-max scaling to numerical
    features and one-hot encoding to categorical features. The transformed
    dataset is ready for use with machine learning models.
    """
    def __init__(self, num_attributes=None, cat_attributes=None,
                 scale=True, imput_data=True, encode=True,
                 strategy='median', scale_mode='StandardScaler',
                 encode_mode='OrdinalEncoder'):

        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.num_attributes = num_attributes 
        self.cat_attributes = cat_attributes 
        self.imput_data = imput_data 
        self.strategy =strategy 
        self.scale = scale
        self.encode = encode 
        self.scale_mode = scale_mode
        self.encode_mode = encode_mode
        
    def fit(self, X, y=None):
        """
        Fit the `FrameUnion` transformer to the input data `X`.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training vector, where `n_samples` is the number of samples,
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target vector relative to X. Not used in this transformer.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self
    
    def transform(self, X):
        """
        Transform `X` by applying specified preprocessing steps.

        This involves auto-detecting or using specified numerical and
        categorical attributes, imputing missing values in numerical
        attributes, scaling numerical attributes, and encoding
        categorical attributes.

        Parameters
        ----------
        X : DataFrame
            The input DataFrame to transform.

        Returns
        -------
        X_transformed : DataFrame or ndarray
            The transformed DataFrame with numerical attributes scaled
            and categorical attributes encoded. It may return a dense
            array or a sparse matrix, depending on the encoding method.
        """
        def extract_arr_columns ( attr, select_type ): 
            """ Extract array and columns from :class:`DataFrameSelector`"""
            frameobj =DataFrameSelector(columns=attr,select_type=select_type)
            arr= frameobj.fit_transform (X )
            return arr, frameobj.columns 
        
        # Construct a Frame if not aleardy dataframe 
        X = build_data_if ( X, to_frame=True, input_name='fu', force=True, 
                           raise_exception= True )
        # Validate and adjust the scale_mode parameter
        self.scale_mode = self.scale_mode if self.scale_mode in [
            "StandardScaler", "MinMaxScaler", "RobustScaler"] else "StandardScaler"
        
        # Select numerical and categorical columns
        num_array, num_columns = extract_arr_columns(self.num_attributes, 'num')
        cat_array, cat_columns = extract_arr_columns(self.cat_attributes, 'cat')
    
        # Impute numerical columns if specified
        if self.imput_data:
            num_array = SimpleImputer(
                strategy=self.strategy).fit_transform(num_array)
    
        # Scale numerical columns if specified
        if self.scale:
            scaler = ( 
                StandardScaler() if self.scale_mode == "StandardScaler" 
                else ( MinMaxScaler() if self.scale_mode =='MinMaxScaler' 
                      else RobustScaler()
                      )
                )
            num_array = scaler.fit_transform(num_array)
        # Encode categorical columns if specified
        if self.encode:
            encoder = ( 
                OrdinalEncoder() if self.encode_mode == "OrdinalEncoder" 
                else OneHotEncoder(sparse_output=True)
                )
            cat_array = encoder.fit_transform(cat_array)
            
            # Handling potential sparse matrix output from OneHotEncoder
            if isinstance(cat_array, np.ndarray):
                warnings.warn('Sparse matrix is converted to a dense Numpy array.',
                              UserWarning)
            elif self.encode_mode == "OneHotEncoder":
                warnings.warn('Using `OneHotEncoder` generates a sparse matrix.'
                              ' Consider handling sparse output accordingly.',
                              UserWarning)
        # Combine numerical and categorical arrays
        try:
            X_transformed = np.c_[num_array, cat_array]
        except ValueError as e:
            raise ValueError(f"Error concatenating transformed features: {e}")
        
        # Try to fallback to DataFrame.
        try : 
            columns = num_columns + cat_columns 
            X_transformed= pd.DataFrame (X_transformed, columns = columns)
        except : 
            pass 
        else: 
            # Keep category values as integers. 
            X_transformed = FloatCategoricalToInt(
                ).fit_transform (X_transformed)
    
        return X_transformed

class FrameUnion(BaseEstimator, TransformerMixin):
    """
    A transformer that combines numerical and categorical feature processing pipelines
    into a unified framework. This includes options for imputing missing values, 
    scaling numerical features, and encoding categorical features. Designed to be
    used within a ColumnTransformer to efficiently preprocess a DataFrame for
    machine learning models.

    Parameters
    ----------
    num_attributes : list of str, default=None
        List of column names in the DataFrame corresponding to numerical attributes.
        These columns will be processed according to the scaling and imputing
        parameters provided.

    cat_attributes : list of str, default=None
        List of column names in the DataFrame corresponding to categorical attributes.
        These columns will be processed according to the encoding parameter provided.

    scale : bool, default=True
        Determines whether numerical features should be scaled. If True, features
        will be scaled using either StandardScaler or MinMaxScaler based on the
        `scale_mode` parameter.

    impute_data : bool, default=True
        If True, missing values in the data will be imputed using SimpleImputer with
        the strategy specified by the `strategy` parameter.

    encode : bool, default=True
        Determines whether categorical features should be encoded. If True,
        features will be encoded using either OrdinalEncoder or OneHotEncoder
        based on the `encode_mode` parameter.

    strategy : str, default='median'
        The strategy used by SimpleImputer to replace missing values. Common strategies
        include 'mean', 'median', and 'most_frequent'.

    scale_mode : str, default='standard'
        Determines the scaling method to be used for numerical features. Options
        are 'standard' for StandardScaler and 'minmax' for MinMaxScaler.

    encode_mode : str, default='ordinal'
        Determines the encoding method to be used for categorical features. Options
        are 'ordinal' for OrdinalEncoder and 'onehot' for OneHotEncoder.

    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.transformers import FrameUnion
    >>> X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    >>> num_attrs = ['age', 'fare']
    >>> cat_attrs = ['embarked', 'sex']
    >>> frame_union = FrameUnion(num_attributes=num_attrs, cat_attributes=cat_attrs,
    ...                          scale=True, impute_data=True, encode=True,
    ...                          strategy='median', scale_mode='standard', encode_mode='onehot')
    >>> X_transformed = frame_union.fit_transform(X)
    >>> print(X_transformed.shape)

    Notes
    -----
    - The `FrameUnion` transformer is particularly useful in preprocessing pipelines
      for machine learning models where both numerical and categorical data require
      different processing steps.
    - It is designed to be flexible and easily adjustable to different preprocessing
      needs by changing its parameters.
    - When `encode_mode` is set to 'onehot', the transformed data might be returned
      as a sparse matrix. Users should handle the output accordingly based on the
      requirements of downstream models or processes.
    """
    def __init__(self, num_attributes=None, cat_attributes=None, scale=True,
                 impute_data=True, encode=True, strategy='median',
                 scale_mode='standard', encode_mode='ordinal'):
        self.num_attributes = num_attributes
        self.cat_attributes = cat_attributes
        self.impute_data = impute_data
        self.scale = scale
        self.encode = encode
        self.strategy = strategy
        self.scale_mode = scale_mode.lower()
        self.encode_mode = encode_mode.lower()
        
    def fit(self, X, y=None):
        return self

    def _validate_attributes(self, X):
        if self.num_attributes is None:
            self.num_attributes = X.select_dtypes([np.number]).columns.tolist()
        if self.cat_attributes is None:
            self.cat_attributes = X.select_dtypes(None, [np.number]).columns.tolist()

        # Validate that the provided attributes are in the DataFrame
        missing_num_attrs = set(self.num_attributes) - set(X.columns)
        missing_cat_attrs = set(self.cat_attributes) - set(X.columns)
        if missing_num_attrs or missing_cat_attrs:
            raise ValueError(f"Missing attributes in the DataFrame: "
                             f"Numerical - {missing_num_attrs}, Categorical"
                             f" - {missing_cat_attrs}")

    def transform(self, X):
        # if not dataframe construct 
        X= build_data_if(X, to_frame=True, input_name='col',
                         force=True, raise_exception=True  )
        # Validate and auto-detect attributes
        self._validate_attributes(X)

        num_pipeline = 'passthrough'
        cat_pipeline = 'passthrough'
        if self.scale:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=self.strategy) 
                 if self.impute_data else 'passthrough'),
                ('scaler', StandardScaler() if 
                 self.scale_mode == 'standard' else MinMaxScaler()),
            ])

        if self.encode:
            cat_pipeline = Pipeline([
                ('encoder', OrdinalEncoder() if self.encode_mode == 'ordinal'
                 else OneHotEncoder(sparse_output=True)),
            ])

        transformers = []
        if self.num_attributes:
            transformers.append(('num', num_pipeline, self.num_attributes))
        if self.cat_attributes:
            transformers.append(('cat', cat_pipeline, self.cat_attributes))

        preprocessor = ColumnTransformer(transformers=transformers)
        X_transformed = preprocessor.fit_transform(X)

        # Handling sparse output warning
        if self.encode_mode == 'onehot' and not isinstance(X_transformed, np.ndarray):
            warnings.warn('Output is a sparse matrix due to OneHotEncoder.'
                          ' Consider using .toarray() or .todense() for '
                          'downstream processing.', UserWarning)

        return X_transformed
   
class FeaturizeX(BaseEstimator, TransformerMixin ): 
    """
    Featurize X using K-Means-based featurization.

    This transformer applies K-Means clustering to the input data `X` and 
    augments it with cluster-based features. It provides options for data 
    scaling, dimensionality reduction using PCA, and the ability to split 
    data into training and testing sets.

    Parameters:
    -----------
    n_clusters: int, default=7
        The number of initial clusters for K-Means.

    target_scale: float, default=5.0
        Scales and includes the scaled values in the input data to 
        enhance K-Means clustering.

    n_components: int, optional
        Number of components for dimensionality reduction using PCA. If not 
        specified, dimensionality reduction is not applied.

    random_state: int, Optional
        The state for shuffling the data before clustering.

    split_X_y: bool, default=False
        If `True`, splits the input data `X` and target data `y` into 
        training and testing sets according to the specified `test_ratio`.

    test_ratio: int, default=0.2
        The ratio of data to keep for testing when splitting data into 
        training and testing sets.

    shuffle: bool, default=True
        Shuffles the data before splitting into training and testing sets.

    return_model: bool, default=False
        If `True`, the K-Means featurization model is included in the
        return results.

    to_sparse: bool, default=False
        If `True`, the output data `X` is converted to a sparse matrix. 
        By default,the sparse matrix is in coordinate matrix (COO) format.

    sparsity: str, default='coo'
        The kind of sparse matrix used to convert `X`. It can be 'csr' or
        'coo'. Any other value will return a coordinate matrix unless 
        `to_sparse` is set to `False`.

    Attributes
    -----------
    kmf_model_: KMeansFeaturizer
        The fitted K-Means Featurizer model.

    Examples
    ---------
    >>> import numpy as np
    >>> from gofast.transformers import FeaturizeX
    >>> X = np.random.randn(12, 7)
    >>> y = np.arange(12)
    >>> y[y < 6] = 0
    >>> y[y > 0] = 1  # For binary data

    # Example 1: Basic Usage
    >>> Xtransf = FeaturizeX(to_sparse=False).fit_transform(X)
    >>> Xtransf.shape
    (12, 8)

    # Example 2: Splitting Data
    >>> Xtransf = FeaturizeX(to_sparse=True, split_X_y=True).fit_transform(X, y)
    >>> Xtransf[0].shape, Xtransf[1].shape
    ((9, 8), (3, 8))

    # Example 3: Returning Model
    >>> *_, kmf_model = FeaturizeX(to_sparse=True, return_model=True
                                   ).fit_transform(X, y)
    >>> kmf_model
    <'KMeansFeaturizer': n_clusters=7, target_scale=5, random_state=None, 
    n_components=None>

    >>> import numpy as np
    >>> from gofast.transformers import FeaturizeX
    >>> X = np.random.randn(12, 7); y = np.arange(12)
    >>> y[y < 6] = 0; y[y > 0] = 1  # For binary data
    >>> Xtransf = FeaturizeX(to_sparse=False).fit_transform(X)
    >>> X.shape, Xtransf.shape
    ((12, 7), (12, 8))
    >>> Xtransf = FeaturizeX(to_sparse=True).fit_transform(X, y)
    >>> Xtransf
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
        with 93 stored elements in COOrdinate format>,
        array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    """

    def __init__(self, 
        n_clusters:int=7, 
        target_scale:float= 5 ,
        random_state:Union [_F, int]=None, 
        n_components: int=None,  
        model: _F =None, 
        test_ratio:Union [float, str]= .2 , 
        shuffle:bool=True, 
        to_sparse: bool=..., 
        sparsity:str ='coo'  
        ): 
        
        self.n_clusters =n_clusters 
        self.target_scale = target_scale 
        self.random_state= random_state 
        self.n_components = n_components 
        self.model=model 
        self.test_ratio=test_ratio 
        self.shuffle=shuffle 
        self.to_sparse=to_sparse 
        self.sparsity=sparsity 
        
    def fit( self, X, y =None): 
        
        """
        Parameters 
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. 
            Note that when `n_components` is set, sparse matrix for `X` is not 
            acceptable. 

        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Return 
        ---------
        self: For chaining methods. 
        
        """
        
        return self 
    
    def transform (self, X, y=None ): 
        """ 
        Parameters 
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features. 
            Note that when `n_components` is set, sparse matrix for `X` is not 
            acceptable. 

        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns 
        -------- 
        X : NDArray shape (m_samples, n_features +1) or \
            shape (m_samples, n_sparse_features)
            Returns transformed array X NDArray of m_features plus the clusters
            features from KMF featurization procedures. The `n_sparse_features`
            is created if `to_sparse` is set to ``True``. 

        """
        
        ( Xtransf,
         * _ 
            ) =  _featurize_X(
                X, 
                y =y, 
                n_cluster = self.n_clusters, 
                target_scale=self.target_scale, 
                random_state= self.random_state,
                n_components = self.n_components, 
                model=self.model,
                test_ratio=self.test_ratio,
                shuffle=self.shuffle,
                to_sparse=self.to_sparse,
                sparsity=self.sparsity,
            )
        
        return Xtransf 
        
def _featurize_X (
    X, 
    y =None, *, 
    n_clusters:int=7, 
    target_scale:float= 5 ,
    random_state:Optional [Union[ _F, int]]=None, 
    n_components: int=None,  
    model: _F =None, 
    split_X_y:bool = False,
    test_ratio:Union[float,str]= .2 , 
    shuffle:bool=True, 
    return_model:bool=...,
    to_sparse: bool=..., 
    sparsity:str ='coo' 
    ): 
    """
    Featurize X using K-Means based featurization.
    
    Parameters:
    -----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training vector, where `n_samples` is the number of samples, and
        `n_features` is the number of features. Note that when `n_components` 
        is set, a sparse matrix for `X` is not acceptable.
    
    y : array-like of shape (n_samples,)
        The target vector relative to X.
    
    n_clusters: int, default=7
        The number of initial clusters.
    
    target_scale: float, default=5.0
        Apply appropriate scaling and include it in the input data to k-means.
    
    n_components: int, optional
        The number of components for reducing the predictor X. It uses PCA to reduce
        the dimension to the most important features.
    
    model: :class:`KMeansFeaturizer`
        K-Means Featurizer model. The model can be provided to featurize test data
        separated from the train data. (added in version 0.2.4)
    
    random_state: int, Optional
        The state for shuffling the data.
    
    split_X_y: bool, default=False
        Split the X, y into training data and test data according to the test size.
    
    test_ratio: int, default=0.2
        The ratio to keep for test data.
    
    shuffle: bool, default=True
        Shuffle the dataset.
    
    return_model: bool, default=False
        If `True`, return the K-Means featurization model along with the 
        transformed X.
    
    to_sparse: bool, default=False
        Convert X data to a sparse matrix. By default, the sparse matrix is in
        coordinate matrix (COO) format.
    
    sparsity: str, default='coo'
        The kind of sparse matrix used to convert `X`. It can be 'csr' or 'coo'. Any
        other value will return a coordinate matrix unless `to_sparse` is set to `False`.
    
    Returns:
    --------
    X, y: NDArray of shape (m_samples, n_features + 1) or shape \
        (m_samples, n_sparse_features)
        Returns an NDArray of m_features plus the cluster features from K-Means
        featurization procedures. The `n_sparse_features` are created if `to_sparse`
        is set to `True`.
    
    X, y, model: NDarray and K-Means Featurizer models
        Returns the transformed array X and y and the model if `return_model`
        is set to `True`.
    
    X, Xtest, y, ytest: NDArray (K-Means Featurizer), ArrayLike
        A split tuple is returned when `split_X_y` is set to `True`.
    
    Note:
    -----
    Whenever `return_model=True`, the K-Means Featurizer model 
    (:class:`KMeansFeaturizer`) is included in the return results.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from gofast.transformers import featurize_X
    >>> X = np.random.randn(12, 7); y = np.arange(12)
    >>> y[y < 6] = 0; y[y > 0] = 1  # For binary data
    >>> Xtransf, _ = featurize_X(X, to_sparse=False)
    >>> X.shape, Xtransf.shape
    ((12, 7), (12, 8))
    >>> Xtransf, y = featurize_X(X, y, to_sparse=True)
    >>> Xtransf, y
    (<12x8 sparse matrix of type '<class 'numpy.float64'>'
        with 93 stored elements in COOrdinate format>,
        array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    >>> featurize_X(X, y, to_sparse=True, split_X_y=True)
    (<9x8 sparse matrix of type '<class 'numpy.float64'>'
        with 71 stored elements in COOrdinate format>,
        <3x8 sparse matrix of type '<class 'numpy.float64'>'
        with 24 stored elements in COOrdinate format>,
        array([0, 1, 1, 0, 0, 0, 0, 1, 1]),
        array([0, 1, 1]))
    >>> *_, kmf_model = featurize_X(X, y, to_sparse=True, return_model=True)
    >>> kmf_model
    <'KMeansFeaturizer': n_clusters=7, target_scale=5, random_state=None,
    n_components=None>
    """

    # set False to value use 
    # ellipsis...
    return_model, to_sparse  =ellipsis2false(return_model, to_sparse )

    # if sparse convert X  to sparse matrix 
    if to_sparse: 
        sparsity= str(sparsity).lower().strip() 
        d_sparsity  = dict ( csr =  sparse.csr_matrix , 
            coo=  sparse.coo_matrix ) 
        sparse_func = sparse.coo_matrix  if sparsity not in (
            'coo', 'csr')  else d_sparsity.get (sparsity ) 
    
    # reduce down feature to two. 
    kmf_data = []
    if n_components: 
        from gofast.analysis import nPCA 
        X =nPCA (X, n_components = n_components  ) 
        
    if split_X_y: 
        X, test_data , y, y_test = train_test_split ( 
            X, y ,test_size = assert_ratio(test_ratio) , 
            random_state = random_state ,
            shuffle =shuffle)
        
    # create a kmeaturization with hint model
    if model: 
        if get_estimator_name(model ) !='KMeansFeaturizer': 
            raise EstimatorError(
                "Wrong model estimator. Expect 'KMeansFeaturizer'"
                f" as the valid estimator. Got {get_estimator_name (model)!r}")
            
        if callable ( model ): 
            model = model (n_clusters=n_clusters, 
            target_scale=target_scale, 
            random_state = random_state)
    else: 
        model = KMeansFeaturizer(
            n_clusters=n_clusters, 
            target_scale=target_scale, 
            random_state = random_state, 
            ).fit(X,y)
        
        ### Use the k-means featurizer to generate cluster features
        # transf_cluster = model.transform(X)
        # Xkmf= np.concatenate (
        #    (X, transf_cluster), axis =1 )
        Xkmf = model.transform(X)
        ### Form new input features with cluster features
        # training_with_cluster

    if to_sparse: 
        Xkmf= sparse_func(Xkmf )

    kmf_data.append(Xkmf)
    kmf_data.append(y) 
    if split_X_y: 
        test_with_cluster= model.transform(test_data)
        if sparse: 
            test_with_cluster= sparse_func(test_with_cluster)
 
        kmf_data.insert(1,test_with_cluster )
        kmf_data.append( y_test)

        
    return  tuple (kmf_data ) + (model, ) \
        if return_model else tuple(kmf_data )

class FeatureSelectorByModel(BaseEstimator, TransformerMixin):
    """
    Select features based on importance weights of a model.

    Parameters
    ----------
    estimator : estimator object, default=RandomForestClassifier()
        The base estimator from which the transformer is built.

    threshold : string, float, optional, default='mean'
        The threshold value to use for feature selection.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification()
    >>> selector = FeatureSelectorByModel()
    >>> X_reduced = selector.fit_transform(X, y)

    Methods
    -------
    fit(X, y)
        Fit the transformer to the data using the provided estimator.

    transform(X, y=None)
        Transform the input data by selecting features based on 
        importance weights.

    """
    def __init__(self, estimator=None, threshold='mean'):
        """
        Initialize the FeatureSelectorByModel.

        Parameters
        ----------
        estimator : estimator object, default=RandomForestClassifier()
            The base estimator from which the transformer is built.

        threshold : string, float, optional, default='mean'
            The threshold value to use for feature selection.

        """
        self.estimator =estimator 
        self.threshold = threshold 
        
    def fit(self, X, y):
        """
        Fit the transformer to the data using the provided 
        estimator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        if self.estimator is None:
            self.estimator = RandomForestClassifier()
        self.selector = SelectFromModel(
            self.estimator, threshold=self.threshold)
        
        self.selector.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the input data by selecting features based on 
        importance weights.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_reduced : array-like, shape (n_samples, n_selected_features)
            Transformed data with selected features.

        """
        return self.selector.transform(X)

class PolynomialFeatureCombiner(BaseEstimator, TransformerMixin):
    """
    Generate polynomial and interaction features.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.

    interaction_only : bool, default=False
        If True, only interaction features are produced.

    Examples
    --------
    >>> X = np.arange(6).reshape(3, 2)
    >>> combiner = PolynomialFeatureCombiner(degree=2)
    >>> X_poly = combiner.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using PolynomialFeatures.

    transform(X, y=None)
        Transform input data by generating polynomial and 
        interaction features.

    """
    def __init__(self, degree=2, interaction_only=False):
        """
        Initialize the PolynomialFeatureCombiner.

        Parameters
        ----------
        degree : int, default=2
            The degree of the polynomial features.

        interaction_only : bool, default=False
            If True, only interaction features are produced.

        """
        self.degree=degree 
        self.interaction_only= interaction_only 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using PolynomialFeatures.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.poly = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only)
        
        self.poly.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data by generating polynomial and interaction 
        features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_poly : array-like, shape (n_samples, n_poly_features)
            Transformed data with polynomial and interaction features.

        """
        return self.poly.transform(X)

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Reduce dimensionality of the data using PCA.

    Parameters
    ----------
    n_components : int, float, None or str
        Number of components to keep. If n_components is not set, 
        all components are kept.

    Examples
    --------
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> reducer = DimensionalityReducer(n_components=1)
    >>> X_reduced = reducer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using PCA.

    transform(X, y=None)
        Transform input data by reducing dimensionality using PCA.

    """
    def __init__(self, n_components=0.95):
        """
        Initialize the DimensionalityReducer.

        Parameters
        ----------
        n_components : int, float, None or str, default=0.95
            Number of components to keep. If n_components is not set,
            all components are kept.

        """
        self.n_components = n_components 
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using PCA.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.reducer = PCA(n_components=self.n_components)
        self.reducer.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform input data by reducing dimensionality using PCA.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_reduced : array-like, shape (n_samples, n_components)
            Transformed data with reduced dimensionality.

        """
        return self.reducer.transform(X)


class BaseCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    drop : {'first', 'if_binary', None}, default='first'
        Specifies a methodology to use to drop one of the categories per 
        feature.

    Attributes
    ----------
    encoder_ : OneHotEncoder object
        The fitted OneHotEncoder instance.

    Examples
    --------
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = CategoricalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using OneHotEncoder.

    transform(X, y=None)
        Transform categorical features into one-hot numeric arrays.

    """
    def __init__(self, drop='first'):
        """
        Initialize the CategoricalEncoder.

        Parameters
        ----------
        drop : {'first', 'if_binary', None}, default='first'
            Specifies a methodology to use to drop one of the categories per 
            feature.

        """
        self.encoder = OneHotEncoder(drop=drop)
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using OneHotEncoder.

        Parameters
        ----------
        X : list or array-like, shape (n_samples, n_features)
            Training data containing categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.encoder.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Transform categorical features into one-hot numeric arrays.

        Parameters
        ----------
        X : list or array-like, shape (n_samples, n_features)
            Input data containing categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_encoded : sparse matrix, shape (n_samples, n_encoded_features)
            Transformed data with one-hot encoding for categorical features.

        """
        return self.encoder.transform(X)

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    This transformer should be applied to categorical features in a dataset 
    before applying it to a machine learning model.

    Parameters
    ----------
    categorical_features : list of str, 
        List of column names to be considered as categorical features.
        If None, features should be detected automatically.

    drop : {'first', 'if_binary', None}, default=None
        Specifies a methodology to use to drop one of the categories per feature.

    Attributes
    ----------
    encoders_ : dict of {str: OneHotEncoder}
        Dictionary containing the OneHotEncoders for each categorical feature.

    Examples
    --------
    >>> from sklearn.compose import ColumnTransformer
    >>> transformer = ColumnTransformer(transformers=[
    ...     ('cat', CategoricalEncoder(categorical_features=['color', 'brand']),
             ['color', 'brand'])
    ... ])
    >>> X = pd.DataFrame({'color': ['red', 'blue', 'green'],
                          'brand': ['ford', 'toyota', 'bmw']})
    >>> transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using OneHotEncoder for each specified
        categorical feature.

    transform(X, y=None)
        Transform categorical features into one-hot numeric arrays.

    """
    def __init__(self, categorical_features=None, drop=None):
        """
        Initialize the CategoricalEncoder2.

        Parameters
        ----------
        categorical_features : list of str
            List of column names to be considered as categorical features.

        drop : {'first', 'if_binary', None}, default=None
            Specifies a methodology to use to drop one of the categories 
            per feature.

        """
        self.categorical_features = categorical_features
        self.drop = drop
        
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using OneHotEncoder for each
        specified categorical feature.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        is_frame(X, df_only= True, raise_exception=True,
                 objname="CategoricalEncoder")
        
        if self.categorical_features is None: 
            *_, self.categorical_features = to_numeric_dtypes(
                X, return_feature_types= True )
            
        self.encoders_ = {
            feature: OneHotEncoder(drop=self.drop) 
            for feature in self.categorical_features}
        
        for feature in self.categorical_features:
            self.encoders_[feature].fit(X[[feature]])
            
        return self
    
    def transform(self, X, y=None):
        """
        Transform categorical features into one-hot numeric arrays.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_encoded : array-like, shape (n_samples, n_encoded_features)
            Transformed data with one-hot encoding for categorical features.

        """
        outputs = []
        for feature in self.categorical_features:
            outputs.append(self.encoders_[feature].transform(X[[feature]]))
        return np.hstack(outputs)

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    This transformer should be applied to numeric features in a dataset before
    applying it to a machine learning model.

    Parameters
    ----------
    numeric_features : list of str
        List of column names to be considered as numeric features.

    Attributes
    ----------
    scaler_ : StandardScaler
        The instance of StandardScaler used for scaling.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> numeric_features = ['age', 'income']
    >>> pipeline = Pipeline(steps=[
    ...     ('scaler', FeatureScaler(numeric_features=numeric_features))
    ... ])
    >>> X = pd.DataFrame({'age': [25, 35, 50], 'income': [50000, 80000, 120000]})
    >>> pipeline.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data using StandardScaler for specified 
        numeric features.

    transform(X, y=None)
        Standardize specified numeric features by removing the mean and 
        scaling to unit variance.

    """
    def __init__(self, numeric_features=None ):
        """
        Initialize the FeatureScaler2.

        Parameters
        ----------
        numeric_features : list of str
            List of column names to be considered as numeric features.

        """
        self.numeric_features = numeric_features
        self.scaler_ = StandardScaler()
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using StandardScaler for specified 
        numeric features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the numeric features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self._is_frame= isinstance (X, pd.DataFrame)
        return self
    
    def transform(self, X):
        """
        Standardize specified numeric features by removing the mean and 
        scaling to unit variance.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the numeric features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_scaled : DataFrame, shape (n_samples, n_features)
            DataFrame with standardized numeric features.

        """
        X = build_data_if(X, input_name='scale_col', force=True, 
                          raise_exception= True )
        
        if self.numeric_features is None: 
            _, self.numeric_features, _ = to_numeric_dtypes(
                X, return_feature_types= True )
        
        exist_features(X, features= self.numeric_features)
        
        self.scaler_.fit(X[self.numeric_features])
        
        X_transformed= X.copy()
        X_transformed[self.numeric_features] = ( 
            self.scaler_.transform(X_transformed[self.numeric_features])
            )
        if not self._is_frame: 
            X_transformed = np.asarray(X_transformed)
            
        return X_transformed


class BaseFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    scaler_ : StandardScaler object
        The fitted StandardScaler instance.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler = FeatureScaler()
    >>> X = [[0, 15], [1, -10]]
    >>> scaler.fit(X)
    >>> scaler.transform(X)
    """

    def __init__(self):
        """
        Initialize the FeatureScaler.

        """
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data using StandardScaler.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data containing the features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.scaler.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Standardize features by removing the mean and scaling to unit variance.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data containing the features to be scaled.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_scaled : array-like, shape (n_samples, n_features)
            Transformed data with standardized features.

        """
        return self.scaler.transform(X)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Imputation transformer for completing missing values.

    This transformer can be applied to both numeric and categorical features 
    in a dataset to impute missing values using the mean, median, or most 
    frequent value.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. If "mean", then replace missing values 
        using the mean along each column. Can also be "median" or "most_frequent".

    Attributes
    ----------
    imputer_ : SimpleImputer
        The instance of SimpleImputer used for imputation.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> imputer = MissingValueImputer(strategy='mean')
    >>> X = pd.DataFrame({'age': [25, np.nan, 50], 'income': [50000, 80000, np.nan]})
    >>> imputer.fit_transform(X)
    >>> imputer = MissingValueImputer(strategy='mean')
    >>> X = [[1, 2], [np.nan, 3], [7, 6]]
    >>> imputer.fit(X)
    >>> imputer.transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Impute missing values using the specified strategy.

    """
    def __init__(self, strategy='mean'):
        """
        Initialize the MissingValueImputer2.

        Parameters
        ----------
        strategy : str, default='mean'
            The imputation strategy. If "mean", then replace missing values 
            using the mean along each column. Can also be "median" or 
            "most_frequent".

        """
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed 
        for this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the features to be imputed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.columns_=[] 
        if isinstance( X, pd.DataFrame): 
            self.columns_= list(X.columns)
            
        self.imputer_.fit(X)
        return self
    
    def transform(self, X):
        """
        Impute missing values using the specified strategy.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the features to be imputed.
        Returns
        -------
        X_transformed : DataFrame, shape (n_samples, n_features)
            DataFrame with missing values imputed based on the chosen strategy.

        """
        # Apply imputation transformation
        X_transformed = self.imputer_.transform(X)
        
        try: 
            if hasattr(self, 'columns_') and self.columns_:
                X_transformed = pd.DataFrame(
                    X_transformed, columns=self.columns_)
        except:
            # Fallback if columns_ attribute doesn't exist or is empty
            X_transformed = pd.DataFrame(
                X_transformed) if self.columns else X_transformed 
    
        return X_transformed

class BaseColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select columns from a DataFrame for processing.

    Parameters
    ----------
    column_names : list of str
        List of column names to select.

    Examples
    --------
    >>> import pandas as pd
    >>> selector = ColumnSelector(column_names=['A', 'B'])
    >>> X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> selector.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed 
        for this transformer.

    transform(X, y=None)
        Select specified columns from the input DataFrame.

    """
    def __init__(self, column_names):
        """
        Initialize the ColumnSelector.

        Parameters
        ----------
        column_names : list of str
            List of column names to select.

        """
        self.column_names = column_names
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed for
        this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Select specified columns from the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_selected : DataFrame, shape (n_samples, n_selected_features)
            DataFrame with only the selected columns.

        """
        if isinstance(X, pd.DataFrame):
            return X[self.column_names]
        else:
            raise TypeError("Input must be a pandas DataFrame")

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select columns from a DataFrame for processing.

    Parameters
    ----------
    column_names : list of str
        List of column names to select.

    Examples
    --------
    >>> selector = ColumnSelector(column_names=['age', 'income'])
    >>> X = pd.DataFrame({'age': [25, 35, 50], 'income': [50000, 80000, 120000], 
                          'gender': ['male', 'female', 'male']})
    >>> selector.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. Checks whether the specified column 
        names exist in the DataFrame.

    transform(X, y=None)
        Select specified columns from the input DataFrame.

    """
    def __init__(self, column_names):
        """
        Initialize the ColumnSelector2.

        Parameters
        ----------
        column_names : list of str
            List of column names to select.

        """
        self.column_names = column_names
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. Checks whether the specified 
        column names exist in the DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        exist_features(X, self.column_names)
        return self
    
    def transform(self, X, y=None):
        """
        Select specified columns from the input DataFrame.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be selected.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_selected : DataFrame, shape (n_samples, n_selected_features)
            DataFrame with only the selected columns.

        """
        is_frame (X, df_only=True, raise_exception= True,
                  objname="LogTransformer" ) 
        
        return X[self.column_names]

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a natural logarithm transformation to numeric features.

    Use this transformer on skewed numeric features to reduce 
    their skewness.

    Parameters
    ----------
    numeric_features : list of str
        List of column names to be considered as numeric features
        for log transformation.

    epsilon : float, default=1e-6
        A small constant to add to input data to avoid taking log of zero.

    Examples
    --------
    >>> transformer = LogTransformer(numeric_features=['income'],
                                     epsilon=1e-6)
    >>> X = pd.DataFrame({'income': [50000, 80000, 120000]})
    >>> transformer.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

    transform(X, y=None)
        Apply the natural logarithm transformation to specified numeric features.

    """
    def __init__(self, numeric_features, epsilon=1e-6):
        """
        Initialize the LogTransformer.

        Parameters
        ----------
        numeric_features : list of str
            List of column names to be considered as numeric features
            for log transformation.

        epsilon : float, default=1e-6
            A small constant to add to input data to avoid taking log of zero.

        """
        self.numeric_features = numeric_features
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data. No actual computation is needed for 
        this transformer.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the columns to be transformed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        return self
    
    def transform(self, X, y=None):
        """
        Apply the natural logarithm transformation to specified numeric features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the columns to be transformed.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        X_transformed : DataFrame, shape (n_samples, n_features)
            DataFrame with the natural logarithm transformation applied to 
            specified numeric features.

        """
        is_frame (X, df_only=True, raise_exception= True,
                  objname="LogTransformer" ) 
            
        X_transformed = X.copy()
        for feature in self.numeric_features:
            X_transformed[feature] = np.log(X_transformed[feature] + self.epsilon)
        return X_transformed



class CategoryFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical variables based on the frequency of each category.

    Parameters
    ----------
    categorical_features : list of str
        List of column names to be considered as categorical features.

    Examples
    --------
    >>> encoder = CategoryFrequencyEncoder(categorical_features=['brand'])
    >>> X = pd.DataFrame({'brand': ['apple', 'apple', 'samsung', 'samsung',
                                    'nokia']})
    >>> encoded_features = encoder.fit_transform(X)

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the data by calculating the frequency of 
        each category in the categorical features.

    transform(X, y=None)
        Encode categorical features based on the calculated frequency mappings.

    Notes
    -----
    CategoryFrequencyEncoder is a transformer that encodes categorical
    variables based on the frequency of each category. It replaces categorical 
    values with their corresponding frequency values, allowing machine learning
    models to capture relationships based on category frequencies.

    The `fit` method calculates the frequency of each category in the specified
    categorical features and stores the mappings. The `transform` method 
    encodes the categorical features using these mappings.

    """
    def __init__(self, categorical_features):
        """
        Initialize the CategoryFrequencyEncoder.

        Parameters
        ----------
        categorical_features : list of str
            List of column names to be considered as categorical features.

        """
        self.categorical_features = categorical_features
        self.frequency_maps_ = None
        
    def fit(self, X, y=None):
        """
        Fit the transformer to the data by calculating the frequency of 
        each category in the categorical features.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Training data containing the categorical features.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        X = build_data_if (X, to_frame=True, force =True, 
                           raise_warning='mute', input_name='cf')
        self.frequency_maps_ = {
            feature: X[feature].value_counts(normalize=True).to_dict() 
            for feature in self.categorical_features
        }
        return self
    
    def transform(self, X, y=None):
        """
        Encode categorical features based on the calculated frequency mappings.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input DataFrame containing the categorical features to be encoded.
        
        y : array-like, shape (n_samples,)
            Target values. Not used in this transformer.

        Returns
        -------
        encoded_data : DataFrame, shape (n_samples, n_features)
            DataFrame with categorical features encoded based on category frequencies.

        """
        X_transformed = X.copy()
        for feature in self.categorical_features:
            X_transformed[feature] = X[feature].map(self.frequency_maps_[feature])
        return X_transformed

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
