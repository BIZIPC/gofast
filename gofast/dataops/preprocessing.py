# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Focuses on initial data preparation tasks and offers core classes and
utilities for data handling and preprocessing. It includes functionality
for  processing features and targets for machine learning tasks.
"""
from __future__ import annotations, print_function 
import copy
from scipy import stats
import warnings 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from .._gofastlog import gofastlog
from ..api.property import BaseClass 
from ..api.summary import ReportFactory
from ..api.types import List, DataFrame, Optional, Series
from ..api.types import Dict, Union, Tuple, ArrayLike
from ..api.util import get_table_size 
from ..decorators import ( 
    Dataify, 
    DynamicMethod, 
    DataTransformer, 
    smartFitRun, 
)
from ..exceptions import NotFittedError
from ..tools.coreutils import ( 
    reshape, 
    is_iterable, 
    assert_ratio, 
    exist_features, 
    format_to_datetime, 
    to_numeric_dtypes, 
)
from ..tools.depsutils import ensure_pkg
from ..tools.validator import ( 
    parameter_validator, 
    check_consistent_length, 
    build_data_if, 
    is_time_series, 
    is_frame, 
    _is_arraylike_1d  
    )

TW = get_table_size() 

__all__= [ 
    "Target", 
    "Features", 
    "apply_bow_vectorization",
    "apply_tfidf_vectorization",
    "apply_word_embeddings",
    "augment_data",
    "transform",
    "boxcox_transformation",
    "transform_dates",
    ]

_logger = gofastlog().get_gofast_logger(__name__)

@smartFitRun
class Target(BaseClass):
    def __init__(self, tnames=None, verbose=False):
        self.tnames = tnames
        self.verbose = verbose

    def fit(self, y, X=None):
        """
        Fits the processor to the target variable and optional feature matrix.

        This method prepares the processor by initializing it with the target
        variable. If the target `y` is specified as a column name and `X` is 
        provided, `y` is extracted from `X`. The method also determines if the 
        target is a multi-label target based on the number of target names.

        Parameters
        ----------
        y : array-like, str, or pandas Series
            The target variable or the name of the target column in `X`.
            If a pandas Series is provided, the name attribute is used as the
            target name.
        X : DataFrame, optional
            The feature matrix containing the target column if `y` is a column name.

        Returns
        -------
        self : Target
            The fitted processor instance.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame when `y` is a column name.
        ValueError
            If the target names do not exist in the provided DataFrame `X`.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.dataops.preprocessing import Target
        >>> data = pd.DataFrame({'feature': np.random.rand(100), 
                                 'target': np.random.choice(['A', 'B', 'C'], 100)})
        >>> processor = Target()
        >>> processor.fit('target', X=data).label_encode()
        >>> print(processor.target_)
        """

        if isinstance(y, str) and X is not None:
            is_frame(X, df_only=True, raise_exception=True)
            self.tnames = is_iterable(y, exclude_string=True, transform=True)
            exist_features(X, features=self.tnames, name='Target')
            y = X[self.tnames].values

        elif hasattr(y, "__array__") and hasattr(y, "name"):
            self.tnames = [y.name]
            y = y.values

        self.multi_label_ = True if self.tnames and len(self.tnames) > 1 else False
        self.target_ = np.asarray(y)
        self.classes_ = np.unique(self.target_)

        return self

    def label_encode(self):
        """
        Encodes class labels to integers.

        This method is used in classification tasks to convert categorical
        class labels into a numerical format, which is a common requirement
        for many machine learning algorithms. The method applies label encoding
        which assigns a unique integer to each class.

        Returns
        -------
        self : Target
            The processor instance after label encoding.

        Raises
        ------
        NotImplementedError
            If the method is called on a multi-label target, which is not supported.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> import numpy as np
        >>> y = np.array(['cat', 'dog', 'bird', 'dog', 'cat'])
        >>> processor = Target()
        >>> processor.fit(y).label_encode()
        >>> print(processor.target_)
        """
        self.inspect  # Ensure the method is fitted
        from sklearn.preprocessing import LabelEncoder
        # Check for multi-label target
        if self.multi_label_:
            raise NotImplementedError(
                "Label encoding for multi-label data is not supported.")

        # Applying label encoding
        encoder = LabelEncoder()
        self.target_ = encoder.fit_transform(self.target_)

        return self

    def one_hot_encode(self):
        """
        Performs one-hot encoding on class labels.

        This method transforms categorical class labels into a one-hot encoded
        format, which is a binary matrix representation where each class is
        represented by a separate column with binary values. One-hot encoding
        is suitable for multi-class classification tasks and is compatible with
        algorithms that expect such a format.

        Returns
        -------
        self : Target
            The processor instance after applying one-hot encoding.

        Raises
        ------
        ValueError
            If the target variable is not set before calling this method.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> import numpy as np
        >>> y = np.array(['red', 'green', 'blue', 'green'])
        >>> processor = Target()
        >>> processor.fit(y).one_hot_encode()
        >>> print(processor.target_)
        """
        self.inspect # Ensure target variable is set
        from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
        
        # Choose the appropriate encoder based on the type of target variable
        if self.multi_label_:
            encoder = MultiLabelBinarizer()
        else:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Apply one-hot encoding
        self.target_ = encoder.fit_transform(self.target_.reshape(-1, 1))

        return self

    def encode_target(self, encoding_type='onehot'):
        """
        Encodes the target variable using label encoding or one-hot encoding.

        The method automatically selects the encoding type based on the nature 
        of the target variable (single-label or multi-label) and user preference.

        Parameters
        ----------
        encoding_type : str, default='onehot'
            The encoding type to use for multi-label data. Options are 'binarize'
            for binary label encoding and 'onehot' for one-hot encoding.

        Returns
        -------
        self : Target
            The processor instance after applying the encoding.

        Raises
        ------
        ValueError
            If an invalid encoding type is specified for multi-label data.
        NotImplementedError
            If label encoding is attempted on multi-label data.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y = np.array(['cat', 'dog', 'fish'])
        >>> processor = Target()
        >>> processor.fit(y).encode_target()
        >>> print(processor.target_)
        """
        from sklearn.preprocessing import ( 
            LabelEncoder, OneHotEncoder, MultiLabelBinarizer)

        self.inspect # Ensure target variable is set
        encoding_type= str(encoding_type).lower() 
        if self.multi_label_:
            if encoding_type == 'binarize':
                encoder = MultiLabelBinarizer()
            elif encoding_type == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            else:
                raise ValueError("Invalid encoding type for multi-label data."
                                 " Choose 'binarize' or 'onehot'.")
            self.target_ = encoder.fit_transform(self.target_)
        else:
            encoder = LabelEncoder()
            self.target_ = encoder.fit_transform(self.target_)

        return self

    def binarize(self, threshold=0.5, multi_label_binarize=False):
        """
        Binarizes the target values based on a specified threshold.

        This method is useful for converting continuous target values into 
        binary outcomes (0 or 1). In the case of multi-label targets, each label 
        can be binarized independently if `multi_label_binarize` is set to True.

        Parameters
        ----------
        threshold : float, default=0.5
            The threshold value used for binarizing the target. Values equal 
            to or greater than the threshold are mapped to 1, and values below 
            the threshold are mapped to 0.
        multi_label_binarize : bool, default=False
            If True and the target is multi-label, each label is binarized 
            independently.

        Returns
        -------
        self : Target
            The processor instance after applying the binarization.

        Raises
        ------
        NotImplementedError
            If the target is multi-label and `multi_label_binarize` 
            is not set to True.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y = np.array([0.2, 0.8, 0.4, 0.9])
        >>> processor = Target()
        >>> processor.fit(y).binarize(threshold=0.5)
        >>> print(processor.target_)
        """
        from sklearn.preprocessing import Binarizer
        self.inspect   # Ensure the Target is fitted
        
        if self.multi_label_ and not multi_label_binarize:
            raise NotImplementedError(
                "Multi-label target detected. Set 'multi_label_binarize' "
                "to True for multi-label binarization.")
            
        if self.multi_label_ and multi_label_binarize:
            # Apply binarization for each label independently
            binarizer = Binarizer(threshold=threshold)
            self.target_ = np.array([binarizer.fit_transform(
                label.reshape(-1, 1)) for label in self.target_.T]).T.squeeze()
        else:
            # Apply binarization for single-label target
            binarizer = Binarizer(threshold=threshold)
            self.target_ = binarizer.fit_transform(self.target_.reshape(-1, 1))

        return self

    def scale(self, method='standardize'):
        """
        Scales the target variable based on the specified method.

        This method adjusts the scale of the target variable, which can be 
        beneficial in regression tasks where the target variable is continuous. 
        Common scaling methods include standardization (zero mean and unit variance)
        and normalization (scaling to a range between 0 and 1).

        Parameters
        ----------
        method : str, default='standardize'
            The scaling method to apply. Options: 'standardize', 'normalize'.

        Returns
        -------
        self : Target
            The processor instance after applying the scaling.

        Raises
        ------
        ValueError
            If an invalid scaling method is specified or if the method is
            applied to multi-label targets.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y = np.random.randn(100)  # Sample continuous target
        >>> processor = Target()
        >>> processor.fit(y).scale(method='normalize')
        >>> print(processor.target_)

        Notes
        -----
        - Normalization/Standardization is less common for multi-label cases,
          especially when labels are binary.
        """
        self.inspect 
        if self.multi_label_:
            raise ValueError("Normalization/Standardization is not applicable"
                             " for multi-label targets.")
        method= str(method).lower() 
        if method == 'standardize':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'normalize':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standardize' or 'normalize'.")

        self.target_ = scaler.fit_transform(self.target_.reshape(-1, 1)).flatten()

        return self

    def split(self, X, test_size=0.3, stratify=True):
        """
        Splits data into training and testing sets, ensuring balanced 
        representation of each class.

        In multi-label contexts, it's important to maintain label distribution 
        across train-test splits. This method optionally applies stratified 
        splitting for this purpose.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        test_size : float, default=0.3
            Proportion of the dataset to include in the test split.
        stratify : bool, default=True
            Whether to use stratified splitting. Stratification is based on 
            the target labels.

        Returns
        -------
        X_train, X_test, y_train, y_test : arrays
            Split data for training and testing.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> X, y = # Load your dataset here
        >>> processor = Target()
        >>> processor.fit(y, X=X)
        >>> X_train, X_test, y_train, y_test = processor.split(X, test_size=0.2)

        Notes
        -----
        - If 'stratify' is set to True, the method will attempt a stratified
          split based on the labels. This is particularly useful for 
          classification tasks with imbalanced classes.
        """
        from sklearn.model_selection import train_test_split

        if self.multi_label_ and stratify:
            raise ValueError("Stratified split is not supported for multi-label targets.")

        stratify_param = self.target_ if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.target_, test_size=test_size, stratify=stratify_param)

        return X_train, X_test, y_train, y_test

    def balance(
        self, X, 
        method='smote', 
        multi_label_handling='independent', 
        return_resampled=False
        ):
        """
        Balances the target data using specified methods like oversampling, 
        undersampling, or SMOTE.

        This method is particularly useful in scenarios where the target variable 
        distribution is imbalanced. For multi-label targets, the balancing can 
        be handled in different ways depending on the specified method.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix corresponding to the target data.
        method : str, default='smote'
           The method used for balancing data. Options:
           - 'oversample': Increases the number of instances in the minority 
              class(es).
           - 'undersample': Decreases the number of instances in the majority
              class(es).
           - 'smote': Uses the Synthetic Minority Over-sampling Technique to 
              create synthetic samples.
        multi_label_handling : str, default='independent'
           The approach for handling multi-label data. Options:
           - 'independent': Treats each label as a separate binary problem.
           - 'combined': Considers the label combinations as a single 
              multi-label problem.
        return_resampled : bool, default=False
            If True, returns the resampled feature matrix along with the
            processor instance.

        Returns
        -------
        self or (self, X_resampled) : Target, (Target, array-like)
            Processor instance after balancing the data, and optionally the 
            resampled feature matrix.

        Raises
        ------
        ValueError
            If an invalid strategy or multi-label handling method is specified.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> X, y = # Load your imbalanced dataset here
        >>> processor = Target()
        >>> processor.fit(y, X=X).balance(X, method='smote',
        ...                                       return_resampled=True)
        >>> print(processor.target_.shape)
        
        More About the Method Parameters 
        ----------------------------------

        The `method` parameter in the `balance_data` method of 
        `Target` defines the strategy used to balance the target 
        distribution. It is particularly crucial in scenarios where the target
        variable is imbalanced, which could potentially lead to biased model 
        training. The available options are:
        
        1. ``'oversample'``: This method involves oversampling the minority 
           class(es) to match the count of the majority class. It randomly 
           replicates examples from the minority class, increasing their 
           presence in the dataset. This approach is straightforward but can 
           lead to overfitting due to the repetition of minority class samples.
        
        2. ``'undersample'``: Contrary to oversampling, undersampling reduces 
           the number of instances in the majority class(es) to equalize with 
           the minority class count. This is done by randomly removing samples
           from the majority class. While it helps in balancing the class 
           distribution, this method can lead to loss of potentially valuable 
           data.
        
        3. ``'smote'`` (Synthetic Minority Over-sampling Technique): SMOTE is 
           an advanced oversampling approach. Instead of simply replicating
           minority class samples, it synthesizes new samples based on the 
           feature space similarities of existing minority instances. SMOTE 
           interpolates between neighboring examples, thereby contributing 
           to more diverse and representative sample generation. This method 
           is widely used due to its effectiveness in handling overfitting.
        
        With `multi_label_handling` Parameter:
        
        In the context of multi-label datasets, where each instance may belong 
        to multiple classes simultaneously, the `multi_label_handling` 
        parameter determines how the balancing is applied across different 
        labels:
        
        1. ``'independent'``: When set to 'independent', the balancing 
           strategy is applied to each label independently. This means the 
           process considers each label as a separate binary classification 
           problem and applies the chosen balancing method (
               oversampling, undersampling, or SMOTE) to each label 
           individually. This approach is suitable when the labels are assumed 
           to be independent of each other.
        
        2. ``'combined'``: The 'combined' setting treats the entire set of 
            labels as one combined multi-label problem. Here, the balancing 
            strategy considers the unique combinations of labels across instances. 
            This approach is more holistic and can be particularly useful when 
            there are dependencies or correlations between different labels.
        
        Both these parameters play a vital role in tailoring the data balancing
        process to the specific needs of the dataset and the learning task, 
        ensuring that the trained models are robust, fair, and less biased.
        
        """
        self.inspect 
        multi_label_handling=str(multi_label_handling).lower()
        if self.multi_label_ and multi_label_handling not in ['independent', 'combined']:
            raise ValueError("Invalid multi-label handling method. Choose "
                             "'independent' or 'combined'.")

        sampler = self._get_sampler(method)
        X_resampled, y_resampled = self._apply_balancing(X, sampler, multi_label_handling)

        self.target_ = y_resampled
        return (self, X_resampled) if return_resampled else self

    @ensure_pkg("imblearn", extra = (
        "Missing 'imbalanced-learn' package. Note `imblearn` is the "
         "shorthand of the package 'imbalanced-learn'.")
        )
    def _get_sampler(self, method):
        """
        Retrieves the appropriate sampler based on the specified balancing method.

        Parameters
        ----------
        method : str
            Balancing method: 'oversample', 'undersample', or 'smote'.

        Returns
        -------
        sampler : imblearn sampler instance
            An instance of the sampler corresponding to the specified method.

        Raises
        ------
        ValueError
            If an invalid balancing method is specified.

        Notes
        -----
        This method simplifies the selection of samplers based on the method string.
        It is used internally in the balance_data method.
        """

        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        
        method = method.lower()
        if method == 'oversample':
            return RandomOverSampler()
        elif method == 'undersample':
            return RandomUnderSampler()
        elif method == 'smote':
            return SMOTE()
        raise ValueError("Invalid balancing method. Choose 'oversample', "
                         "'undersample', or 'smote'.")

    def _apply_balancing(self, X, sampler, multi_label_handling):
        """
        Applies the selected balancing strategy to the data.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix corresponding to the target data.
        sampler : imblearn sampler instance
            The sampler instance to be used for data balancing.
        multi_label_handling : str
            The approach for handling multi-label data: 'independent' or 'combined'.

        Returns
        -------
        X_resampled, y_resampled : array-like, array-like
            The resampled feature matrix and target variable.

        Notes
        -----
        This method is responsible for applying the specified sampling strategy 
        to balance the data. It delegates the handling of multi-label data to 
        another private method when necessary.
        """
        if self.multi_label_ and multi_label_handling == 'independent':
            return self._balance_multi_label_independently(X, sampler)
        return sampler.fit_resample(X, self.target_)

    def _balance_multi_label_independently(self, X, sampler):
        """
        Balances each label independently in multi-label scenarios.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix corresponding to the target data.
        sampler : imblearn sampler instance
            The sampler instance to be used for data balancing.

        Returns
        -------
        X_resampled, y_resampled : array-like, array-like
            The resampled feature matrix and target variable.

        Notes
        -----
        In multi-label contexts, each label is treated as a separate binary 
        problem, and the sampling strategy is applied independently to each label.
        This method ensures that the label distribution is maintained across 
        all labels.
        """
        X_resampled, y_resampled = X, np.empty_like(self.target_)
        for label in range(self.target_.shape[1]):
            X_resampled, y_label = sampler.fit_resample(X_resampled, 
                                                        self.target_[:, label])
            y_resampled[:, label] = y_label
        return X_resampled, y_resampled

    def calculate_metrics(
        self, 
        y_pred, 
        y_true=None, 
        metrics=None, 
        average='macro'
        ):
        """
        Calculates performance metrics for the target variable.

        This method computes various metrics such as accuracy, precision, recall,
        and F1 score to evaluate the performance of a model. In multi-label cases,
        metrics can be calculated for each label individually and then aggregated.

        Parameters
        ----------
        y_pred : array-like
            Predicted target values by the model.
        y_true : array-like, optional
            True target values. If None, the processor's stored target values
            are used.
        metrics : list of str, optional
            The list of metrics to calculate. Available options include 'accuracy',
            'precision', 'recall', 'f1'. If None, all these metrics are calculated.
        average : str, default='macro'
            Determines the type of averaging performed on the data:
            - 'binary': Only report results for the class specified by pos_label.
            - 'micro': Calculate metrics globally by counting the total true positives,
              false negatives, and false positives.
            - 'macro': Calculate metrics for each label, and find their unweighted mean.
            - 'weighted': Calculate metrics for each label, and find their average weighted
              by support (the number of true instances for each label).

        Returns
        -------
        metrics_dict : dict
            A dictionary containing the calculated metrics.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0]
        >>> processor = Target()
        >>> processor.fit(y_true)
        >>> metrics = processor.calculate_metrics(y_pred)
        >>> print(metrics)

        Notes
        -----
        - The choice of averaging method is crucial in multi-label scenarios.
          'macro' averaging treats all labels equally, whereas 'weighted' averaging
          accounts for label imbalance.
        - Precision and recall are particularly useful for imbalanced datasets.
        """
        from sklearn.metrics import ( 
            accuracy_score, precision_score, recall_score, f1_score)

        metrics = metrics or ['accuracy', 'precision', 'recall', 'f1']
        metrics_functions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score
        }
        metrics = is_iterable(metrics, exclude_string=True, transform =True, 
                              parse_string= True )
        # Use the stored target if y_true is not provided
        if y_true is None: 
            self.inspect ;  y_true = self.target_
            
        if y_true is None:
            raise ValueError(
                "y_true is not set. Fit the processor or provide y_true.")
            
        # update multilabel 
        self.multi_label_ =False if _is_arraylike_1d(y_true) else True
            
        metrics_dict = {}
        for metric in metrics:
            if metric in metrics_functions:
                func = metrics_functions[metric]
                # Adjust 'average' parameter for binary and multi-label cases
                calculated_metric = func(y_true, y_pred, average=average) if self.multi_label_ \
                                    else func(y_true, y_pred)
                metrics_dict[metric] = calculated_metric
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return metrics_dict

    def cost_sensitive(self, class_weights=None):
        """
        Adjusts the processor for cost-sensitive learning.

        This method is used to give more importance to certain classes, especially
        useful in imbalanced datasets where minority classes might be of greater interest.
        It can be applied in both single and multi-label scenarios.

        Parameters
        ----------
        class_weights : dict or 'balanced', optional
            Weights associated with classes in the form {class_label: weight}.
            If 'balanced', weights are automatically adjusted inversely proportional
            to class frequencies in the input data. If None, no adjustment is made.

        Returns
        -------
        self : Target
            The processor instance after adjusting for cost-sensitive learning.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y = np.array([0, 1, 0, 1, 1])
        >>> processor = Target()
        >>> processor.fit(y).cost_sensitive({'balanced'})
        >>> print(processor.target_)

        Notes
        -----
        - The class_weights parameter can be crucial in handling imbalanced datasets.
        - In multi-label scenarios, handling class imbalance can be complex and might
          require custom strategies.
          
        The weights are then stored in the class_weights_ attribute of the 
        `Target`. It's important to note that these weights are not 
        directly applied to the target array as it would transform the target 
        variable itself, which is not the typical use case. Instead, these 
        weights are usually used during the training of machine learning models 
        to adjust their learning according to the class importance. The method 
        is highly beneficial for addressing class imbalance issues in datasets,
        especially for classification tasks.
        """
        self.inspect # Ensure the Target is fitted.

        if class_weights == 'balanced':
            # Compute class weights for 'balanced' option
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(self.target_)
            weights = compute_class_weight('balanced', classes=classes, y=self.target_)
            self.class_weights_ = dict(zip(classes, weights))
        elif isinstance(class_weights, dict):
            # Use provided class weights
            self.class_weights_ = class_weights
        else:
            # No class weight adjustment
            self.class_weights_ = None

        # Apply weights to the target - This is an example of how we could 
        # use the weights.
        # In practice, these weights are typically used during model training,
        # not directly applied to the target array.
        # self.target_ = np.array([self.target_weights[val] for val in self.target_])
        
        return self

    def correlation(self, X, method='pearson'):
        """
        Analyzes the correlation between features and the target variable.

        This method helps in understanding how different features influence the target.
        It's particularly useful in feature selection and model interpretability. In
        multi-label scenarios, it can provide insights into feature influence 
        on each label.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        method : str, default='pearson'
            Correlation method to be used. Options include 'pearson', 'spearman', 'kendall'.

        Returns
        -------
        correlation_dict : dict
            A dictionary containing correlation values of each feature with the target.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], 100)
        })
        >>> processor = Target()
        >>> processor.fit('target', X=X)
        >>> correlation_dict = processor.correlation(X)
        >>> print(correlation_dict)

        Notes
        -----
        - The choice of correlation method depends on the nature of the data.
          'pearson' is commonly used for linear relationships, while 'spearman' and
          'kendall' are used for non-linear relationships.
        - In multi-label scenarios, correlation is calculated for each label separately.
        """
        self.inspect # Ensure the Target is fitted.
        
        #if X not dataframe, create a fake data 
        X=build_data_if(X, to_frame=True, input_name= "temp_feature_", 
                        raise_warning ="mute", force=True )
        # Validate correlation method
        method=str(method).lower()
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(f"Invalid correlation method: {method}")

        correlation_dict = {}
        for feature in X.columns:
            if self.multi_label_:
                correlation_dict[feature] = {}
                for idx, label in enumerate(self.tnames):
                    target_series = pd.Series(reshape(self.target_[:, idx]))  
                    correlation_value = X[feature].corr(target_series,method=method)
                    correlation_dict[feature][label] = correlation_value
            else:
                # Convert target to a pandas Series
                target_series = pd.Series(reshape(self.target_))  
                correlation_value = X[feature].corr(target_series, method=method)
                correlation_dict[feature] = correlation_value

        return correlation_dict
    
    @ensure_pkg ("skmultilearn")
    def transform_multi_label(
            self, method='binary_relevance', return_transformed=False):
        """
        Transforms a multi-label problem into a single-label problem.

        This method is useful for making multi-label datasets compatible with 
        standard algorithms that typically handle only single-label formats.  
        Available transformation techniques include binary relevance, 
        classifier chains, and label powerset.

        Parameters
        ----------
        method : str, default='binary_relevance'
            The multi-label transformation technique to apply.
            Options:
            - 'binary_relevance': Treats each label as a separate binary problem. 
              Each label gets its own classifier which predicts the presence or 
              absence of that label independently.
            - 'classifier_chains': Creates a chain of binary classifiers, each 
              predicting a label conditioned on the previous labels. This method 
              takes into account the label correlations.
            - 'label_powerset': Transforms the problem into a multi-class problem 
              with one multi-class classifier trained on all unique label combinations 
              found in the training data. This method effectively captures label
              correlations.
              
        return_transformed : bool, default=False
            If True, returns the transformed target data.
            
        Returns
        -------
        self or transformed_target : Target or ndarray
            If return_transformed is False, returns the processor instance 
            after transforming the multi labels.
            If return_transformed is True, returns the transformed target data.
        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y = # Your multi-label target array
        >>> processor = Target()
        >>> processor.fit(y).transform_multi_label(method='classifier_chains')
        >>> print(processor.transformed_target_)

        Notes
        -----
        The choice of transformation method depends on the nature of the problem and 
        the label correlations. Binary relevance is simple and fast but does not 
        account for label correlations. Classifier chains consider label correlations 
        but are sensitive to the order of labels. Label powerset captures label 
        correlations but can lead to a large number of classes in case of many labels.
        """
        from skmultilearn.problem_transform import ( 
            BinaryRelevance, ClassifierChain, LabelPowerset) 

        self.inspect # Ensure the Target is fitted with multi-label data.

        if not self.multi_label_:
            raise ValueError("The transform_multi_label method is "
                             "applicable only for multi-label data.")

        # Choose the transformation method
        if method == 'binary_relevance':
            transformer = BinaryRelevance()
        elif method == 'classifier_chains':
            transformer = ClassifierChain()
        elif method == 'label_powerset':
            transformer = LabelPowerset()
        else:
            raise ValueError("Invalid method. Choose 'binary_relevance',"
                             " 'classifier_chains', or 'label_powerset'.")

        # Transform the multi-label data
        self.transformed_target_ = transformer.fit_transform(self.target_)

        return self.transformed_target_ if return_transformed else self 

    def threshold_tuning(self, y_scores, threshold=0.5):
        """
        Adjusts the decision threshold for classification tasks.

        This method is useful in scenarios with imbalanced classes or when the 
        costs of different types of misclassifications vary. By adjusting the 
        threshold, the balance between false positives and false negatives can
        be optimized according to the specific requirements of the task.

        Parameters
        ----------
        y_scores : array-like
            The score or probability estimates for the positive class in binary 
            classification.
        threshold : float, default=0.5
            The decision threshold to be applied. Scores above this threshold 
            will be classified as the positive class, and scores below as 
            the negative class.

        Returns
        -------
        y_pred : array-like
            The predicted binary classes after applying the threshold.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> y_scores = np.array([0.3, 0.7, 0.4, 0.6])
        >>> processor = Target()
        >>> y_pred = processor.threshold_tuning(y_scores, threshold=0.6)
        >>> print(y_pred)

        Notes
        -----
        - The choice of threshold should be guided by the specific objectives 
          and constraints of the classification task, such as precision-recall 
          trade-offs.
        - This method is typically applied to the output of binary classifiers 
          where the model outputs a score or probability estimate for the 
          positive class.
        """
        # Convert scores to binary predictions based on the threshold
        y_pred = np.where(y_scores >= threshold, 1, 0)

        return y_pred

    def visualization(
        self, 
        plot_type='distribution', 
        figsize=(10, 6), 
        colormap='viridis'
        ):
        """
        Visualizes the distribution of target classes or labels.

        Provides options for different types of visualizations, including the 
        distribution of classes or labels and, for multi-label data, 
        the co-occurrence or correlation of labels.

        Parameters
        ----------
        plot_type : str, default='distribution'
            Type of plot to generate. Options: 'distribution', 'co_occurrence',
            'correlation'.
        figsize : tuple, default=(10, 6)
            Size of the figure.
        colormap : str, default='viridis'
            Colormap to be used for plots.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Target
        >>> processor = Target()
        >>> y = np.random.choice(['A', 'B', 'C'], 100)
        >>> processor.fit(y).visualization(plot_type='distribution')

        Notes
        -----
        - 'distribution' plots the frequency of each class or label.
        - 'co_occurrence' visualizes how often labels occur together 
          (multi-label data).
        - 'correlation' shows the correlation between different labels 
          (multi-label data).
        """
        
        self.inspect 
        self._validate_plot_type(plot_type)
        plt.figure(figsize=figsize)

        if plot_type == 'distribution':
            self._plot_distribution(colormap)
        elif plot_type == 'co_occurrence':
            self._plot_co_occurrence(colormap)
        elif plot_type == 'correlation':
            self._plot_correlation(colormap)

        plt.title(f"Target Variable Visualization - {plot_type.capitalize()}")
        plt.show()
        
        return self 

    def _validate_plot_type(self, plot_type):
        """
        Validates the plot type for visualization.

        This method checks if the provided plot type is among the supported 
        options and raises a ValueError if not.

        Parameters
        ----------
        plot_type : str
            The type of plot to be generated.

        Raises
        ------
        ValueError
            If the plot_type is not among the supported options.
        """
        valid_types = ['distribution', 'co_occurrence', 'correlation']
        if plot_type not in valid_types:
            raise ValueError(f"Invalid plot type. Choose from {valid_types}.")

    def _plot_distribution(self, colormap):
        """
        Plots the distribution of the target variable.

        This method uses seaborn's countplot to visualize the frequency of 
        each class or label in the target variable.

        Parameters
        ----------
        colormap : str
            The colormap to be used for the plot.
        """
        sns.countplot(x=self.target_, palette=colormap)

    def _plot_co_occurrence(self, colormap):
        """
        Plots the co-occurrence matrix for multi-label data.

        This method visualizes how frequently different labels occur together 
        in the dataset. It's only applicable for multi-label data.

        Parameters
        ----------
        colormap : str
            The colormap to be used for the heatmap.

        Raises
        ------
        ValueError
            If called on a non-multi-label dataset.
        """
        if not self.multi_label_:
            raise ValueError("Co-occurrence plot is only valid for multi-label data.")
        co_occurrence_matrix = np.dot(self.target_.T, self.target_)
        sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap=colormap)

    def _plot_correlation(self, colormap):
        """
        Plots the correlation matrix for multi-label data.

        This method visualizes the correlation between different labels in the 
        dataset, which can be crucial for understanding label relationships in 
        multi-label scenarios.

        Parameters
        ----------
        colormap : str
            The colormap to be used for the heatmap.

        Raises
        ------
        ValueError
            If called on a non-multi-label dataset.
        """
        if not self.multi_label_:
            raise ValueError("Correlation plot is only valid for multi-label data.")
        label_correlation = pd.DataFrame(self.target_).corr()
        sns.heatmap(label_correlation, annot=True, cmap=colormap)

    @property
    def inspect(self):
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `Target` is not fitted yet."""

        msg = ("{dobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )

        if not hasattr (self, "target_"):
            raise NotFittedError(msg.format(
                dobj=self)
            )
        return 1


Target.__doc__ = """\
A comprehensive class for processing and transforming target variables 
in datasets. `Target` handles a range of operations for both single 
and multi-label target datasets. These operations include encoding, binarizing, 
normalizing, balancing imbalanced data, calculating metrics, and visualization.

.. math::
    Target_{processed} = encode(balance(binarize(scale(Target_{original})))

Parameters
----------
tnames : list of str or str, optional
    Names of the target columns in a DataFrame. If the target is a separate 
    array, this can be omitted. Default is `None`.
verbose : bool, optional
    If `True`, the processor will print additional information during processing.
    Default is `False`.

Attributes
----------
multi_label_ : bool
    Indicates whether the target is multi-label. Automatically determined 
    based on the input data.
target_ : array-like
    The processed target variable after applying various transformations.
classes_ : array-like
    Unique classes or labels present in the target variable.
class_weights_ : dict, optional
    Weights associated with classes for cost-sensitive learning.

Methods
-------
fit(y, X=None)
    Fits the processor to the target variable. Extracts target from a 
    DataFrame if `X` is provided and `y` is a column name in `X`.
label_encode()
    Encodes class labels to integers. Suitable for classification tasks.
one_hot_encode()
    Applies one-hot encoding to class labels, returning a binary matrix 
    representation.
binarize(threshold=0.5, multi_label_binarize=False)
    Binarizes the target variable based on a specified threshold.
scale(method='standardize')
    Normalizes continuous target variables, typically for regression tasks.
split(X, test_size=0.3, stratify=True)
    Splits data into training and testing sets, ensuring balanced 
    representation of each class.
balance(X, method='smote', multi_label_handling='independent', 
            return_resampled=False)
    Balances the target data using techniques like SMOTE, oversampling, or
    undersampling.
calculate_metrics(y_pred, y_true=None, metrics=['accuracy', 'precision', 
                'recall', 'f1'], average='macro')
    Calculates various performance metrics for the target variable.
cost_sensitive(class_weights=None)
    Adjusts the model for cost-sensitive learning, giving more importance 
    to certain classes.
correlation(X, method='pearson')
    Analyzes the correlation of features with the target variable.
transform_multi_label(method='binary_relevance', return_transformed=False)
    Transforms a multi-label problem into a single-label problem using 
    various techniques.
threshold_tuning(y_scores, threshold=0.5)
    Adjusts the decision thresholds for classification tasks.
visualization(plot_type='distribution', figsize=(10, 6), 
             colormap='viridis')
    Creates plots for visualizing different aspects of the target variable.

Examples
--------
>>> from gofast.dataops.preprocessing import Target
>>> import pandas as pd
>>> import numpy as np
>>> 
>>> # Sample data
>>> data = pd.DataFrame({
...     'feature': np.random.rand(100),
...     'target': np.random.choice(['A', 'B', 'C'], 100)
... })
>>> 
>>> # Instantiate and process targets
>>> processor = Target(tnames='target', verbose=True)
>>> processor.fit('target', X=data)\
...          .label_encode()\
...          .balance_data(data[['feature']], method='smote', 
...                        return_resampled=True)
>>> print(processor.target_)
[1 0 2 ... 1 0 1]

Notes
-----
- The `Target` class is designed to streamline the processing 
  of target variables in both single-label and multi-label contexts.
- It supports a variety of transformation techniques to prepare the target 
  data for machine learning models.
- Method chaining is utilized to allow for a fluent and readable processing 
  pipeline.
- Ensure that the target data is properly fitted using the `fit` method before 
  applying any transformations.

See Also
--------
pandas.DataFrame : The primary data structure used for data manipulation.
sklearn.preprocessing : Module containing preprocessing utilities.
imbalanced-learn : Library for handling imbalanced datasets.
skmultilearn : Library for multi-label classification.

References
----------
.. [1] Zhang, M., & Zhou, Z.-H. (2014). A review on multi-label learning algorithms.
       *IEEE Transactions on Knowledge and Data Engineering*, 26(8), 1819-1837.
.. [2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
       SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial 
       Intelligence Research*, 16, 321-357.
.. [3] Tsoumakas, G., & Katakis, I. (2007). Multi-label classification: An overview.
       *International Journal of Data Warehousing and Mining*, 3(3), 1-13.
.. [4] Girshick, R., & He, K. (2015). Fast R-CNN. *Proceedings of the IEEE International 
       Conference on Computer Vision* (ICCV), 1440-1448.
.. [5] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
       *arXiv preprint arXiv:1412.6980*.
"""


@smartFitRun 
class Features(BaseClass):
    def __init__(self, features=None, tnames=None, verbose=False):
        self.features = features
        self.tnames = tnames
        self.verbose = verbose

    def fit(self, X, y=None):
        """
        Fit the processor to the data.
    
        This method prepares the processor for subsequent operations. It stores
        the data and identifies numeric and categorical features for processing.
        If a target variable 'y' is provided, it is also stored for potential use
        in feature processing tasks that may require the target variable (like
        supervised feature selection).
    
        Parameters
        ----------
        X : DataFrame, dict-like, or array-like
            The data to process. If a dictionary or an array is passed, it is
            converted into a DataFrame using specified or default column names.
        y : array-like, optional
            The target variable for supervised learning tasks. Default is None.
    
        Returns
        -------
        self : Features
            The fitted processor instance.
    
        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame after processing.
    
        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features
        >>> import pandas as pd
        >>> data = pd.DataFrame({'age': [25, 32, 40], 'gender': ['M', 'F', 'M']})
        >>> processor = Features(features=['age', 'gender'])
        >>> processor.fit(data)
        >>> print(processor.numeric_features_, processor.categorical_features_)
    
        Notes
        -----
        The method converts input data into a DataFrame, extracts the target variable
        if provided, and separates numeric and categorical features. This setup
        facilitates various feature processing tasks that may follow.
        """
        from ..tools.baseutils import get_target, select_features 
        from ..tools.mlutils import bi_selector
   
        # Ensure input data is a DataFrame
        X = build_data_if(
            X, 
            columns=self.features, 
            to_frame=True, 
            input_name='feature_', 
            force=True, 
            raise_warning='mute'
        )
        # verify integrity 
        X = to_numeric_dtypes(X)
        # Extract target from 'tnames'
        if self.tnames is not None: 
            y, X = get_target(X, tname=self.tnames)
        
        # Type check for DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
    
        # Select specified features or use all
        if self.features is not None:
            X = select_features(X, features=self.features)
        else:
            self.features = list(X.columns)
    
        # Automatically separate numeric and categorical features
        self.numeric_features_, self.categorical_features_ = bi_selector(X)
        
        # Store a copy of the data and target variable
        self.data = X.copy()
        self.y = copy.deepcopy(y)
    
        return self

    def normalize(self, numeric_features=None):
        """
        Normalizes numeric features to a range [0, 1].

        This operation scales each feature to a given range and is useful in 
        scenarios where features vary widely in scale and you want to 
        normalize their range  without distorting differences in the ranges 
        of values.

        The normalization is performed using the formula:
        \[
            X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
        \]
        where \(X\) is the original value, \(X_{\text{min}}\) is the minimum value 
        of the feature, and \(X_{\text{max}}\) is the maximum value of the feature.

        Parameters
        ----------
        numeric_features : list of str, optional
            List of numeric feature names to normalize. If None, uses the numeric
            features identified during fitting.

        Returns
        -------
        self : Features
            The processor instance with normalized data.

        Raises
        ------
        RuntimeError
            If the method is called before fitting the processor with data.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features
        >>> processor = Features()
        >>> data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> processor.fit(data)
        >>> processor.normalize()
        >>> print(processor.data)
        """
        self.inspect  # Ensure the method is fitted

        numeric_features = numeric_features or self.numeric_features_
        if self.data is None:
            raise RuntimeError("Fit the processor to the data first using fit(X).")

        # Normalization operation
        self.data[numeric_features] = (
            self.data[numeric_features] - self.data[numeric_features].min()
            ) / (self.data[numeric_features].max() - self.data[numeric_features].min())

        return self

    def standardize(self, numeric_features=None):
        """
        Standardizes numeric features to have zero mean and unit variance.

        Standardization is a common requirement for many machine learning algorithms,
        particularly those that assume the data is centered around zero and scaled,
        such as Support Vector Machines (SVM) and k-Nearest Neighbors (k-NN). This
        method transforms the features to ensure they have a mean of zero and a 
        standard deviation of one.

        The standardization is performed using the formula:
        \[
            X_{\text{std}} = \frac{X - \mu}{\sigma}
        \]
        where \(X\) is the original value, \(\mu\) is the mean of the feature, 
        and \(\sigma\) is the standard deviation of the feature.

        Parameters
        ----------
        numeric_features : list of str, optional
            List of numeric feature names to standardize. If None, all numeric
            features identified during fitting are standardized.

        Returns
        -------
        self : Features
            The processor instance with standardized data.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features
        >>> processor = Features()
        >>> data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> processor.fit(data)
        >>> processor.standardize()
        >>> print(processor.data)
        """
        self.inspect  # Ensure the method is fitted
        from sklearn.preprocessing import StandardScaler 
        numeric_features = numeric_features or self.numeric_features_
        
        if len(numeric_features) == 0:
            warnings.warn("Missing numeric features. Standardization cannot be performed.")
            return self 

        scaler = StandardScaler()
        self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

        return self

    def handle_missing(self, strategy='mean', missing_values=np.nan):
        """
        Handles missing values in the dataset by imputing them based on a 
        specified strategy.

        Imputation is a fundamental step in data preprocessing, especially in 
        datasets where missing values are present. This method provides various 
        strategies for imputation, such as replacing missing values with the 
        mean, median, most frequent value, or a constant.

        Parameters
        ----------
        strategy : str, default='mean'
            Strategy to impute missing values. Options: 'mean', 'median', 
            'most_frequent', 'constant'.
        missing_values : number, str, np.nan, default=np.nan
            The placeholder for the missing values.

        Returns
        -------
        self : Features
            The processor instance with imputed data.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features
        >>> processor = Features()
        >>> data = pd.DataFrame({'feature1': [1, np.nan, 3], 'feature2': [4, 5, np.nan]})
        >>> processor.fit(data).handle_missing(strategy='mean')
        >>> print(processor.data)
        """
        self.inspect  # Ensure the method is fitted

        from sklearn.impute import SimpleImputer 
        imputer = SimpleImputer(strategy=strategy, missing_values=missing_values)
        for column in self.data.columns:
            if self.data[column].isna().any():
                self.data[column] = imputer.fit_transform(self.data[[column]])
        
        # update features 
        self.update_features
        
        return self

    def encode_categorical(self, features=None, method='onehot'):
        """
        Encodes categorical features in the dataset using specified encoding 
        method.

        Encoding categorical variables is an essential step in data 
        preprocessing, especially for machine learning models that require 
        numerical input. This method offers options for one-hot encoding and 
        label encoding.

        Parameters
        ----------
        features : list, optional
            List of categorical feature names to encode. If None, all 
            categorical features identified during fitting are encoded.
        method : str, default='onehot'
            Method to encode categorical variables. Options: 'onehot', 'label'.

        Returns
        -------
        self : Features
            The processor instance with encoded data.

        Examples
        --------
        >>> processor = Features()
        >>> data = pd.DataFrame({'category_feature': ['A', 'B', 'C'], 'feature2': [1, 2, 3]})
        >>> processor.fit(data).encode_categorical(['category_feature'])
        >>> print(processor.data)
        """
        self.inspect  # Ensure the method is fitted

        from sklearn.preprocessing import ( OneHotEncoder, LabelEncoder ) 
        method = str(method).lower()  # Normalize the method to lowercase
        features = features or self.categorical_features_

        if method not in ['onehot', 'label']:
            raise ValueError("Invalid encoding method. Choose 'onehot' or 'label'.")

        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(self.data[features])
            encoded_features = encoder.get_feature_names_out(features)
            self.data.drop(features, axis=1, inplace=True)
            self.data = pd.concat([self.data, pd.DataFrame(
                encoded_data, columns=encoded_features)], axis=1)
        elif method == 'label':
            encoder = LabelEncoder()
            for feature in features:
                self.data[feature] = encoder.fit_transform(self.data[feature])

        # update features 
        self.update_features
        
        return self

    def selection(self, y=None, k=10, method='chi2'):
        """
        Selects the top K features based on a specified selection method.

        Feature selection is crucial for reducing the dimensionality of the data 
        and improving the performance of machine learning models. This method 
        supports various statistical methods to select the most significant 
        features.

        Parameters
        ----------
        y : array-like, optional
            The target variable for supervised learning models. If not provided,
            the method will use `self.y` if it is available. It is used to help in 
            selecting features based on their relationship with the target. 
            Default is None.
        k : int, default=10
            Number of top features to select.
        method : str, default='chi2'
            Method for feature selection. Options: 'chi2', 'f_classif',
            'f_regression'.

        Returns
        -------
        self : Features
            The processor instance with selected features.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame(np.random.rand(100, 50))  # 100 samples, 50 features
        >>> processor.fit(data).selection(k=5, method='chi2')
        >>> print(processor.data.shape)
        
        Notes
        -----
        The 'y' parameter, or `self.y` if `y` is not provided, is particularly 
        important for methods like 'chi2', 'f_classif', and 'f_regression', 
        which are used in the context of supervised learning and require a target 
        variable to determine the relevance of features.
        """
        self.inspect  # Ensure the method is fitted

        y = y or self.y  # Use provided y or fall back to self.y
        method = str(method).lower()  # Normalize the method to lowercase
        from sklearn.feature_selection import ( 
            SelectKBest, chi2, f_classif, f_regression)
        if method == 'chi2':
            selector = SelectKBest(chi2, k=k)
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'f_regression':
            selector = SelectKBest(f_regression, k=k)
        else:
            raise ValueError("Invalid feature selection method. Choose 'chi2',"
                             " 'f_classif', or 'f_regression'.")
        # If y is still None at this point, raise an error
        if y is None:
            raise ValueError("Target variable y is not provided and self.y is not set. "
                             "Feature selection requires a target variable.")

        self.data = selector.fit_transform(self.data, y=y)

        # update features 
        self.update_features
        
        return self

    def extraction(self, n_components=2, **pca_kws):
        """
        Performs feature extraction using Principal Component Analysis (PCA).

        Feature extraction is a way to reduce dimensionality by transforming 
        features into a lower-dimensional space. PCA is commonly used for this 
        purpose, especially useful for visualization or preprocessing before 
        applying machine learning algorithms.

        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to extract.
        **pca_kws : dict, optional
            Additional keyword arguments to be passed to the PCA constructor.

        Returns
        -------
        self : Features
            The processor instance with extracted features.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame(np.random.rand(100, 50))  # 100 samples, 50 features
        >>> processor.fit(data).extraction(n_components=3)
        >>> print(processor.data.shape)
        
        Notes
        -----
        The `n_components` parameter determines the number of principal 
        components that the PCA will extract. This can be adjusted based on 
        the desired level of dimensionality reduction.

        The `**pca_kws` allows for additional customization of the PCA model, 
        enabling the user to specify other parameters such as `svd_solver`, 
        `whiten`, etc., as per sklearn's PCA documentation.
        """
        self.inspect  # Ensure the method is fitted
        
        from sklearn.decomposition import PCA
        # Expect numeric features 
        _, numeric_cols, cat_cols = to_numeric_dtypes(
            self.data, return_feature_types=True)
        if len(cat_cols)!=0: 
            warnings.warn(
                "Categorical features {cat_cols} detected, and skipped"
                " for feature extraction analyses."
            )
        if len(numeric_cols)==0: 
            warnings.warn("No numeric features found in the dataset."
                          " Please check your data.")
            
            return self 
             
        pca = PCA(n_components=n_components, **pca_kws)
        self.data = pca.fit_transform(self.data)
        

        return self

    def correlation(
        self, 
        threshold=0.8, 
        drop_correlated_features=True, 
        method='pearson'):
        """
        Performs correlation analysis using a specified method on the features.

        This method identifies features that are highly correlated with each 
        other. High correlation between features can imply redundancy, and 
        such features can be considered for removal to reduce model complexity 
        and multicollinearity.

        Parameters
        ----------
        threshold : float, default=0.8
            The threshold for identifying high correlation.
        drop_correlated_features : bool, default=True
            If True, automatically drops the identified highly correlated 
            features from the data.
        method : str, default='pearson'
            Method to compute correlation. Options: 'pearson', 'spearman', 
            'kendall'.

        Returns
        -------
        self : Features
            The processor instance after the correlation analysis.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame(np.random.rand(100, 10))
        >>> processor.fit(data).correlation(threshold=0.85, method='spearman')
        >>> print(processor.data.shape)

        Notes
        -----
        The choice of correlation method depends on the data distribution and
        the nature of the relationships between features.
        """
        self.inspect  # Ensure the method is fitted

        valid_methods = ['pearson', 'spearman', 'kendall']
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {valid_methods}.")

        corr_matrix = self.data.corr(method=method).abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identifying pairs of highly correlated features
        highly_correlated_features = [column for column in upper_tri.columns 
                                      if any(upper_tri[column] > threshold)]
        if drop_correlated_features:
            self.data.drop(columns=highly_correlated_features, inplace=True)
            print(f"Dropped highly correlated features using {method} method:"
                  f" {highly_correlated_features}")
        else:
            print(f"Highly correlated features using {method} method"
                  f" (threshold > {threshold}): {highly_correlated_features}")
        
        # update features 
        self.update_features
        
        return self

    def ts_features(
        self, 
        datetime_column, 
        drop_original_column=True,
        include=None
        ):
        """
        Extracts time series features from a specified datetime column.

        This method enhances the dataset by decomposing a datetime column into
        its constituent parts like year, month, day, etc., when specified. If
        'include' is None, it formats the datetime column without decomposing it
        into separate components.

        Parameters
        ----------
        datetime_column : str
            The column name in the DataFrame which contains datetime objects.
        drop_original_column : bool, default=True
            If True, the original datetime column will be dropped from the DataFrame.
        include : list of str or None, optional
            Specifies the time components to be extracted. Each element in the list
            should be one of 'year', 'month', 'day', 'weekday', 'hour'.
            If None, the datetime column is formatted without decomposition.

        Returns
        -------
        self : Features
            The processor instance with the extracted or formatted features added.

        Examples
        --------
        Extract specific time components:
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame({'datetime': pd.date_range(start='2021-01-01',
        ...                                                   periods=100, freq='D')})
        >>> processor.fit(data).time_series_features('datetime',
                                                     include=['year', 'month', 'day'])

        Format datetime without decomposition:
        >>> processor.fit(data).ts_features('datetime')

        Notes
        -----
        - 'weekday' extracts the day of the week, where Monday=0 and Sunday=6.
        - Ensure that the datetime_column contains valid datetime objects. Use
          pd.to_datetime() if necessary to convert to a datetime format.
        - The method also checks if the column is suitable for time series analysis
          when 'include' is None.
        """
        self.inspect  # Check if the processor is already fitted with data.

        # Ensure the datetime column exists in the DataFrame
        if datetime_column not in self.data.columns:
            raise ValueError(f"{datetime_column} not found in the DataFrame.")

        if include is None: 
            # Format the datetime column without decomposition
            self.data = format_to_datetime(self.data, date_col= datetime_column )
            is_time_series(self.data, time_col = datetime_column )
            
            return self  
        
        # set include as iterable is not 
        include = is_iterable(include, exclude_string= True, transform =True, 
                              parse_string= True)
        # Convert column to datetime format if not already
        datetime_series = pd.to_datetime(self.data[datetime_column])
        
        # Extracting specified time components
        time_features = {
            'year': datetime_series.dt.year,
            'month': datetime_series.dt.month,
            'day': datetime_series.dt.day,
            'weekday': datetime_series.dt.weekday,
            'hour': datetime_series.dt.hour
        }

        for feature in include:
            if feature in time_features:
                self.data[f'{datetime_column}_{feature}'] = time_features[feature]
            else:
                raise ValueError(f"Invalid time component '{feature}' in 'include' list.")

        # Dropping the original datetime column if specified
        if drop_original_column:
            self.data.drop(columns=datetime_column, inplace=True)

        # update features 
        self.update_features
        
        return self

    def interaction(
        self, combinations, 
        method='multiply',
        drop_original=False):
        """
        Creates new features by interacting existing features.

        This method generates new features based on interactions (such as
        multiplication, addition, subtraction, or division) between pairs of
        existing features. This can help in capturing non-linear relationships
        and interactions between features.

        Parameters
        ----------
        combinations : list of tuples
            List of pairs of feature names to interact. Each tuple should contain
            two feature names.
        method : str, default='multiply'
            Interaction method. Options: 'multiply', 'add', 'subtract', 'divide'.
            
        drop_original : bool, default=False
            If True, the original columns used in the interactions 
            will be dropped.
        Returns
        -------
        self : Features
            The processor instance after feature interaction.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        >>> processor.fit(data).interaction([('feature_0', 'feature_1'),
                                                     ('feature_2', 'feature_3')], method='add')

        Notes
        -----
        - The method parameter defines the type of interaction between features.
          For example, 'multiply' will multiply the values of the two features,
          while 'add' will sum them.
        - The new features are named using the pattern 'featureA_method_featureB',
          which helps in identifying the original features and the interaction method used.
        - Ensure that the features specified in combinations exist in the data.
        """
        self.inspect  # Ensure the processor is fitted with data

        # Define the interaction operations
        operations = {
            'multiply': lambda a, b: a * b,
            'add': lambda a, b: a + b,
            'subtract': lambda a, b: a - b,
            'divide': lambda a, b: a / b
        }

        if method not in operations:
            raise ValueError("Invalid method. Choose from 'multiply',"
                             " 'add', 'subtract', 'divide'.")

        original_columns = set()

        for feature_a, feature_b in combinations:
            if feature_a not in self.data.columns or feature_b not in self.data.columns:
                raise ValueError(f"Features {feature_a} or {feature_b} not found in data.")

            operation = operations[method]
            new_feature_name = f"{feature_a}_{method}_{feature_b}"
            self.data[new_feature_name] = operation(self.data[feature_a], self.data[feature_b])

            original_columns.update([feature_a, feature_b])

        if drop_original:
            self.data.drop(columns=list(original_columns), inplace=True)

        # update features 
        self.update_features
         
        return self

    def binning(self, features, bins,
                labels=None, method='quantile', 
                drop_original=False):
        """
        Applies binning (or discretization) to a specified numeric feature.

        Binning is used to convert continuous numeric features into categorical
        bins. This can be useful for linear models, decision trees, or when
        working with non-linear relationships in the data.

        This method is valuable for modeling non-linear relationships and can 
        simplify models by converting continuous variables into categorical ones.

        Parameters
        ----------
        features : list of str
            Names of numeric features to be binned.
        bins : int, dict
            Number of bins for each feature. If int, the same number of bins 
            is applied to all features.
            If dict, specifies individual bin counts for each feature 
            (e.g., {'feature1': 4, 'feature2': 3}).
        labels : list, dict, optional
            Labels for the bins. Can be a list (common labels for all features)
            or a dict with feature-specific labels.
        method : str, default='quantile'
            Binning method: 'quantile' for quantile-based binning, 'uniform' 
            for equal-width bins.
        drop_original : bool, default=False
            If True, the original features are dropped after binning.

        Returns
        -------
        self : Features
            The instance itself, updated with binned features.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame({'value': np.random.rand(100)})
        >>> processor.fit(data).binning('value', bins=5, method='uniform')
        >>> processor = Features()
        >>> data = pd.DataFrame({'feature_0': np.random.randn(100),
                                 'feature_1': np.random.randn(100)})
        >>> processor.fit(data)
        >>> processor.binning(['feature_0', 'feature_1'], bins=5, 
                              method='uniform', drop_original=True)
        Notes
        -----
        The choice of binning method can significantly impact model performance
        and interpretation, especially in cases of non-linear relationships.
        """
        self.inspect  # Check if the processor is already fitted

        # Normalize the bins and labels parameters
        bin_settings = bins if isinstance(bins, dict) else {
            feature: bins for feature in features}
        label_settings = labels if isinstance(labels, dict) else {
            feature: labels for feature in features}

        for feature in features:
            self._validate_feature_in_data(feature)
            bin_count = bin_settings.get(feature)
            feature_labels = label_settings.get(feature)

            binned_feature = self._create_binned_feature(
                self.data[feature], bin_count, feature_labels, method)
            self.data[f'{feature}_binned'] = binned_feature

            if drop_original:
                self.data.drop(columns=feature, inplace=True)
                
        # update features 
        self.update_features
        
        return self

    def clustering(
        self, n_clusters,
        features=None, 
        new_feature_name='cluster', 
        drop_original=False):
        """
        Performs clustering on the features and adds cluster labels as a new 
        feature.

        This method applies a clustering algorithm (e.g., KMeans) to group data
        points into clusters. This can be used as a form of feature engineering
        to capture non-linear relationships between features.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to form.
        features : list, optional
            List of feature names to be used for clustering. If None, all features
            are used.
        new_feature_name : str, default='cluster'
            Name of the new feature to be created with cluster labels.
        drop_original : bool, default=False
            If True, the original features used for clustering are dropped 
            from the DataFrame.
        Returns
        -------
        self : Features
            Instance with an additional clustering feature.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame(np.random.rand(100, 5))
        >>> processor.fit(data).feature_clustering(n_clusters=3)

        Notes
        -----
        The number of clusters and the choice of features for clustering
        should be informed by domain knowledge and exploratory data analysis.
        """
        self.inspect  # Check if the processor is already fitted
        # get the numeric features only 
        features = features or self.numeric_features_ 
        if features is None: 
            # for consistency try to get it explicity with the data 
            features = self.data.select_dtypes(include=[np.number]).columns 
            
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_init="auto", n_clusters=n_clusters)
        self.data[new_feature_name] = kmeans.fit_predict(self.data[features])

        # Dropping original features used for clustering if specified
        if drop_original:
            self.data.drop(columns=features, inplace=True)
            
        # update features 
        self.update_features
        
        return self

    def text_extraction(
        self, 
        text_column, 
        method='tfidf', 
        max_features=100, 
        **kwargs
        ):
        """
        Extracts features from text data using specified text analysis 
        techniques.

        This method applies text feature extraction techniques like TF-IDF or
        CountVectorizer to convert text data into a more usable format for 
        machine learning models.

        Parameters
        ----------
        text_column : str
            The name of the column containing text data.
        method : str, default='tfidf'
            The method to use for text feature extraction. Options: 'tfidf', 'count'.
        max_features : int, default=100
            The maximum number of features to extract.
        **kwargs : dict
            Additional keyword arguments to pass to the text extraction method.

        Returns
        -------
        self : Features
            The instance itself, updated with the extracted text features.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = pd.DataFrame({'text': ['sample text data', 'another sample text']})
        >>> processor.fit(data).text_extraction('text', method='tfidf', max_features=50)

        Notes
        -----
        - 'tfidf' refers to Term Frequency-Inverse Document Frequency, which 
          considers the overall document weightage of a word.
        - 'count' refers to CountVectorizer, which counts the number of times 
          a word appears in the document.
        """
        self.inspect  # Ensure the method is fitted with data

        if method not in ['tfidf', 'count']:
            raise ValueError("Invalid method for text feature extraction."
                             " Choose 'tfidf' or 'count'.")
        
        # Text Feature Extraction
        if method == 'tfidf':
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
        elif method == 'count':
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer(max_features=max_features, **kwargs)

        text_features = vectorizer.fit_transform(self.data[text_column]).toarray()
        try: 
            feature_names = vectorizer.get_feature_names_out()
        except: 
            # Alternatively, you can use the old method
            feature_names = vectorizer.get_feature_names()
            
        text_features_df = pd.DataFrame(text_features, columns=feature_names)

        # Update DataFrame
        self.data = pd.concat([self.data.drop(columns=[text_column]),
                               text_features_df], axis=1)
        
        self.update_features 
        return self

    @ensure_pkg("skimage", 
                extra= ( 
                        "Image Feature Extraction expects 'scikit-image'"
                        " package to be installed."
                        ), 
                condition=lambda *args, **kwargs: kwargs.get(
                    'method', args[1] if len(args) > 1 else '') == 'hog'
            )
    def image_extraction(
        self, image_column, 
        method='hog', **kwargs
        ):
        """
        Extracts features from image data using specified techniques.

        This method applies image processing techniques to convert image data into
        a format that can be used for machine learning models. Techniques like HOG
        (Histogram of Oriented Gradients) or custom methods can be utilized.

        Parameters
        ----------
        image_column : str
            The name of the column containing image data. Each entry is expected
            to be an image.
        method : str, default='hog'
            The method to use for image feature extraction. Default is 'hog'.
            Custom methods can be defined and passed.
        **kwargs : dict
            Additional keyword arguments to pass to the image extraction method.

        Returns
        -------
        self : Features
            The instance itself, updated with the extracted image features.

        Examples
        --------
        >>> from gofast.dataops.preprocessing import Features 
        >>> processor = Features()
        >>> data = # DataFrame containing an image column
        >>> processor.fit(data).image_extraction('image_column', method='hog')

        Notes
        -----
        - Ensure that the image data is preprocessed (e.g., resized) as 
          needed before using this method.
        - Custom methods for image feature extraction can be implemented by 
          the user and passed through the 'method' parameter.
        """
        self.inspect  # Ensure the method is fitted with data

        # Image Feature Extraction
        if method == 'hog':
            from skimage.feature import hog
            extractor_function = lambda image: hog(image, **kwargs)
        elif callable(method):
            extractor_function = method
        else:
            raise ValueError("Invalid method for image feature extraction."
                             " Provide 'hog' or a custom function.")

        # Applying extractor to each image
        extracted_features = self.data[image_column].apply(extractor_function)
        self.data = pd.concat([self.data.drop(columns=[image_column]), 
                               pd.DataFrame(extracted_features)], axis=1)
        self.update_features 
        
        return self

    @property 
    def update_features (self ): 
        """ Update the numeric and categorical features lists based "
        "on the current DataFrame."""
        from ..tools.mlutils import bi_selector 
        self.features = list (self.data.columns )
        self.numeric_features_, self.categorical_features_ = bi_selector ( 
            self.data )

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `Features` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'numeric_features_'): 
            raise NotFittedError(msg.format(expobj=self))
        return 1 
    
    def _validate_feature_in_data(self, feature):
        """Validate if a feature exists in the data."""
        if feature not in self.data.columns:
            raise ValueError(f"Feature '{feature}' not found in the data.")

    def _create_binned_feature(self, feature_data, bin_count, labels, method):
        """Create a binned feature based on the specified method."""
        if method == 'quantile':
            return pd.qcut(feature_data, bin_count, labels=labels)
        elif method == 'uniform':
            return pd.cut(feature_data, bin_count, labels=labels)
        else:
            raise ValueError("Invalid method. Choose 'quantile' or 'uniform'.")

Features.__doc__ = """\
A class for processing and transforming dataset features. Provides a 
comprehensive suite of methods for feature normalization, standardization,
handling missing values, encoding categorical features, feature selection,
extraction, correlation analysis, feature interaction, binning, clustering,
and feature extraction from text and images. 
Designed to work seamlessly with pandas DataFrames, facilitating streamlined 
preprocessing workflows for machine learning tasks.

Parameters
----------
features : `List[str]`, optional
    List of features to process. If `None`, all features in the dataset will be 
    considered during processing.
tnames : `Union[str, List[str]]`, optional
    Name(s) of the target variable(s). Used to separate features from targets in 
    supervised tasks.
verbose : `bool`, optional
    Enables verbose output during processing.

Attributes
----------
numeric_features_ : `List[str]`
    List of numeric feature names identified in the dataset.
categorical_features_ : `List[str]`
    List of categorical feature names identified in the dataset.
data : `pd.DataFrame`
    A copy of the processed dataset.

Methods
-------
fit(X, y=None)
    Fit the processor to the data.
normalize(numeric_features=None)
    Normalize specified numeric features or all numeric features.
standardize(numeric_features=None)
    Standardize specified numeric features or all numeric features.
handle_missing(strategy='mean', missing_values=`np.nan`)
    Handle missing values in the data.
encode_categorical(features=None, method='onehot')
    Encode categorical features.
selection(y=None, k=10, method='chi2')
    Select top K features based on a specified selection method.
extraction(n_components=2, **pca_kws)
    Perform feature extraction using Principal Component Analysis (PCA).
correlation(threshold=0.8, drop_correlated_features=True, method='pearson')
    Perform correlation analysis and identify highly correlated features.
ts_features(datetime_column, drop_original_column=True, include=None)
    Extract time series features from a specified datetime column.
interaction(combinations, method='multiply', drop_original=False)
    Create new features by interacting existing features.
binning(features, bins, labels=None, method='quantile', drop_original=False)
    Perform binning or discretization on numeric features.
clustering(n_clusters, features=None, new_feature_name='cluster', drop_original=False)
    Perform clustering on features and add cluster labels as a new feature.
text_extraction(text_column, method='tfidf', max_features=100, **kwargs)
    Extract features from text data using specified text analysis techniques.
image_extraction(image_column, method='hog', **kwargs)
    Extract features from image data using specified techniques.

Formulation
-----------
- **Normalization**: Scales features to a range [0, 1] using the formula:

  .. math::
      X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}

- **Standardization**: Centers features around zero with unit variance using:

  .. math::
      X_{\text{std}} = \frac{X - \mu}{\sigma}

Examples
--------
>>> from gofast.dataops.preprocessing import Features
>>> import pandas as pd
>>> import numpy as np

# Initialize and fit the processor
>>> data = pd.DataFrame({
...     'age': [25, 32, 40, np.nan],
...     'gender': ['M', 'F', 'M', 'F'],
...     'income': [50000, 60000, 55000, 58000]
... })
>>> processor = Features(features=['age', 'gender', 'income'], tnames='income')
>>> processor.fit(data)

# Normalize numeric features
>>> processor.normalize()
>>> print(processor.data)

# Encode categorical features
>>> processor.encode_categorical(features=['gender'], method='onehot')
>>> print(processor.data)

# Handle missing values
>>> processor.handle_missing(strategy='mean')
>>> print(processor.data)

# Feature selection
>>> processor.selection(y=processor.y, k=2, method='chi2')
>>> print(processor.data)

Notes
-----
- The `fit` method must be called before performing any transformations to 
  initialize and identify feature types.
- Methods like `normalize`, `standardize`, and `handle_missing` modify the 
  dataset in place and return the processor instance for method chaining.
- The `encode_categorical` method supports both 'onehot' and 'label' encoding 
  methods, enabling flexibility based on the modeling requirements.
- Feature selection methods require a target variable (`y`) for supervised 
  feature selection techniques like 'chi2'.
- The `image_extraction` method requires the `scikit-image` package when using 
  the 'hog' method.

See Also
--------
MergeableSeries, Frames, MergeableFrames, Missing

References
----------
.. [1] Doe, J. (2023). *Advanced Data Operations with Pandas*. Data Science 
       Publishing.
.. [2] Smith, A. (2022). *DataFrame Manipulations in Python*. Python Data 
       Publishing.
.. [3] Johnson, L. (2020). *Handling Missing Data in Pandas*. Data Science 
       Essentials.
.. [4] Lee, B. (2021). *Logical Operations in DataFrames*. Data Analysis 
       Journal.
"""

@Dataify (auto_columns=True)
def transform_dates(
    data: DataFrame, /, 
    transform: bool = True, 
    fmt: Optional[str] = None, 
    return_dt_columns: bool = False,
    include_columns: Optional[List[str]] = None,
    exclude_columns: Optional[List[str]] = None,
    force: bool = False,
    errors: str = 'coerce',
    **dt_kws
) -> Union[pd.DataFrame, List[str]]:
    """
    Detects and optionally transforms columns in a DataFrame that can be 
    interpreted as dates. 
    
    Funtion uses advanced parameters for greater control over the 
    conversion process.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to inspect and process.
    transform : bool, optional
        Determines whether to perform the conversion of detected datetime 
        columns. Defaults to True.
    fmt : str or None, optional
        Specifies the datetime format string to use for conversion. If None, 
        pandas will infer the format.
    return_dt_columns : bool, optional
        If True, the function returns a list of column names detected 
        (and potentially converted) as datetime. 
        Otherwise, it returns the modified DataFrame or the original DataFrame
        if no transformation is performed.
    include_columns : List[str] or None, optional
        Specifies a list of column names to consider for datetime conversion. 
        If None, all columns are considered.
    exclude_columns : List[str] or None, optional
        Specifies a list of column names to exclude from datetime conversion. 
        This parameter is ignored if `include_columns` is provided.
    force : bool, optional
        If True, forces the conversion of columns to datetime objects, even 
        for columns with mixed or unexpected data types.
    errors : str, optional
        Determines how to handle conversion errors. Options are 'raise', 
        'coerce', and 'ignore' (default is 'coerce').
        'raise' will raise an exception for any errors, 'coerce' will convert 
        problematic data to NaT, and 'ignore' will
        return the original data without conversion.
    **dt_kws : dict
        Additional keyword arguments to be passed to `pd.to_datetime`.

    Returns
    -------
    Union[pd.DataFrame, List[str]]
        Depending on `return_dt_columns`, returns either a list of column names 
        detected as datetime or the DataFrame with the datetime conversions 
        applied.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import transform_dates
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> transform_dates(data, fmt='%Y-%m-%d', return_dt_columns=True)
    ['date', 'timestamp']

    >>> transform_dates(data, include_columns=['date', 'timestamp'], 
    ...                    errors='ignore').dtypes
    date          datetime64[ns]
    value                  int64
    timestamp     datetime64[ns]
    text                  object
    dtype: object
    """
    # Use the helper function to identify potential datetime columns
    potential_dt_columns = detect_datetime_columns(data)
    
    # Filter columns based on include/exclude lists if provided
    if include_columns is not None:
        datetime_columns = [col for col in include_columns 
                            if col in potential_dt_columns]
    elif exclude_columns is not None:
        datetime_columns = [col for col in potential_dt_columns 
                            if col not in exclude_columns]
    else:
        datetime_columns = potential_dt_columns

    df = data.copy()
    
    if transform:
        for col in datetime_columns:
            if force or col in datetime_columns:
                df[col] = pd.to_datetime(df[col], format=fmt,
                                         errors=errors, **dt_kws)
    
    if return_dt_columns:
        return datetime_columns
    
    return df
    
def detect_datetime_columns(data: DataFrame, / ) -> List[str]:
    """
    Detects columns in a DataFrame that can be interpreted as date and time,
    with an improved check to avoid false positives on purely numeric columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to inspect.

    Returns
    -------
    List[str]
        A list of column names that can potentially be formatted as datetime objects.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import detect_datetime_columns
    >>> data = pd.DataFrame({
    ...     'date': ['2021-01-01', '2021-01-02'],
    ...     'value': [1, 2],
    ...     'timestamp': ['2021-01-01 12:00:00', None],
    ...     'text': ['Some text', 'More text']
    ... })
    >>> detect_datetime_columns(data)
    ['date', 'timestamp']
    """
    datetime_columns = []
    is_frame (data, df_only=True, raise_exception= True, objname='Data' )
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].dropna().empty:
            # Skip numeric columns with no values, as they cannot be dates
            continue
        if pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
            try:
                # Attempt conversion on columns with string-like or mixed types
                _ = pd.to_datetime(data[col], errors='raise')
                datetime_columns.append(col)
            except (ValueError, TypeError):
                # If conversion fails, skip the column.
                continue

    return datetime_columns

def boxcox_transformation(
    data: DataFrame, 
    columns: Optional[Union[str, List[str]]] = None, 
    min_value: float = 1, 
    adjust_non_positive: str = 'skip',
    verbose: int = 0, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> (DataFrame, Dict[str, Optional[float]]):
    """
    Apply Box-Cox transformation to each numeric column of a pandas DataFrame.
    
    The Box-Cox transformation can only be applied to positive data. This function
    offers the option to adjust columns with non-positive values by either 
    skipping those columns or adding a constant to make all values positive 
    before applying the transformation.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing numeric data to transform. Non-numeric columns 
        will be ignored.
    columns : str or list of str, optional
        List of column names to apply the Box-Cox transformation to. If None, 
        the transformation is applied to all numeric columns in the DataFrame.
    min_value : float or int, optional
        The minimum value to be considered positive. Default is 1. Values in 
        columns must be greater than this minimum to apply the Box-Cox 
        transformation.
    adjust_non_positive : {'skip', 'adjust'}, optional
        Determines how to handle columns with values <= min_value:
            - 'skip': Skip the transformation for these columns.
            - 'adjust': Add a constant to all elements in these columns to 
              make them > min_value before applying the transformation.
    verbose : int, optional
        Verbosity mode. 0 = silent, 1 = print messages about columns being 
        skipped or adjusted.
        
    view : bool, optional
        If True, displays visualizations of the data before and after 
        transformation.
    cmap : str, optional
        The colormap for visualizing the data distributions. Default is 
        'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualizations. Default is (12, 5).

    Returns
    -------
    transformed_data : pandas.DataFrame
        The DataFrame after applying the Box-Cox transformation to eligible 
        columns.
    lambda_values : dict
        A dictionary mapping column names to the lambda value used for the 
        Box-Cox transformation. Columns that were skipped or not numeric will 
        have a lambda value of None.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.dataops.preprocessing import boxcox_transformation
    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({
    ...     'A': np.random.rand(10) * 100,
    ...     'B': np.random.normal(loc=50, scale=10, size=10),
    ...     'C': np.random.randint(1, 10, size=10)  # Ensure positive values for example
    ... })
    >>> transformed_data, lambda_values = boxcox_transformation(
    ...     data, columns=['A', 'B'], adjust_non_positive='adjust', verbose=1)
    >>> print(transformed_data.head())
    >>> print(lambda_values)
    """
    is_frame (data, df_only=True, raise_exception= True,
              objname='Boxcox transformation' )
    
    transformed_data = pd.DataFrame()
    lambda_values = {}
    verbosity_texts = {}

    if columns is not None:
        columns = is_iterable(columns, exclude_string=True, transform= True )
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            verbosity_texts['missing_columns']=(
                f" Columns {missing_cols} not found in DataFrame."
                 " Skipping these columns.")
        columns = [col for col in columns if col in data.columns]

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    columns_to_transform = columns if columns is not None else numeric_columns
    verbosity_texts['columns_to_transform']=(
        f"Transformed columns: {list(columns_to_transform)}")
    
    # validate adjust_non_positive parameter
    adjust_non_positive=parameter_validator(
        "adjust_non_positive", ["adjust", "skip"], 
        error_msg= (
                "`adjust_non_positive` argument expects ['skip', 'adjust']"
               f" Got: {adjust_non_positive!r}")
        ) (adjust_non_positive)

    skipped_num_columns=[]
    skipped_non_num_columns =[]
    for column in columns_to_transform:
        if column in numeric_columns:
            col_data = data[column]
            if adjust_non_positive == 'adjust' and (col_data <= min_value).any():
                adjustment = min_value - col_data.min() + 1
                col_data += adjustment
                transformed, fitted_lambda = stats.boxcox(col_data)
            elif (col_data > min_value).all():
                transformed, fitted_lambda = stats.boxcox(col_data)
            else:
                transformed = col_data.copy()
                fitted_lambda = None
                skipped_num_columns.append (column)
                
            transformed_data[column] = transformed
            lambda_values[column] = fitted_lambda
        else:
            skipped_non_num_columns.append (column)
            
    if skipped_num_columns: 
        verbosity_texts['skipped_columns']=(
            f"Column(s) '{skipped_num_columns}' skipped: contains values <= {min_value}.")
    if skipped_non_num_columns: 
        verbosity_texts['non_numeric_columns']=(
            f"Column(s) '{skipped_non_num_columns}' is not numeric and will be skipped.")

    # Include non-transformed columns in the returned DataFrame
    for column in data.columns:
        if column not in transformed_data:
            transformed_data[column] = data[column]
            
    if view:
        # Initialize a flag to check the presence of valid data for heatmaps
        valid_data_exists = True
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(fig_size[0] * 2, fig_size[1]))
        
        # Determine columns suitable for the heatmap
        heatmap_columns = columns if columns else data.select_dtypes(
            include=[np.number]).columns
        heatmap_columns = [col for col in heatmap_columns if col in data.columns 
                           and np.issubdtype(data[col].dtype, np.number)]
        
        # Check for non-NaN data in original dataset and plot heatmap
        if heatmap_columns and not data[heatmap_columns].isnull().all().all():
            sns.heatmap(data[heatmap_columns].dropna(axis=1, how='all').corr(), ax=axs[0],
                        annot=True, cmap=cmap)
            axs[0].set_title('Correlation Matrix Before Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['before_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix Before Transformation') 
        
        # Verify transformed_data's structure and plot its heatmap
        if 'transformed_data' in locals() and not transformed_data[heatmap_columns].isnull(
                ).all().all():
            sns.heatmap(transformed_data[heatmap_columns].dropna(
                axis=1, how='all').corr(), ax=axs[1], annot=True, cmap=cmap)
            axs[1].set_title('Correlation Matrix After Transformation')
        else:
            valid_data_exists = False
            verbosity_texts['after_transformed_data_status'] = ( 
                'No valid data for Correlation Matrix After Transformation') 
        
        # Display the plots if valid data exists; otherwise, close the plot
        # to avoid displaying empty figures
        if valid_data_exists:
            plt.tight_layout()
            plt.show()
        else:
            verbosity_texts['matplotlib_window_status']='Closed'
            plt.close()  # Closes the matplotlib window if no valid data is present
    
    # Print verbose recommendations if any and if verbose mode is enabled
    if verbose and verbosity_texts:
        recommendations = ReportFactory('BoxCox Transformation').add_recommendations(
            verbosity_texts, max_char_text=TW)
        print(recommendations)
        
    return transformed_data, lambda_values


@DataTransformer('data', mode='lazy')
@DynamicMethod(
    'both', 
    capture_columns=True, 
)
def transform(
    data: DataFrame, 
    target_columns: Optional[str|List[str]] = None, 
    columns: Optional[str|List[str]] = None, 
    decode_sparse_matrix: bool = True, 
    noise_level: float = None, 
    error: str = "ignore", 
    seed: int = None, 
    view=None, 
):
    """
    Applies preprocessing transformations to the specified DataFrame, including 
    handling of missing values, feature scaling, encoding categorical variables, 
    and optionally introducing noise to numeric features. Transformations can 
    be selectively applied to specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to undergo preprocessing transformations.
    target_columns : str or list of str, optional
        The name(s) of the column(s) considered as target variable(s). 
        These columns are excluded from transformations to prevent leakage. 
        Default is None, indicating no columns are treated as targets.
    columns : str or list of str, optional
        Specific columns to which transformations should be applied. 
        If None, transformations are applied to all columns excluding 
        `target_columns`. Default is None.
    decode_sparse_matrix : bool, optional, default=True
        If True, the function will decode sparse matrices back into dense format 
        if the data is encoded as sparse matrices.
    noise_level : float, optional, default=None
        The level of noise (as a fraction between 0 and 1) to introduce into 
        the numeric columns. If None, no noise is added.
    error : str, optional, default='ignore'
        The error handling strategy. Default is 'ignore'. If set to 'raise', 
        an exception is raised if any error occurs during processing.
    seed : int, optional, default=None
        Seed for the random number generator, ensuring reproducibility of 
        the noise and other random aspects of preprocessing. Default is None.
    view : Any, optional, default=None
        This parameter is here for API consistency with methods like 
        `go_transform` but does not affect the function behavior. It does
        nothing.

    Returns
    -------
    pandas.DataFrame
        The preprocessed DataFrame with numeric features scaled, missing 
        values imputed, categorical variables encoded, and optionally noise 
        added.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import make_classification
    >>> from gofast.dataops.preprocessing import transform 

    # Generating a synthetic dataset
    >>> X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    >>> data = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 
    ...                                 'feature_4'])
    >>> data['target'] = y

    # Apply base_transform to preprocess features, excluding the target column
    >>> from gofast.dataops.preprocessing import base_transform 
    >>> preprocessed_data = transform(data, target_columns='target', 
    ...                                    noise_level=0.1, seed=42)
    >>> print(preprocessed_data.head())

    Note
    ----
    This function is designed to be flexible, allowing selective preprocessing
    on parts of the DataFrame. It leverages `ColumnTransformer` to efficiently 
    process columns based on their data type. The inclusion of `noise_level` 
    allows for simulating real-world data imperfections. Furthermore, this 
    method ensures that the original dataset structure is preserved, with 
    options for restoring target columns and handling sparse matrices.

    The preprocessing steps include:
    - Handling missing values for both numeric and categorical columns using 
      median imputation for numeric columns and constant imputation for 
      categorical ones.
    - Scaling numeric features using `StandardScaler` for normalization.
    - One-hot encoding for categorical features.
    - Optionally, adding noise to numeric columns to simulate real-world data 
      imperfections.
    - Handling sparse matrices by decoding them back to dense format if 
      specified.

    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from ..tools.coreutils import is_sparse_matrix
    
    # collect the target if target_columns is passed 
    # then concatenated later to build original 
    # dataset.
    target = None 
    np.random.seed(seed)
    target_columns = [target_columns] if isinstance(
        target_columns, str) else target_columns

    if target_columns is not None:
        exist_features(data, features= target_columns, name="Target")
        target = data[target_columns]
        data = data.drop(columns=target_columns, errors='ignore')
    # get original data columns 
    original_columns = list(data.columns)
    # Identify numeric and categorical features for transformation
    numeric_features = data.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(
        include=['object']).columns.tolist()

    # Define transformations for numeric and categorical features
    numeric_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Apply ColumnTransformer to handle each feature type
    preprocessor = ColumnTransformer(transformers=[ 
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    data_processed = preprocessor.fit_transform(data)
    processed_columns = numeric_features + categorical_features 
    try: 
        data_processed = pd.DataFrame(
            data_processed, columns=processed_columns,  
            index=data.index)
    except : 
        data_processed = pd.DataFrame(
            data_processed, index=data.index)

    # Apply noise to non-target numeric columns if specified
    if noise_level is not None:
        noise_level = assert_ratio ( 
            noise_level , bounds=(0, 1), 
            name="noise_level"
        )
        for column in numeric_features:
            noise_mask = np.random.rand(data.shape[0]) < noise_level
            data_processed.loc[noise_mask, column] = np.nan
            
    if is_sparse_matrix(data_processed) and decode_sparse_matrix: 
        data_processed = _decode_sparse_processed_data(
            data_processed, processed_columns= processed_columns, 
            error=error
        )
    
    if target is not None: 
        # try to reconstruct the original dataset.
        original_columns += target_columns 
        data_processed = pd.concat([data_processed, target], axis=1)
        
    if view is not None: 
        warnings.warn(
                "The 'view' parameter is included for API consistency"
                " with methods like 'go_transform' but does not affect"
                " the behavior of the 'transform' function.",
                UserWarning
            )
    try:
        return data_processed[original_columns]
    except:
        return data_processed # if something wrong, return it

def _decode_sparse_processed_data(
        processed_data, 
        processed_columns=None, 
        error="warn"
    ):
    """
    Decodes processed sparse data and converts float categorical 
    features to integer categories.
    
    This helper function performs the following operations:
    1. Decodes string-encoded sparse matrix data back into a dense 
       pandas DataFrame.
    2. Transforms float categorical features into integer 
       categorical features using the `FloatCategoricalToInt` transformer.
    3. Reassigns original column names to the processed DataFrame 
       if provided.
    
    Parameters
    ----------
    processed_data : pd.Series or scipy.sparse.spmatrix
        The processed data to decode. This can be either a pandas Series 
        containing string-encoded sparse matrix data or an actual 
        scipy sparse matrix.
    
    processed_columns : list of str, optional, default=None
        The original column names to assign to the decoded DataFrame. 
        If provided, the function attempts to reassign these names to 
        the processed DataFrame. If not provided or if reassignment 
        fails, the original column names are retained.
    
    Returns
    -------
    pd.DataFrame
        A dense pandas DataFrame with decoded sparse data and transformed 
        categorical features. If decoding fails, returns the original 
        `processed_data`.
    
    Raises
    ------
    ValueError
        If the `processed_data` cannot be decoded into a DataFrame.
    
    Notes
    -----
    - This function assumes that the `FloatCategoricalToInt` transformer 
      is properly implemented and imported from the 
      `gofast.tools.transformers.feature_engineering` module.
    - The function attempts to decode the data and transform categorical 
      features. If any step fails, it gracefully handles the exception 
      and returns the original `processed_data`.
    - Reassigning column names is optional and dependent on the availability 
      and correctness of `processed_columns`.
    
    Examples
    --------
    1. **Decoding a pandas Series with string-encoded sparse data:**
    
        ```python
        import pandas as pd
        from gofast.tools.coreutils import decode_sparse_data, is_sparse_matrix
        from gofast.tools.transformers.feature_engineering import FloatCategoricalToInt
        
        # Sample string-encoded sparse data
        processed_data = pd.Series([
            "(0, 0)\t1.0\n(1, 1)\t2.0\n(2, 2)\t3.0",
            "(0, 1)\t1.5\n(1, 0)\t1.0\n(2, 1)\t2.5"
        ])
        
        # Original column names
        original_columns = ['A', 'B', 'C']
        
        # Decode and transform the data
        decoded_df = _decode_sparse_processed_data(
            processed_data, 
            processed_columns=original_columns
        )
        
        print(decoded_df)
        ```
    
    2. **Decoding a scipy sparse matrix:**
    
        ```python
        import scipy.sparse as sp
        from gofast.tools.transformers.feature_engineering import FloatCategoricalToInt
        
        # Sample scipy sparse matrix
        sparse_matrix = sp.csr_matrix([
            [0, 1, 0],
            [1, 0, 2],
            [0, 3, 0]
        ])
        
        # Decode and transform the data without reassigning column names
        decoded_df = _decode_sparse_processed_data(sparse_matrix)
        
        print(decoded_df)
        ```
    
    References
    ----------
    - **SciPy Sparse Matrices Documentation**:
      https://docs.scipy.org/doc/scipy/reference/sparse.html
    - **pandas Series Documentation**:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    - **FloatCategoricalToInt Transformer**:
      Assumed to be implemented in `gofast.tools.transformers.feature_engineering`.
    """
    from ..tools.coreutils import decode_sparse_data
    from ..transformers.feature_engineering import FloatCategoricalToInt

    try:
        # Attempt to decode the processed data if it is string-encoded
        processed_data = decode_sparse_data(processed_data)
        
        # Transform float categorical features to integer categories
        transformer = FloatCategoricalToInt()
        processed_data = transformer.fit_transform(processed_data)
        
        # Reassign original column names if provided
        if processed_columns is not None:
            try:
                processed_data.columns = processed_columns
            except Exception as e:
                # If reassignment fails, retain existing column names
                msg=f"Warning: Failed to assign processed_columns. {e}"
                if error=="warn": 
                    warnings.warn(msg)
                elif error =="raise": 
                    raise TypeError (msg )
        
    except Exception as decode_exception:
        # Handle any exceptions that occur during decoding or transformation
        msg = f"Warning: Decoding or transformation failed. {decode_exception}"
        if error =='warn': 
            warnings.warn(msg)
        elif error =="raise": 
            raise TypeError(msg)
        # Optionally, you can choose to return the original data or raise an error
        return processed_data
    
    return processed_data

def augment_data(
    X: Union[DataFrame, ArrayLike], 
    y: Optional[Union[Series, np.ndarray]] = None, 
    augmentation_factor: int = 2, 
    shuffle: bool = True
) -> Union[Tuple[Union[DataFrame, ArrayLike], Optional[Union[
    Series, ArrayLike]]], Union[DataFrame, ArrayLike]]:
    """
    Augment a dataset by repeating it with random variations to enhance 
    training diversity.

    This function is useful in scenarios with limited data, helping improve the 
    generalization of machine learning models by creating a more diverse training set.

    Parameters
    ----------
    X : Union[pd.DataFrame, np.ndarray]
        Input data, either as a Pandas DataFrame or a NumPy ndarray.

    y : Optional[Union[pd.Series, np.ndarray]], optional
        Target labels, either as a Pandas Series or a NumPy ndarray. If `None`, 
        the function only processes the `X` data. This is useful in unsupervised 
        learning scenarios where target labels may not be applicable.
    augmentation_factor : int, optional
        The multiplier for data augmentation. Defaults to 2 (doubling the data).
    shuffle : bool, optional
        If True, shuffle the data after augmentation. Defaults to True.

    Returns
    -------
    X_augmented : pd.DataFrame or np.ndarray
        Augmented input data in the same format as `X`.
    y_augmented : pd.Series, np.ndarray, or None
        Augmented target labels in the same format as `y`, if `y` is not None.
        Otherwise, None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if the lengths of `X` and `y` 
        are mismatched when `y` is not None.

    Raises
    ------
    ValueError
        If `augmentation_factor` is less than 1 or if `X` and `y` have mismatched lengths.

    Examples
    --------
    >>> from gofast.dataops.preprocessing import augment_data 
    >>> X, y = np.array([[1, 2], [3, 4]]), np.array([0, 1])
    >>> X_aug, y_aug = augment_data(X, y)
    >>> X_aug.shape, y_aug.shape
    ((4, 2), (4,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> X_aug = augment_data(X, y=None)
    >>> X_aug.shape
    (4, 2)
    """
    from sklearn.utils import shuffle as shuffle_data
    
    if augmentation_factor < 1:
        raise ValueError("Augmentation factor must be at least 1.")

    is_X_df = isinstance(X, pd.DataFrame)
    is_y_series = isinstance(y, pd.Series) if y is not None else False

    if is_X_df:
        # Separating numerical and categorical columns
        num_columns = X.select_dtypes(include=['number']).columns
        cat_columns = X.select_dtypes(exclude=['number']).columns

        # Augment only numerical columns
        X_num = X[num_columns]
        X_num_augmented = np.concatenate([X_num] * augmentation_factor)
        X_num_augmented += np.random.normal(loc=0.0, scale=0.1 * X_num.std(axis=0), 
                                            size=X_num_augmented.shape)
        # Repeat categorical columns without augmentation
        X_cat_augmented = pd.concat([X[cat_columns]] * augmentation_factor
                                    ).reset_index(drop=True)

        # Combine numerical and categorical data
        X_augmented = pd.concat([pd.DataFrame(
            X_num_augmented, columns=num_columns), X_cat_augmented], axis=1)
   
    else:
        # If X is not a DataFrame, it's treated as a numerical array
        X_np = np.asarray(X, dtype= float) 
        X_augmented = np.concatenate([X_np] * augmentation_factor)
        X_augmented += np.random.normal(
            loc=0.0, scale=0.1 * X_np.std(axis=0), size=X_augmented.shape)

    y_np = np.asarray(y) if y is not None else None
    
    # Shuffle if required
    if y_np is not None:
        check_consistent_length(X, y_np )
        y_augmented = np.concatenate([y_np] * augmentation_factor)
        if shuffle:
            X_augmented, y_augmented = shuffle_data(X_augmented, y_augmented)
        if is_y_series:
            y_augmented = pd.Series(y_augmented, name=y.name)
            
        return X_augmented, y_augmented

    else:
        if shuffle:
            X_augmented = shuffle_data(X_augmented)
            
        return X_augmented

def apply_tfidf_vectorization(
    data: DataFrame,/,
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies TF-IDF (Term Frequency-Inverse Document Frequency) vectorization 
    to one or more text columns in a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    TF-IDF method weighs the words based on their occurrence in a document 
    relative to their frequency across all documents, helping to highlight 
    words that are more interesting, i.e., frequent in a document but not 
    across documents.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the TF-IDF vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the TF-IDF features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_tfidf_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 
    ...                                   'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_tfidf_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(text_columns, exclude_string= True, transform=True)
    tfidf_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        column_tfidf_df = _generate_tfidf_features(
            handled_data, max_features, stop_words
            )
        tfidf_features_df = pd.concat(
            [tfidf_features_df, column_tfidf_df], 
            axis=1
            )

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat(
        [data.reset_index(drop=True), tfidf_features_df.reset_index(drop=True)
         ], axis=1
    )

    return prepared_data


@Dataify(auto_columns=True) 
def apply_word_embeddings(
    data: DataFrame,/, 
    text_columns: Union[str, List[str]],
    embedding_file_path: str,
    n_components: int = 50,
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
  ) -> DataFrame:
    """
    Applies word embedding vectorization followed by dimensionality reduction 
    to text columns in a pandas DataFrame.
    
    This process converts text data into a numerical form that captures 
    semantic relationships between words, making it suitable for use in machine
    learning models. The function leverages pre-trained word embeddings 
    (e.g., Word2Vec, GloVe) to represent words in a high-dimensional space and 
    then applies PCA (Principal Component Analysis) to reduce the
    dimensionality of these embeddings to a specified number of components.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to be processed.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data to be 
        vectorized. Can be a single column name or a list of names for multiple
        columns.
    embedding_file_path : str
        The file path to the pre-trained word embeddings. This file should be 
        in a format compatible with Gensim's KeyedVectors, such as Word2Vec's .
        bin format or GloVe's .txt format.
    n_components : int, optional
        The number of dimensions to reduce the word embeddings to using PCA. 
        Defaults to 50, balancing between retaining
        semantic information and ensuring manageability for machine learning models.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. Options are:
        - 'fill': Replace missing values with `fill_value`.
        - 'drop': Remove rows with missing values in any of the specified text columns.
        - 'ignore': Leave missing values as is, which may affect the embedding process.
        Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` 
        if `missing_value_handling` is 'fill'. This can be an empty string 
        (default) or any placeholder text.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True, removing the text
        columns to only include the generated features and original non-text data.

    Returns
    -------
    pd.DataFrame
        A DataFrame consisting of the original DataFrame 
        (minus the text columns if `drop_text_columns` is True) concatenated
        with the dimensionality-reduced word embedding features. The new 
        features are numerical and ready for use in machine learning models.

    Examples
    --------
    Assuming we have a DataFrame `df` with a text column 'reviews' and 
    pre-trained word embeddings stored at 'path/to/embeddings.bin':

    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_word_embeddings
    >>> df = pd.DataFrame({'reviews': [
    ...  'This product is great', 'Terrible customer service', 'Will buy again', 
    ... 'Not worth the price']})
    >>> processed_df = apply_word_embeddings(df,
    ...                                      text_columns='reviews',
    ...                                      embedding_file_path='path/to/embeddings.bin',
    ...                                      n_components=50,
    ...                                      missing_value_handling='fill',
    ...                                      fill_value='[UNK]',
    ...                                      drop_text_columns=True)
    >>> processed_df.head()

    This will create a DataFrame with 50 new columns, each representing a 
    component of the reduced dimensionality word embeddings, ready for further 
    analysis or machine learning.
    """
    embeddings = _load_word_embeddings(embedding_file_path)
    
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]

    all_reduced_embeddings = []

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value
            )
        avg_embeddings = _average_word_embeddings(handled_data, embeddings)
        reduced_embeddings = _reduce_dimensions(avg_embeddings, n_components)
        all_reduced_embeddings.append(reduced_embeddings)

    # Combine reduced embeddings into a DataFrame
    embeddings_df = pd.DataFrame(np.hstack(all_reduced_embeddings))

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True), 
                               embeddings_df.reset_index(drop=True)],
                              axis=1
                              )
    return prepared_data

def _generate_bow_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]
    ) -> pd.DataFrame:
    """
    Generates Bag of Words (BoW) features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the BoW vectorizer. Can be 'english' or a 
        custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the BoW features for the text data.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    bow_vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    bow_features = bow_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'bow_{i}' for i in range(bow_features.shape[1])]
    return pd.DataFrame(bow_features, columns=feature_names, index=text_data.index)

@ensure_pkg("gensim","Word-Embeddings expect 'gensim'to be installed." )
def _load_word_embeddings(embedding_file_path: str):
    """
    Loads pre-trained word embeddings from a file.

    Parameters
    ----------
    embedding_file_path : str
        Path to the file containing the pre-trained word embeddings.

    Returns
    -------
    KeyedVectors
        The loaded word embeddings.
    """
    from gensim.models import KeyedVectors
    embeddings = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)
    
    return embeddings

def _average_word_embeddings(
        text_data: Series, embeddings
    ) -> ArrayLike:
    """
    Generates an average word embedding for each text sample in a Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    embeddings : KeyedVectors
        The pre-trained word embeddings.

    Returns
    -------
    np.ndarray
        An array of averaged word embeddings for the text data.
    """
    def get_embedding(word):
        try:
            return embeddings[word]
        except KeyError:
            return np.zeros(embeddings.vector_size)

    avg_embeddings = text_data.apply(lambda x: np.mean(
        [get_embedding(word) for word in x.split() if word in embeddings],
        axis=0)
        )
    return np.vstack(avg_embeddings)

def _reduce_dimensions(
        embeddings: ArrayLike, n_components: int = 50
        ) -> ArrayLike:
    """
    Reduces the dimensionality of word embeddings using PCA.

    Parameters
    ----------
    embeddings : np.ndarray
        The word embeddings array.
    n_components : int
        The number of dimensions to reduce to.

    Returns
    -------
    np.ndarray
        The dimensionality-reduced word embeddings.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def _handle_missing_values(
        column_data: Series, missing_value_handling: str, fill_value: str = ''
   ) -> Series:
    """
    Handles missing values in a pandas Series according to the specified method.

    Parameters
    ----------
    column_data : pd.Series
        The Series (column) from the DataFrame for which missing values 
        need to be handled.
    missing_value_handling : str
        The method for handling missing values: 'fill', 'drop', or 'ignore'.
    fill_value : str, optional
        The value to use for filling missing values if `missing_value_handling`
        is 'fill'. Defaults to an empty string.

    Returns
    -------
    pd.Series
        The Series with missing values handled according to the specified method.

    Raises
    ------
    ValueError
        If an invalid `missing_value_handling` option is provided.
    """
    if missing_value_handling == 'fill':
        return column_data.fillna(fill_value)
    elif missing_value_handling == 'drop':
        return column_data.dropna()
    elif missing_value_handling == 'ignore':
        return column_data
    else:
        raise ValueError("Invalid missing_value_handling option. Choose"
                         " 'fill', 'drop', or 'ignore'.")
        
def _generate_tfidf_features(
        text_data: pd.Series, max_features: int, 
        stop_words: Union[str, List[str]]) -> pd.DataFrame:
    """
    Generates TF-IDF features for a given text data Series.

    Parameters
    ----------
    text_data : pd.Series
        The Series (column) from the DataFrame containing the text data.
    max_features : int
        The maximum number of features to generate.
    stop_words : Union[str, List[str]]
        The stop words to use for the TF-IDF vectorizer. Can be 'english' 
        or a custom list of stop words.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the TF-IDF features for the text data.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    tfidf_features = tfidf_vectorizer.fit_transform(text_data).toarray()
    feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    return pd.DataFrame(tfidf_features, columns=feature_names, index=text_data.index)

def apply_bow_vectorization(
    data: DataFrame, /, 
    text_columns: Union[str, List[str]],
    max_features: int = 100,
    stop_words: Union[str, List[str]] = 'english',
    missing_value_handling: str = 'fill',
    fill_value: str = '',
    drop_text_columns: bool = True
 ) -> pd.DataFrame:
    """
    Applies Bag of Words (BoW) vectorization to one or more text columns in 
    a pandas DataFrame. 
    
    Function concatenates the resulting features back into the original 
    DataFrame.
    
    Bow is a simpler approach that creates a vocabulary of all the unique 
    words in the dataset and then models each text as a count of the number 
    of times each word appears.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the text data to vectorize.
    text_columns : Union[str, List[str]]
        The name(s) of the column(s) in `data` containing the text data.
    max_features : int, optional
        The maximum number of features to generate. Defaults to 100.
    stop_words : Union[str, List[str]], optional
        The stop words to use for the BoW vectorizer. Can be 'english' or 
        a custom list of stop words. Defaults to 'english'.
    missing_value_handling : str, optional
        Specifies how to handle missing values in `text_columns`. 'fill' will 
        replace them with `fill_value`, 'ignore' will keep them as is, and 
        'drop' will remove rows with missing values. Defaults to 'fill'.
    fill_value : str, optional
        The value to use for replacing missing values in `text_columns` if 
        `missing_value_handling` is 'fill'. Defaults to an empty string.
    drop_text_columns : bool, optional
        Whether to drop the original text columns from the returned DataFrame.
        Defaults to True.

    Returns
    -------
    pd.DataFrame
        The original DataFrame concatenated with the BoW features. The 
        original text column(s) can be optionally dropped.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.dataops.preprocessing import apply_bow_vectorization
    >>> data = pd.DataFrame({
    ...     'message_to_investigators': ['This is a sample message', 'Another sample message', np.nan],
    ...     'additional_notes': ['Note one', np.nan, 'Note three']
    ... })
    >>> processed_data = apply_bow_vectorization(
    ... data, text_columns=['message_to_investigators', 'additional_notes'])
    >>> processed_data.head()
    """
    is_frame(data, df_only= True, raise_exception= True )
    text_columns = is_iterable(
        text_columns, exclude_string= True, transform=True)

    bow_features_df = pd.DataFrame()

    for column in text_columns:
        column_data = data[column]
        handled_data = _handle_missing_values(
            column_data, missing_value_handling, fill_value)
        column_bow_df = _generate_bow_features(
            handled_data, max_features, stop_words)
        bow_features_df = pd.concat([bow_features_df, column_bow_df], axis=1)

    if drop_text_columns:
        data = data.drop(columns=text_columns)
    prepared_data = pd.concat([data.reset_index(drop=True),
                               bow_features_df.reset_index(drop=True)],
                              axis=1
                              )

    return prepared_data



    
    
    
    






















