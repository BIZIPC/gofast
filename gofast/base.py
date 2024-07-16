# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
:mod:`gofast.base` module offers core classes and utilities for data handling 
and preprocessing. It includes functionality for managing missing data, 
merging data frames and series, and processing features and targets for 
machine learning tasks.
"""

from __future__ import annotations
import re
import copy
from warnings import warn
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .api.docstring import DocstringComponents, _core_docs
from ._gofastlog import gofastlog
from .api.types import List, Optional, DataFrame, Tuple

from .exceptions import NotFittedError
from .tools.baseutils import is_readable
from .tools.coreutils import sanitize_frame_cols, exist_features
from .tools.coreutils import repr_callable_obj, reshape 
from .tools.coreutils import smart_strobj_recognition, is_iterable
from .tools.coreutils import format_to_datetime, fancier_repr_formatter
from .tools.coreutils import to_numeric_dtypes
from .tools.funcutils import ensure_pkg
from .tools.validator import array_to_frame, check_array, build_data_if
from .tools.validator import is_time_series, is_frame, _is_arraylike_1d  

_logger = gofastlog().get_gofast_logger(__name__)


__all__ = [
    "Data", 
    "MissingHandler",
    "MergeableSeries",
    "MergeableFrames",
    "FrameOperations",
    "FeatureProcessor",
    "TargetProcessor"
   ]

class TargetProcessor:
    """
    A comprehensive class for processing and transforming target variables 
    in datasets.
    
    TargetProcessor handles a range of operations for both single and multi-label 
    target datasets. These operations include encoding, binarizing, normalizing, 
    balancing imbalanced data, calculating metrics, and visualization.

    Parameters
    ----------
    tnames : list of str or str, optional
        Names of the target columns in a DataFrame. If the target is a separate 
        array, this can be omitted. Default is None.
    verbose : bool, optional
        If True, the processor will print additional information during processing.
        Default is False.

    Attributes
    ----------
    multi_label_ : bool
        Indicates whether the target is multi-label. Automatically determined 
        based  on the input data.
    target_ : array-like
        The processed target variable after applying various transformations.
    classes_ : array-like
        Unique classes or labels present in the target variable.
    
    Methods
    -------
    fit(y, X=None):
        Fits the processor to the target variable. Extracts target from a 
        DataFrame if 'X' is provided and 'y' is a column name in 'X'.
    label_encode():
        Encodes class labels to integers. Suitable for classification tasks.
    one_hot_encode():
        Applies one-hot encoding to class labels, returning a binary matrix 
        representation.
    binarize(threshold=0.5):
        Binarizes the target variable based on a specified threshold.
    scale_target():
        Normalizes continuous target variables, typically for regression tasks.
    balance_data(method='smote'):
        Balances the target data using techniques like SMOTE, oversampling, or
        undersampling.
    calculate_metrics(y_pred, metrics=['accuracy', 'precision', 'recall', 'f1']):
        Calculates various performance metrics for the target variable.
    visualize_distribution(plot_type='distribution'):
        Visualizes the distribution or relationship of class labels in the 
        target variable.
    adjust_for_cost_sensitive_learning():
        Adjusts the model for cost-sensitive learning, giving more importance 
        to certain classes.
    analyze_feature_correlation(X):
        Analyzes the correlation of features with the target variable.
    transform_multi_label(method='binary_relevance'):
        Transforms a multi-label problem into a single-label problem using 
        various techniques.
    threshold_tuning():
        Adjusts the decision thresholds for classification tasks.
    visualization(plot_type):
        Creates plots for visualizing different aspects of the target variable.

    Examples
    --------
    >>> from gofast.base import TargetProcessor
    >>> import pandas as pd
    >>> data = pd.DataFrame({'feature': np.random.rand(100), 
                             'target': np.random.choice(['A', 'B', 'C'], 100)})
    >>> processor = TargetProcessor()
    >>> processor.fit('target', X=data).label_encode()
    >>> print(processor.target_)
    """

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
        self : TargetProcessor
            The fitted processor instance.

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame when `y` is a column name.
        ValueError
            If the target names do not exist in the provided DataFrame `X`.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> import pandas as pd
        >>> data = pd.DataFrame({'feature': np.random.rand(100), 
                                 'target': np.random.choice(['A', 'B', 'C'], 100)})
        >>> processor = TargetProcessor()
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
        self : TargetProcessor
            The processor instance after label encoding.

        Raises
        ------
        NotImplementedError
            If the method is called on a multi-label target, which is not supported.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> import numpy as np
        >>> y = np.array(['cat', 'dog', 'bird', 'dog', 'cat'])
        >>> processor = TargetProcessor()
        >>> processor.fit(y).label_encode()
        >>> print(processor.target_)
        """
        self.inspect  # Ensure the method is fitted
        from sklearn.preprocessing import LabelEncoder
        # Check for multi-label target
        if self.multi_label_:
            raise NotImplementedError("Label encoding for multi-label data is not supported.")

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
        self : TargetProcessor
            The processor instance after applying one-hot encoding.

        Raises
        ------
        ValueError
            If the target variable is not set before calling this method.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> import numpy as np
        >>> y = np.array(['red', 'green', 'blue', 'green'])
        >>> processor = TargetProcessor()
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
        self : TargetProcessor
            The processor instance after applying the encoding.

        Raises
        ------
        ValueError
            If an invalid encoding type is specified for multi-label data.
        NotImplementedError
            If label encoding is attempted on multi-label data.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> y = np.array(['cat', 'dog', 'fish'])
        >>> processor = TargetProcessor()
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
        self : TargetProcessor
            The processor instance after applying the binarization.

        Raises
        ------
        NotImplementedError
            If the target is multi-label and `multi_label_binarize` 
            is not set to True.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> y = np.array([0.2, 0.8, 0.4, 0.9])
        >>> processor = TargetProcessor()
        >>> processor.fit(y).binarize(threshold=0.5)
        >>> print(processor.target_)
        """
        from sklearn.preprocessing import Binarizer
        self.inspect   # Ensure the TargetProcessor is fitted
        
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

    def scale_target(self, method='standardize'):
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
        self : TargetProcessor
            The processor instance after applying the scaling.

        Raises
        ------
        ValueError
            If an invalid scaling method is specified or if the method is
            applied to multi-label targets.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> y = np.random.randn(100)  # Sample continuous target
        >>> processor = TargetProcessor()
        >>> processor.fit(y).scale_target(method='normalize')
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

    def split_train_test(self, X, test_size=0.3, stratify=True):
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
        >>> from gofast.base import TargetProcessor
        >>> X, y = # Load your dataset here
        >>> processor = TargetProcessor()
        >>> processor.fit(y, X=X)
        >>> X_train, X_test, y_train, y_test = processor.split_train_test(X, test_size=0.2)

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

    def balance_data(
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
        self or (self, X_resampled) : TargetProcessor, (TargetProcessor, array-like)
            Processor instance after balancing the data, and optionally the 
            resampled feature matrix.

        Raises
        ------
        ValueError
            If an invalid strategy or multi-label handling method is specified.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> X, y = # Load your imbalanced dataset here
        >>> processor = TargetProcessor()
        >>> processor.fit(y, X=X).balance_data(X, method='smote',
        ...                                       return_resampled=True)
        >>> print(processor.target_.shape)
        
        More About the Method Parameters 
        ----------------------------------

        The `method` parameter in the `balance_data` method of 
        `TargetProcessor` defines the strategy used to balance the target 
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
        >>> from gofast.base import TargetProcessor
        >>> y_true = [0, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0]
        >>> processor = TargetProcessor()
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

    def adjust_for_cost_sensitive_learning(self, class_weights=None):
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
        self : TargetProcessor
            The processor instance after adjusting for cost-sensitive learning.

        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> y = np.array([0, 1, 0, 1, 1])
        >>> processor = TargetProcessor()
        >>> processor.fit(y).adjust_for_cost_sensitive_learning({'balanced'})
        >>> print(processor.target_)

        Notes
        -----
        - The class_weights parameter can be crucial in handling imbalanced datasets.
        - In multi-label scenarios, handling class imbalance can be complex and might
          require custom strategies.
          
        The weights are then stored in the class_weights_ attribute of the 
        `TargetProcessor`. It's important to note that these weights are not 
        directly applied to the target array as it would transform the target 
        variable itself, which is not the typical use case. Instead, these 
        weights are usually used during the training of machine learning models 
        to adjust their learning according to the class importance. The method 
        is highly beneficial for addressing class imbalance issues in datasets,
        especially for classification tasks.
        """
        self.inspect # Ensure the TargetProcessor is fitted.

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

    def analyze_feature_correlation(self, X, method='pearson'):
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
        >>> from gofast.base import TargetProcessor
        X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1], 100)
        })
        >>> processor = TargetProcessor()
        >>> processor.fit('target', X=X)
        >>> correlation_dict = processor.analyze_feature_correlation(X)
        >>> print(correlation_dict)

        Notes
        -----
        - The choice of correlation method depends on the nature of the data.
          'pearson' is commonly used for linear relationships, while 'spearman' and
          'kendall' are used for non-linear relationships.
        - In multi-label scenarios, correlation is calculated for each label separately.
        """
        self.inspect # Ensure the TargetProcessor is fitted.
        
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
        self or transformed_target : TargetProcessor or ndarray
            If return_transformed is False, returns the processor instance 
            after transforming the multi labels.
            If return_transformed is True, returns the transformed target data.
        Examples
        --------
        >>> from gofast.base import TargetProcessor
        >>> y = # Your multi-label target array
        >>> processor = TargetProcessor()
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

        self.inspect # Ensure the TargetProcessor is fitted with multi-label data.

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
        >>> from gofast.base import TargetProcessor
        >>> y_scores = np.array([0.3, 0.7, 0.4, 0.6])
        >>> processor = TargetProcessor()
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
        >>> from gofast.base import TargetProcessor
        >>> processor = TargetProcessor()
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
        Raises `NotFittedError` if `TargetProcessor` is not fitted yet."""

        msg = ("{dobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )

        if not hasattr (self, "target_"):
            raise NotFittedError(msg.format(
                dobj=self)
            )
        return 1
    
    # def __getattr__(self, name ): 
    #     return generic_getattr(self, name) 
        
    def __repr__(self): 
        return fancier_repr_formatter(self, max_attrs= 7 )
    
class FeatureProcessor:
    """
    A class for processing and transforming dataset features.

    This class offers methods for various feature operations including
    normalization, standardization, handling missing values, encoding categorical
    features, feature selection, extraction, and more. It's designed to work
    with pandas DataFrames.

    Parameters
    ----------
    features : list of str, optional
        List of features to process. If None, all features in the dataset
        will be considered during processing.
    tnames : str or list of str, optional
        Name(s) of the target variable(s). Used to separate features from
        targets in supervised tasks.
    verbose : bool, optional
        Enables verbose output during processing.

    Attributes
    ----------
    numeric_features_ : list of str
        List of numeric feature names identified in the dataset.
    categorical_features_ : list of str
        List of categorical feature names identified in the dataset.
    data : DataFrame
        A copy of the processed dataset.

    Methods
    -------
    fit(X, y=None):
        Fit the processor to the data.
    normalize(numeric_features=None):
        Normalize specified numeric features or all numeric features.
    standardize(numeric_features=None):
        Standardize specified numeric features or all numeric features.
    handle_missing_values(strategy='mean', missing_values=np.nan):
        Handle missing values in the data.
    encode_categorical(features=None, method='onehot'):
        Encode categorical features.
    feature_selection(y=None, k=10, method='chi2'):
        Select top K features based on a specified selection method.
    feature_extraction(n_components=2, **pca_kws):
        Perform feature extraction using Principal Component Analysis (PCA).
    correlation_analysis(threshold=0.8):
        Perform correlation analysis and identify highly correlated features.
    feature_interaction(combinations, method='multiply', drop_original=False):
        Create new features by interacting existing features.
    binning(numeric_features=None, bins=5, drop_original=True):
        Perform binning or discretization on numeric features.
    feature_clustering(n_clusters, features=None, new_feature_name='cluster'):
        Perform clustering on features and add cluster labels as a new feature.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': ['a', 'b', 'a', 'b']})
    >>> processor = FeatureProcessor()
    >>> processor.fit(data)
    >>> data_normalized = processor.normalize()
    >>> data_encoded = processor.encode_categorical(['B'])
    """
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
        self : FeatureProcessor
            The fitted processor instance.
    
        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame after processing.
    
        Examples
        --------
        >>> from gofast.base import FeatureProcessor
        >>> import pandas as pd
        >>> data = pd.DataFrame({'age': [25, 32, 40], 'gender': ['M', 'F', 'M']})
        >>> processor = FeatureProcessor(features=['age', 'gender'])
        >>> processor.fit(data)
        >>> print(processor.numeric_features_, processor.categorical_features_)
    
        Notes
        -----
        The method converts input data into a DataFrame, extracts the target variable
        if provided, and separates numeric and categorical features. This setup
        facilitates various feature processing tasks that may follow.
        """
        from .tools.mlutils import ( 
            bi_selector,
            build_data_if, 
            select_features,
            get_target
        )
   
    
        # Ensure input data is a DataFrame
        X = build_data_if(X, columns=self.features, to_frame=True, 
                          input_name='feature_', force=True, raise_warning='mute')
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
        self : FeatureProcessor
            The processor instance with normalized data.

        Raises
        ------
        RuntimeError
            If the method is called before fitting the processor with data.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor
        >>> processor = FeatureProcessor()
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
        self : FeatureProcessor
            The processor instance with standardized data.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        >>> processor.fit(data)
        >>> processor.standardize()
        >>> print(processor.data)
        """
        self.inspect  # Ensure the method is fitted
        from sklearn.preprocessing import StandardScaler 
        numeric_features = numeric_features or self.numeric_features_
        
        if len(numeric_features) == 0:
            warn("Missing numeric features. Standardization cannot be performed.")
            return self 

        scaler = StandardScaler()
        self.data[numeric_features] = scaler.fit_transform(self.data[numeric_features])

        return self

    def handle_missing_values(self, strategy='mean', missing_values=np.nan):
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
        self : FeatureProcessor
            The processor instance with imputed data.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame({'feature1': [1, np.nan, 3], 'feature2': [4, 5, np.nan]})
        >>> processor.fit(data).handle_missing_values(strategy='mean')
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
        self : FeatureProcessor
            The processor instance with encoded data.

        Examples
        --------
        >>> processor = FeatureProcessor()
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

    def feature_selection(self, y=None, k=10, method='chi2'):
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
        self : FeatureProcessor
            The processor instance with selected features.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame(np.random.rand(100, 50))  # 100 samples, 50 features
        >>> processor.fit(data).feature_selection(k=5, method='chi2')
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

    def feature_extraction(self, n_components=2, **pca_kws):
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
        self : FeatureProcessor
            The processor instance with extracted features.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame(np.random.rand(100, 50))  # 100 samples, 50 features
        >>> processor.fit(data).feature_extraction(n_components=3)
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

        pca = PCA(n_components=n_components, **pca_kws)
        self.data = pca.fit_transform(self.data)

        return self

    def correlation_analysis(
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
        self : FeatureProcessor
            The processor instance after the correlation analysis.

        Mathematical Formulas
        ---------------------
        - Pearson: \( r = \frac{\sum [(x_i - \bar{x})(y_i - \bar{y})]}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}} \)
        - Spearman: \( \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} \)
        - Kendall: \( \tau = \frac{\text{Number of concordant pairs - Number of discordant pairs}}{\frac{n(n - 1)}{2}} \)

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame(np.random.rand(100, 10))
        >>> processor.fit(data).correlation_analysis(threshold=0.85, method='spearman')
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

    def time_series_features(
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
        self : FeatureProcessor
            The processor instance with the extracted or formatted features added.

        Examples
        --------
        Extract specific time components:
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame({'datetime': pd.date_range(start='2021-01-01',
        ...                                                   periods=100, freq='D')})
        >>> processor.fit(data).time_series_features('datetime',
                                                     include=['year', 'month', 'day'])

        Format datetime without decomposition:
        >>> processor.fit(data).time_series_features('datetime')

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

    def feature_interaction(
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
        self : FeatureProcessor
            The processor instance after feature interaction.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
        >>> processor.fit(data).feature_interaction([('feature_0', 'feature_1'),
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
        self : FeatureProcessor
            The instance itself, updated with binned features.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame({'value': np.random.rand(100)})
        >>> processor.fit(data).binning('value', bins=5, method='uniform')
        >>> processor = FeatureProcessor()
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

    def feature_clustering(
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
        self : FeatureProcessor
            Instance with an additional clustering feature.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
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

    def text_feature_extraction(
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
        self : FeatureProcessor
            The instance itself, updated with the extracted text features.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = pd.DataFrame({'text': ['sample text data', 'another sample text']})
        >>> processor.fit(data).text_feature_extraction('text', method='tfidf', max_features=50)

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
    def image_feature_extraction(
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
        self : FeatureProcessor
            The instance itself, updated with the extracted image features.

        Examples
        --------
        >>> from gofast.base import FeatureProcessor 
        >>> processor = FeatureProcessor()
        >>> data = # DataFrame containing an image column
        >>> processor.fit(data).image_feature_extraction('image_column', method='hog')

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
        from .tools.mlutils import bi_selector 
        self.features = list (self.data.columns )
        self.numeric_features_, self.categorical_features_ = bi_selector ( 
            self.data )

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `FeatureProcessor` is not fitted yet."""
        
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
            
# +++ add base documentations +++
_base_params = dict(
    axis="""
axis: {0 or 'index', 1 or 'columns'}, default 0
    Determine if rows or columns which contain missing values are 
    removed.
    * 0, or 'index' : Drop rows which contain missing values.
    * 1, or 'columns' : Drop columns which contain missing value.
    Changed in version 1.0.0: Pass tuple or list to drop on multiple 
    axes. Only a single axis is allowed.    
    """,
    columns="""
columns: str or list of str 
    columns to replace which contain the missing data. Can use the axis 
    equals to '1'.
    """,
    name="""
name: str, :attr:`pandas.Series.name`
    A singluar column name. If :class:`pandas.Series` is given, 'name'  
    denotes the attribute of the :class:`pandas.Series`. Preferably `name`
    must correspond to the label name of the target. 
    """,
    sample="""
sample: int, Optional, 
    Number of row to visualize or the limit of the number of sample to be 
    able to see the patterns. This is usefull when data is composed of 
    many rows. Skrunked the data to keep some sample for visualization is 
    recommended.  ``None`` plot all the samples ( or examples) in the data     
    """,
    kind="""
kind: str, Optional 
    type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
    ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
    and :mod:`msno` correlation  visualization respectively: 
        * ``bar`` plot counts the  nonmissing data  using pandas
        *  ``mbar`` use the :mod:`msno` package to count the number 
            of nonmissing data. 
        * dendrogram`` show the clusterings of where the data is missing. 
            leaves that are the same level predict one onother presence 
            (empty of filled). The vertical arms are used to indicate how  
            different cluster are. short arms mean that branch are 
            similar. 
        * ``corr` creates a heat map showing if there are correlations 
            where the data is missing. In this case, it does look like 
            the locations where missing data are corollated.
        * ``None`` is the default vizualisation. It is useful for viewing 
            contiguous area of the missing data which would indicate that 
            the missing data is  not random. The :code:`matrix` function 
            includes a sparkline along the right side. Patterns here would 
            also indicate non-random missing data. It is recommended to limit 
            the number of sample to be able to see the patterns. 
    Any other value will raise an error. 
    """,
    inplace="""
inplace: bool, default False
    Whether to modify the DataFrame rather than creating a new one.    
    """
)

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    base=DocstringComponents(_base_params)
)
# +++ end base documentations +++

class Data:
    def __init__(self, verbose: int = 0):
        self._logging = gofastlog().get_gofast_logger(self.__class__.__name__)
        self.verbose = verbose

    @property
    def data(self):
        """ return verified data """
        return self.data_

    @data.setter
    def data(self, d):
        """ Read and parse the data"""
        self.data_ = is_readable(d)

    @property
    def describe(self):
        """ Get summary stats  as well as see the cound of non-null data.
        Here is the default behaviour of the method i.e. it is to only report  
        on numeric columns. To have have full control, do it manually by 
        yourself. 

        """
        return self.data.describe()

    def fit(self, data: str | DataFrame = None):
        """ Read, assert and fit the data.

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        """

        if data is not None:
            self.data = data
        check_array(
            self.data,
            force_all_finite='allow-nan',
            dtype=object,
            input_name='Data',
            to_frame=True
        )
        # for consistency if not a frame, set to aframe
        self.data = array_to_frame(
            self.data, to_frame=True, input_name='col_', force=True
        )
        data = sanitize_frame_cols(self.data, fill_pattern='_')
        for col in data.columns:
            setattr(self, col, data[col])

        return self

    def shrunk(self,
               columns: list[str],
               data: str | DataFrame = None,
               **kwd
               ):
        """ Reduce the data with importance features

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str 
            Columns or features to keep in the datasets

        kwd: dict, 
        additional keywords arguments from :func:`gofast.tools.mlutils.selectfeatures`

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        """
        self.inspect

        self.data = select_features(
            self.data, features=columns, **kwd)

        return self

    @property
    def inspect(self):
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""

        msg = ("{dobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )

        if self.data_ is None:
            raise NotFittedError(msg.format(
                dobj=self)
            )
        return 1

    @ensure_pkg("pandas_profiling", 
                extra= ("'Data.profilingReport' method uses"
                        " 'pandas-profiling' as a dependency.")
                )
    def profilingReport(self, data: str | DataFrame = None, **kwd):
        """Generate a report in a notebook. 

        It will summarize the types of the columns and allow yuou to view 
        details of quatiles statistics, a histogram, common values and extreme 
        values. 

        Parameters 
        ------------
        data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        Returns 
        ---------
        :class:`Data` instance
            Returns ``self`` for easy method chaining.

        Examples 
        ---------
        >>> from gofast.base import Data 
        >>> Data().fit(data).profilingReport()

        """
        self.inspect
        self.data = data or self.data

        from pandas_profiling import ProfileReport
        return ProfileReport(self.data, **kwd)

    def rename(self,
               data: str | DataFrame = None,
               columns: List[str] = None,
               pattern: Optional[str] = None
               ):
        """ 
        rename columns of the dataframe with columns in lowercase and spaces 
        replaced by underscores. 

        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str, Optional 
            the  specific columns in dataframe to renames. However all columns 
            is put in lowercase. If columns not in dataframe, error raises.  

        pattern: str, Optional, 
            Regular expression pattern to strip the data. By default, the 
            pattern is ``'[ -@*#&+/]'``.

        Return
        -------
        ``self``: :class:`~gofast.base.Data` instance 
            returns ``self`` for easy method chaining.

        """
        pattern = str(pattern)

        if pattern == 'None':
            pattern = r'[ -@*#&+/]'
        regex = re.compile(pattern, flags=re.IGNORECASE)

        if data is not None:
            self.data = data

        self.data.columns = self.data.columns.str.strip()
        if columns is not None:
            exist_features(self.data, columns, 'raise')

        if columns is not None:
            self.data[columns].columns = self.data[columns].columns.str.lower(
            ).map(lambda o: regex.sub('_', o))
        if columns is None:
            self.data.columns = self.data.columns.str.lower().map(
                lambda o: regex.sub('_', o))

        return self

    # XXX TODO # use logical and to quick merge two frames
    def merge(self):
        """ Merge two series whatever the type with operator `&&`. 

        When series as dtype object as non numeric values, dtypes should be 
        change into a object 
        """
        # try :
        #     self.data []
    # __and__= __rand__ = merge

    def drop(
        self,
        labels: list[str | int] = None,
        columns: List[str] = None,
        inplace: bool = False,
        axis: int = 0, **kws
    ):
        """ Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding 
        axis, or by specifying directly index or column names. When using a 
        multi-index, labels on different levels can be removed by specifying 
        the level.

        Parameters 
        -----------
        labels: single label or list-like
            Index or column labels to drop. A tuple will be used as a single 
            label and not treated as a list-like.

        axis: {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') 
            or columns (1 or 'columns').

        columns: single label or list-like
            Alternative to specifying axis 
            (labels, axis=1 is equivalent to columns=labels)
        kws: dict, 
            Additionnal keywords arguments passed to :meth:`pd.DataFrame.drop`.

        Returns 
        ----------
        DataFrame or None
            DataFrame without the removed index or column labels or 
            None if `inplace` equsls to ``True``.

        """
        self.inspect

        data = self.data.drop(labels=labels,  inplace=inplace,
                              columns=columns, axis=axis, **kws)
        return data

    def __repr__(self):
        """ Pretty format for programmer guidance following the API... """
        return repr_callable_obj(self, skip='y')

    def __getattr__(self, name):
        if name.endswith('_'):
            if name not in self.__dict__.keys():
                if name in ('data_', 'X_'):
                    raise NotFittedError(
                        f'Fit the {self.__class__.__name__!r} object first'
                    )

        rv = smart_strobj_recognition(name, self.__dict__, deep=True)
        appender = "" if rv is None else f'. Do you mean {rv!r}'

        raise AttributeError(
            f'{self.__class__.__name__!r} object has no attribute {name!r}'
            f'{appender}{"" if rv is None else "?"}'
        )


Data.__doc__ = """\
Data base class

Typically, we train a model with a matrix of data. Note that pandas Dataframe 
is the most used because it is very nice to have columns lables even though 
Numpy arrays work as well. 

For supervised Learning for instance, suc as regression or clasification, our 
intent is to have a function that transforms features into a label. If we 
were to write this as an algebra formula, it would be look like:
    
.. math::
    
    y = f(X)

:code:`X` is a matrix. Each row represent a `sample` of data or information 
about individual. Every columns in :code:`X` is a `feature`.The output of 
our function, :code:`y`, is a vector that contains labels (for classification)
or values (for regression). 

In Python, by convention, we use the variable name :code:`X` to hold the 
sample data even though the capitalization of variable is a violation of  
standard naming convention (see PEP8). 

Parameters 
-----------
{params.core.data}
{params.base.columns}
{params.base.axis}
{params.base.sample}
{params.base.kind}
{params.base.inplace}
{params.core.verbose}

Returns
-------
{returns.self}
   
Examples
--------
.. include:: ../docs/data.rst

""".format(
    params=_param_docs,
    returns=_core_docs["returns"],
)


class MissingHandler (Data):
    """ Deal with missing values in Data 

    Most algorithms will not work with missing data. Notable exceptions are the 
    recent boosting libraries such as the XGBoost 
    (:doc:`gofast.documentation.xgboost.__doc__`) CatBoost and LightGBM. 
    As with many things in machine learning , there are no hard answaers for how 
    to treat a missing data. Also, missing data could  represent different 
    situations. There are three warious way to handle missing data:: 

        * Remove any row with missing data 
        * Remove any columns with missing data 
        * Impute missing values 
        * Create an indicator columns to indicator data was missing 

    Parameters
    ----------- 
    in_percent: bool, 
        give the statistic of missing data in percentage if ser to ``True``. 

    sample: int, Optional, 
        Number of row to visualize or the limit of the number of sample to be 
        able to see the patterns. This is usefull when data is composed of 
        many rows. Skrunked the data to keep some sample for visualization is 
        recommended.  ``None`` plot all the samples ( or examples) in the data 
    kind: str, Optional 
        type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
        ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
        and :mod:`msno` correlation  visualization respectively: 

            * ``bar`` plot counts the  nonmissing data  using pandas
            *  ``mbar`` use the :mod:`msno` package to count the number 
                of nonmissing data. 
            * dendrogram`` show the clusterings of where the data is missing. 
                leaves that are the same level predict one onother presence 
                (empty of filled). The vertical arms are used to indicate how  
                different cluster are. short arms mean that branch are 
                similar. 
            * ``corr` creates a heat map showing if there are correlations 
                where the data is missing. In this case, it does look like 
                the locations where missing data are corollated.
            * ``None`` is the default vizualisation. It is useful for viewing 
                contiguous area of the missing data which would indicate that 
                the missing data is  not random. The :code:`matrix` function 
                includes a sparkline along the right side. Patterns here would 
                also indicate non-random missing data. It is recommended to limit 
                the number of sample to be able to see the patterns. 

        Any other value will raise an error 

    Examples 
    --------
    >>> from gofast.base import MissingHandler
    >>> data ='data/geodata/main.bagciv.data.csv' 
    >>> ms= Missing().fit(data) 
    >>> ms.plot_.fig_size = (12, 4 ) 
    >>> ms.plot () 

    """

    def __init__(self,
                 in_percent=False,
                 sample=None,
                 kind=None,
                 drop_columns: List[str] = None,
                 **kws):

        self.in_percent = in_percent
        self.kind = kind
        self.sample = sample
        self.drop_columns = drop_columns
        self.isnull_ = None

        super().__init__(**kws)

    @property
    def isnull(self):
        """ Check the mean values  in the data  in percentge"""
        self.isnull_ = self.data.isnull().mean(
        ) * 1e2 if self.in_percent else self.data.isnull().mean()

        return self.isnull_

    def plot(self, figsize: Tuple[int] = None,  **kwd):
        """
        Vizualize patterns in the missing data.

        Parameters 
        ------------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        kind: str, Optional 
            kind of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar`` plot 
            for dendrogram , :mod:`msno` bar and :mod:`plt` visualization 
            respectively: 

                * ``bar`` plot counts the  nonmissing data  using pandas
                *  ``mbar`` use the :mod:`msno` package to count the number 
                    of nonmissing data. 
                * dendrogram`` show the clusterings of where the data is missing. 
                    leaves that are the same level predict one onother presence 
                    (empty of filled). The vertical arms are used to indicate how  
                    different cluster are. short arms mean that branch are 
                    similar. 
                * ``corr` creates a heat map showing if there are correlations 
                    where the data is missing. In this case, it does look like 
                    the locations where missing data are corollated.
                * ``None`` is the default vizualisation. It is useful for viewing 
                    contiguous area of the missing data which would indicate that 
                    the missing data is  not random. The :code:`matrix` function 
                    includes a sparkline along the right side. Patterns here would 
                    also indicate non-random missing data. It is recommended to limit 
                    the number of sample to be able to see the patterns. 

                Any other value will raise an error 

        sample: int, Optional
            Number of row to visualize. This is usefull when data is composed of 
            many rows. Skrunked the data to keep some sample for visualization is 
            recommended.  ``None`` plot all the samples ( or examples) in the data 

        kws: dict 
            Additional keywords arguments of :mod:`msno.matrix` plot. 

        Return
        -------
        ``self``: :class:`~gofast.base.Missing` instance 
            returns ``self`` for easy method chaining.


        Examples 
        --------
        >>> from gofast.base import Missing
        >>> data ='data/geodata/main.bagciv.data.csv' 
        >>> ms= Missing().fit(data) 
        >>> ms.plot(figsize = (12, 4 ) ) 


        """
        self.inspect
        from .plot.explore import QuestPlotter

        QuestPlotter(fig_size=figsize).fit(self.data).plotMissing(
            kind=self.kind, sample=self.sample, **kwd)
        return self

    @property
    def get_missing_columns(self):
        """ return columns with Nan Values """
        return list(self.data.columns[self.data.isna().any()])

    def drop(self,
             data: str | DataFrame = None,
             columns: List[str] = None,
             inplace=False,
             axis=1,
             **kwd
             ):
        """Remove missing data 

        Parameters 
        -----------
        data: Dataframe of shape (M, N) from :class:`pandas.DataFrame` 
            Dataframe containing samples M  and features N

        columns: str or list of str 
            columns to drop which contain the missing data. Can use the axis 
            equals to '1'.

        axis: {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are 
            removed.
            * 0, or 'index' : Drop rows which contain missing values.

            * 1, or 'columns' : Drop columns which contain missing value.
            Changed in version 1.0.0: Pass tuple or list to drop on multiple 
            axes. Only a single axis is allowed.

        how: {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we 
            have at least one NA or all NA.

            * 'any': If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh: int, optional
            Require that many non-NA values. Cannot be combined with how.

        subset: column label or sequence of labels, optional
            Labels along other axis to consider, e.g. if you are dropping rows 
            these would be a list of columns to include.

        inplace: bool, default False
            Whether to modify the DataFrame rather than creating a new one.

        Returns 
        -------
        ``self``: :class:`~gofast.base.Missing` instance 
            returns ``self`` for easy method chaining.

        """
        if data is not None:
            self.data = data

        self.inspect
        if columns is not None:
            self.drop_columns = columns

        exist_features(self.data, self.drop_columns, error='raise')

        if self.drop_columns is None:
            if inplace:
                self.data.dropna(axis=axis, inplace=True, **kwd)
            else:
                self.data = self.data .dropna(
                    axis=axis, inplace=False, **kwd)

        elif self.drop_columns is not None:
            if inplace:
                self.data.drop(columns=self.drop_columns,
                               axis=axis, inplace=True,
                               **kwd)
            else:
                self.data.drop(columns=self.columns, axis=axis,
                               inplace=False, **kwd)

        return self

    @property
    def sanity_check(self):
        """Ensure that we have deal with all missing values. The following 
        code returns a single boolean if there is any cell that is missing 
        in a DataFrame """

        return self.data.isna().any().any()

    @ensure_pkg("pyjanitor", condition = "return_non_null")
    def replace(
        self,
        data: str | DataFrame = None,
        columns: str| List[str] = None,
        fill_value: float = None,
        new_column_name: str = None,
        return_non_null: bool = False,
        **kwargs
    ) -> 'MissingHandler':
        """
        Replace missing values in the DataFrame.

        This method can either fill missing values with a specified value 
        or leverage the `coalesce` function from the `pyjanitor` library to 
        find and return the first non-null value across columns for each row.

        Parameters
        ----------
        data : pd.DataFrame, optional
            DataFrame containing the data. If not provided, uses the 
            instance's data.
        columns : str or List[str], optional
            Columns to consider when replacing missing values. If not provided,
            all columns are considered.
        fill_value : float, optional
            Value to replace missing values with. If not provided, no fill 
            operation is performed.
        new_column_name : str, optional
            Name for the new column generated by coalesce when 
            `return_non_null` is True.
        return_non_null : bool, default False
            If True, uses `pyjanitor.coalesce` to return the first non-null 
            value acrossspecified columns for each row.
        **kwargs
            Additional keyword arguments to pass to `fillna` method if 
            `fill_value` is used.

        Returns
        -------
        self : MissingHandler
            The instance itself for method chaining.

        Examples
        --------
        >>> handler = MissingHandler(data)
        >>> handler.replace(fill_value=0).data
        >>> handler.replace(return_non_null=True, new_column_name='first_non_null').data
        """
        if data is not None:
            self.data = data

        if isinstance(columns, str):
            columns = [columns]

        if return_non_null:
            from pyjanitor import coalesce  
            new_column_name = self._assert_str (new_column_name, "new_column_name")
            self.data = coalesce(
                self.data,
                columns=columns,
                new_column_name=new_column_name
            )
        elif fill_value is not None:
            self.data.fillna(value=fill_value, inplace=True, **kwargs)

        return self

    @staticmethod
    def _assert_str(obj, name: str) -> str:
        """
        Ensure an object is a string, raising a TypeError otherwise.

        Parameters
        ----------
        obj : Any
            The object to check.
        name : str
            The name of the parameter for error messages.

        Returns
        -------
        str
            The object if it is a string.

        Raises
        ------
        TypeError
            If `obj` is not a string.
        """
        if not isinstance(obj, str):
            raise TypeError(f"{name} must be a string.")
        return obj


def select_features(
    df: DataFrame,
    features: List[str] = None,
    include=None,
    exclude=None,
    coerce: bool = False,
    **kwd
):
    """ Select features  and return new dataframe.  

    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieved in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param kwd: additional keywords arguments from `pd.astype` function 

    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    """

    if features is not None:
        exist_features(df, features, error='raise')
    # change the dataype
    df = df.astype(float, errors='ignore', **kwd)
    # assert whether the features are in the data columns
    if features is not None:
        return df[features]
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return df if coerce else df.select_dtypes(include, exclude)

class MergeableSeries:
    """
    A class that wraps a pandas Series to enable logical AND operations
    using the & operator, even for non-numeric data types.
    """

    def __init__(self, series):
        """
        Initialize the MergeableSeries object.

        Parameters
        ----------
        series : pandas.Series
            The pandas Series to be wrapped.
        """
        self.series = series

    def __and__(self, other):
        """
        Overload the & operator to perform logical AND between two Series.

        Parameters
        ----------
        other : MergeableSeries
            Another MergeableSeries object to perform logical AND with.

        Returns
        -------
        pandas.Series
            A new Series containing the result of the logical AND operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableSeries.
        Example 
        --------

        >>> from gofast.base import MergeableSeries 
        >>> s1 = MergeableSeries(pd.Series([True, False, 'non-numeric']))
        >>> s2 = MergeableSeries(pd.Series(['non-numeric', False, True]))
        >>> result = s1 & s2
        >>> print(result)
        """
        if not isinstance(other, MergeableSeries):
            raise ValueError("Operand must be an instance of MergeableSeries")

        # Convert non-numeric types to string for logical operations
        series1 = (self.series.astype(str) if self.series.dtype == 'object'
                   else self.series)
        series2 = (other.series.astype(str) if other.series.dtype == 'object'
                   else other.series)

        # Perform logical AND operation
        return series1 & series2


class FrameOperations:
    """
    A class for performing various operations on pandas DataFrames.

    This class provides methods to merge, concatenate, compare, and
    perform arithmetic operations on two or more pandas DataFrames.

    """

    def __init__(self, ):
        ...

    def fit(self, *frames, **kws):
        """ Inspect frames 

        Parameters 
        -----------
        *frames : pandas.DataFrame
            Variable number of pandas DataFrame objects to be operated on.

        kws: dict, 
           Additional keywords arguments passed to 
           func:`gofast.tools.coreutils.inspect_data`
        Returns 
        ---------
        self: Object for chainings methods. 

        """
        frames = []
        for frame in self.frames:
            frames.append(to_numeric_dtypes(frames, **kws))

        self.frames = frames

        return self

    def merge_frames(self, on, how='inner', **kws):
        """
        Merge two or more DataFrames on a key column.

        Parameters
        ----------
        on : str or list
            Column or index level names to join on. Must be found 
            in both DataFrames.
        how : str, default 'inner'
            Type of merge to be performed. Options include 'left',
            'right', 'outer', 'inner'.

        kws: dict, 
           Additional keyword arguments passed to `pd.merge`
        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the merge operation.

        Examples
        --------
        >>> from gofast.base import FrameOperations
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [2, 3], 'C': [5, 6]})
        >>> df_ops = FrameOperations.fit(df1, df2)
        >>> df_ops.merge_frames(on='A')
        """
        self.inspect 
        result = self.frames[0]
        for df in self.frames[1:]:
            result = pd.merge(result, df, on=on, how=how, **kws)
        return result

    def concat_frames(self, axis=0, **kws):
        """
        Concatenate two or more DataFrames along a particular axis.

        Parameters
        ----------
        axis : {0/'index', 1/'columns'}, default 0
            The axis to concatenate along.

        kws: dict, 
           keywords arguments passed to `pandas.concat`
        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the concatenation.

        Examples
        --------
        >>> from gofast.base import FrameOperations
        >>> df1 = pd.DataFrame({'A': [1, 2]})
        >>> df2 = pd.DataFrame({'B': [3, 4]})
        >>> df_ops = FrameOperations(df1, df2)
        >>> df_ops.concat_frames(axis=1)
        """
        self.inspect 
        return pd.concat(self.dataframes, axis=axis, **kws)

    def compare_frames(self):
        """
        Compare the dataframes for equality.

        Returns
        -------
        bool
            True if all dataframes are equal, False otherwise.

        Examples
        --------
        >>> from gofast.base import FrameOperations
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df_ops = FrameOperations.fit(df1, df2)
        >>> df_ops.compare_frames()
        """
        self.inspect 
        first_df = self.frames[0]
        for df in self.frames[1:]:
            if not first_df.equals(df):
                return False

        return True

    def add_frames(self):
        """
        Perform element-wise addition of two or more DataFrames.

        Returns
        -------
        pandas.DataFrame
            A DataFrame resulting from the element-wise addition.

        Examples
        --------
        >>> from gofast.base import FrameOperations
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        >>> df_ops = FrameOperations.fit(df1, df2)
        >>> df_ops.add_frames()
        """
        self.inspect 
        result = self.frames[0].copy()
        for df in self.frames[1:]:
            result = result.add(df, fill_value=0)
        return result
    
    def conditional_filter(self, conditions):
        """
        Filter the DataFrame based on multiple conditional criteria.
    
        Parameters
        ----------
        conditions : dict
            A dictionary where keys are column names and values are 
            functions that take a single argument and return a boolean.
    
        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
    
        Examples
        --------
        >>> from gofast.base import FrameOperations
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df_ops = FrameOperations.fit(df)
        >>> conditions = {'A': lambda x: x > 1, 'B': lambda x: x < 6}
        >>> df_ops.conditional_filter(conditions)
        """
        self.inspect 
        new_frames=[]
        for frame in self.frames: 
            mask = pd.Series(True, index=frame.index)
            for col, condition in conditions.items():
                mask &= frame[col].apply(condition)
                
            new_frames.append (frame[mask])
                
        return new_frames[0] if len(self.frames)==1 else new_frames
    
    @property
    def inspect(self):
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""

        msg = ("{dobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )

        if self.data_ is None:
            raise NotFittedError(msg.format(
                dobj=self)
            )
        return 1

class MergeableFrames:
    """
    A class that wraps pandas DataFrames to enable logical operations
    (like AND, OR) using bitwise operators on DataFrames.

    This class provides a way to intuitively perform logical operations
    between multiple DataFrames, especially useful for conditional
    filtering and data analysis.
    
    Parameters
    ----------
    frame : pandas.DataFrame
        The pandas DataFrame to be wrapped.
    kws: dict, 
       Additional keyword arguments  to build a frame if the array is passed 
       rather than the dataframe. 

    Attributes
    ----------
    dataframe : pandas.DataFrame
        The pandas DataFrame to be wrapped.

    Methods
    -------
    __and__(self, other)
        Overloads the & operator to perform logical AND between DataFrames.

    __or__(self, other)
        Overloads the | operator to perform logical OR between DataFrames.

    Examples
    --------
    >>> from gofast.base import MergeableFrames
    >>> df1 = pd.DataFrame({'A': [True, False], 'B': [False, True]})
    >>> df2 = pd.DataFrame({'A': [False, True], 'B': [True, False]})
    >>> mergeable_df1 = MergeableFrames(df1)
    >>> mergeable_df2 = MergeableFrames(df2)
    >>> and_result = mergeable_df1 & mergeable_df2
    >>> or_result = mergeable_df1 | mergeable_df2
    """

    def __init__(self, frame, **kws ):

        self.frame = build_data_if(frame, force=True , **kws )

    def __and__(self, other):
        """
        Overload the & operator to perform logical AND between 
        two DataFrames.

        Parameters
        ----------
        other : MergeableFrames
            Another MergeableFrames object to perform logical AND with.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing the result of the logical AND operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableFrames.
        """
        if not isinstance(other, MergeableFrames):
            raise ValueError("Operand must be an instance of MergeableFrames")

        return self.frame & other.frame

    def __or__(self, other):
        """
        Overload the | operator to perform logical OR between two DataFrames.

        Parameters
        ----------
        other : MergeableDataFrames
            Another MergeableDataFrames object to perform logical OR with.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame containing the result of the logical OR operation.

        Raises
        ------
        ValueError
            If 'other' is not an instance of MergeableFrames.
        """
        if not isinstance(other, MergeableFrames):
            raise ValueError("Operand must be an instance of MergeableFrames")

        return self.frame | other.frame


