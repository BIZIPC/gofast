# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Learning utilities for data transformation, model learning, and inspections.
This module provides tools for data preprocessing, model evaluation, feature 
engineering, and utilities for handling machine learning workflows efficiently.
"""

import os
import re
import math
import copy
import tarfile
import pickle
import joblib
import warnings
import shutil
from six.moves import urllib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder, RobustScaler, OrdinalEncoder, StandardScaler,
    MinMaxScaler, LabelBinarizer, LabelEncoder, Normalizer, PolynomialFeatures
)
from sklearn.utils import resample

from .._gofastlog import gofastlog
from ..api.types import List, Tuple, Any, Dict, Optional, Union, Series
from ..api.types import _F, ArrayLike, NDArray, DataFrame, Callable
from ..api.formatter import MetricFormatter
from ..api.summary import ReportFactory, ResultSummary
from ..compat.sklearn import get_feature_names, train_test_split
from ..exceptions import DependencyError
from ..decorators import SmartProcessor
from .baseutils import select_features
from .coreutils import (
    _assert_all_types, is_in_if, ellipsis2false, smart_format, is_iterable,
    get_valid_kwargs, is_classification_task, to_numeric_dtypes,
    validate_feature, exist_features, contains_delimiter, 
    str2columns 
)
from .depsutils import ensure_pkg
from .validator import (
    _is_numeric_dtype, _is_arraylike_1d, get_estimator_name, check_array,
    check_consistent_length, is_frame, build_data_if, check_is_fitted,
    check_mixed_data_types, validate_data_types, _check_consistency_size
)


# Logger Configuration
_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    'bi_selector',
    'bin_counting',
    'build_data_preprocessor',
    'compute_smart_batch_size', 
    'discretize_categories',
    'display_feature_contributions',
    'dynamic_batch_size', 
    'evaluate_model',
    'fetch_model',
    'fetch_tgz',
    'fetch_tgz2',
    'format_model_score',
    'get_correlated_features',
    'get_feature_contributions',
    'get_global_score',
    'get_batch_size', 
    'handle_imbalance',
    'laplace_smoothing',
    'laplace_smoothing_categorical',
    'laplace_smoothing_word',
    'load_model',
    'make_pipe',
    'one_click_prep',
    'process_data_types', 
    'resampling',
    'save_dataframes',
    'select_feature_importances',
    'smart_label_classifier', 
    'smart_split',
    'soft_data_split',
    'soft_encoder',
    'soft_imputer',
    'soft_scaler',
    'stats_from_prediction',
    'stratify_categories'
]

TORCH_DEP_EMSG=(
    "Error: PyTorch is not installed. Please install PyTorch by running:\n"
    "'pip install torch' or follow the installation instructions at:\n"
    "https://pytorch.org/get-started/locally/"
)

TF_DEP_EMSG= (
    "Error: TensorFlow is not installed. Please install TensorFlow "
    "by running:\n 'pip install tensorflow' or visit the installation "
    "guide at:\n https://www.tensorflow.org/install"
)

@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend") in ("tensorflow", "tf")
    )
def get_batch_size(
    data_or_train_loader, *, 
    model=None, 
    device=None,
    initial_batch_size=32, 
    max_batch_size=512,
    default_size=None, 
    backend=None, 
    verbose=0,
):
    """
    Automatically determines the optimal batch size for training based on
    available hardware, model, and backend.

    This function adjusts the batch size by running the model with increasing
    batch sizes until it encounters memory limitations, at which point it will
    fall back to the largest feasible batch size. The backend can be set to 
    either 'torch' (PyTorch) or 'tensorflow' (TensorFlow), and depending on
    the device availability (GPU or CPU), it will adjust the batch size 
    accordingly.

    The process involves testing batch sizes starting from `initial_batch_size`
    and doubling them until either memory limits are reached or `max_batch_size`
    is exceeded. If an error occurs due to lack of memory, the batch size is
    reduced until it is a feasible value. If no GPU is available or there are
    insufficient resources, a fallback batch size is returned.

    Parameters
    ----------
    data_or_train_loader : Dataset or DataLoader
        The dataset or data loader to be used for training. If a DataLoader 
        is provided, the batch size will be determined dynamically. If a 
        raw dataset is provided, it will be wrapped in a DataLoader during 
        the process.
        
    model : torch.nn.Module or tf.keras.Model, optional
        The model to be used for training. If a model is provided, it will
        be used to test the batch size. If no model is provided, the batch 
        size determination will be done based on dataset characteristics.

    device : torch.device or tf.device, optional
        The device (CPU or GPU) where the model is to be trained. This 
        should be provided if using a framework like PyTorch that requires
        explicit device placement.

    initial_batch_size : int, optional
        The starting batch size for testing. This value is used as the 
        initial batch size for determining the optimal batch size. Default 
        is 32.

    max_batch_size : int, optional
        The maximum batch size to attempt. The batch size will never exceed 
        this value, even if the available memory allows for larger sizes. 
        Default is 512.

    default_size : int, optional
        The fallback batch size if the batch size determination process fails 
        or the backend is unsupported. Default is None, which means the function 
        will use `initial_batch_size` as the fallback.

    backend : str, optional
        The backend to use for training. This can either be 'torch' for 
        PyTorch or 'tensorflow' for TensorFlow. Default is 'torch'.

    verbose : int, optional
        The verbosity level. If set to 1, the function will print information 
        about the batch size determination process. Default is 0 (silent).

    Returns
    -------
    int
        The determined batch size that can be used for training.

    Methods
    -------
    - `get_batch_size`: Main function to dynamically determine the batch size.
    - `_use_torch_available`: Checks if PyTorch is available.
    - `_prepare_tf_dataset`: Prepares TensorFlow dataset for testing.
    - `_analyze_dataset`: Analyzes dataset to estimate batch size.
    - `compute_smart_batch_size`: Heuristic computation for batch size based 
      on dataset size and number of features.

    Notes
    -----
    The batch size determination process involves the following steps:

    1. **Test Memory Capacity**:
        The function starts with an initial batch size, `B_0`, and 
        progressively doubles it, checking for memory overflow at each step:
        
        .. math::
            B_{i+1} = 2 \cdot B_i \quad \text{until memory overflows.}
            
    2. **Handling Memory Overflow**:
        Upon encountering memory limitations, the batch size is reduced:
        
        .. math::
            B_{\text{final}} = \max \left( \frac{B_i}{2}, B_{\text{min}} \right)
            
    3. **Final Batch Size**:
        The final batch size is determined by returning either the largest
        feasible batch size or the fallback batch size:
        
        .. math::
            B_{\text{final}} = \min \left( B_{\text{final}}, B_{\text{max}} \right)
            
    - This function requires the PyTorch or TensorFlow library to be installed.
    - If neither GPU nor sufficient memory is available, the function falls back 
      to a heuristic method to determine a reasonable batch size based on dataset 
      characteristics.
    - The `device` argument is only applicable to PyTorch. TensorFlow handles 
      device placement internally.

    Example
    -------
    >>> from gofast.tools.mlutils import get_batch_size
    >>> batch_size = get_batch_size(data, model=model, device=device, 
    ...                             backend='torch', verbose=1)
    >>> print(f"Optimal batch size: {batch_size}")

    See Also
    --------
    - `torch.utils.data.DataLoader`
    - `tensorflow.data.Dataset`
    - `compute_smart_batch_size`
    
    References
    ----------
    .. [1] Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance
           Deep Learning Library," in NeurIPS, 2019.
    .. [2] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning 
          on Heterogeneous Distributed Systems," 2016.
    """
    # Determine batch size for PyTorch backend
    backend= str(backend).lower() 
    if backend== 'torch':
        try:
            import torch
            import torch.utils.data as Tc_data
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG)
            
        if torch.cuda.is_available():
            batch_size = initial_batch_size
            while batch_size <= max_batch_size:
                try:
                    data_loader = Tc_data.DataLoader(
                        data_or_train_loader.dataset,
                        batch_size=batch_size
                    )

                    # Test the batch size by running a single batch
                    for batch in data_loader:
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        model(inputs)
                        break  # Only need to test one batch

                    if verbose:
                        print(f"Batch size {batch_size} fits in memory.")
                    batch_size *= 2  # Double the batch size for next iteration

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()  # Free memory
                        batch_size = max(batch_size // 2, initial_batch_size)
                        if verbose:
                            print(
                                f"Out of memory! Largest batch size: {batch_size}."
                            )
                        return batch_size
                    else:
                        raise e
            return min(batch_size, max_batch_size)
        else:
            if verbose:
                print(
                    "CUDA not available or not enough memory."
                    " Using fallback batch size."
                )
            return default_size if default_size is not None else 32

    # Determine batch size for TensorFlow backend
    elif backend.lower() == 'tensorflow':
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError(TF_DEP_EMSG)

        if tf.config.list_physical_devices('GPU'):
            batch_size = initial_batch_size
            while batch_size <= max_batch_size:
                try:
                    # Prepare dataset for testing the batch size
                    test_ds = prepare_tf_dataset(data_or_train_loader, batch_size)

                    # Run a single batch through the model
                    for inputs, targets in test_ds.take(1):
                        model(inputs)

                    if verbose:
                        print(f"Batch size {batch_size} fits in memory.")
                    batch_size *= 2

                except tf.errors.ResourceExhaustedError:
                    batch_size = max(batch_size // 2, initial_batch_size)
                    if verbose:
                        print(
                            f"Out of memory! Largest batch size: {batch_size}."
                        )
                    return batch_size
            return min(batch_size, max_batch_size)
        else:
            if verbose:
                print(
                    "No GPU available for TensorFlow or insufficient memory."
                    " Using fallback batch size."
                )
            return default_size if default_size is not None else 32

    # Fallback heuristic for other cases (non-GPU or non-supported backends)
    else:
        if default_size is not None:
            if verbose:
                print(f"Using fallback batch size: {default_size}")
            return default_size

        try:
            # Analyze dataset to estimate optimal batch size
            dataset_size, num_features = analyze_dataset(data_or_train_loader)
            # Use heuristic based on number of features and dataset size
            if num_features > dataset_size:
                warnings.warn(
                    "High number of features detected."
                    " Using dataset_size // 4 for batch size."
                )
                heuristic_batch_size = max(1, dataset_size // 4)
            else:
                heuristic_batch_size = compute_smart_batch_size(
                    dataset_size, num_features, max_batch_size)

            if verbose:
                print(f"Using heuristic batch size: {heuristic_batch_size}")
            return heuristic_batch_size

        except Exception:
            warnings.warn(
                "Unable to determine batch size using heuristics. "
                "Using default batch size of 32."
            )
            if verbose:
                print("Using default batch size: 32")
            return 32

@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    )
def prepare_tf_dataset(data_or_train_loader, batch_size):
    """
    Prepare a TensorFlow dataset with the specified batch size.

    This function takes a TensorFlow Dataset or a compatible data 
    structure and applies batching to it based on the provided `batch_size`.
    It ensures that the data is properly batched for efficient training
    with TensorFlow models.

    Parameters
    ----------
    data_or_train_loader : tf.data.Dataset or TF_Dataset
        The TensorFlow Dataset or compatible data to be batched. This can 
        include any iterable TensorFlow data structures that support batching.
    
    batch_size : int
        The number of samples per batch. This determines how many samples 
        will be propagated through the network at once.
    
    Returns
    -------
    tf.data.Dataset
        A batched TensorFlow dataset ready for training. The returned dataset 
        will yield batches of size `batch_size`.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from gofast.tools.mlutils import _prepare_tf_dataset
    >>> dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([100, 10]), tf.random.uniform([100], maxval=2, dtype=tf.int32)))
    >>> batched_dataset = _prepare_tf_dataset(dataset, batch_size=32)
    >>> for batch in batched_dataset.take(1):
    ...     inputs, targets = batch
    ...     print(inputs.shape, targets.shape)
    (TensorShape([32, 10]), TensorShape([32]))
    
    Notes
    -----
    - This function is intended for internal use within the `mlutils` module.
    - It assumes that the input `data_or_train_loader` is compatible with TensorFlow's 
      batching operations. Unsupported data types will raise a `TypeError`.
    
    See Also
    --------
    tf.data.Dataset.batch : Apply batching to a TensorFlow dataset.
    
    References
    ----------
    .. [1] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Distributed Systems," 2016.
    """
        
    try:
        import tensorflow as tf
        from tensorflow.data import Dataset as TF_Dataset
    except ImportError:
        raise DependencyError(TORCH_DEP_EMSG)
        
    if isinstance(data_or_train_loader, TF_Dataset):
        return data_or_train_loader.batch(batch_size)
    elif isinstance(data_or_train_loader, tf.data.Dataset):
        return data_or_train_loader.batch(batch_size)
    else:
        raise TypeError("Unsupported data type for TensorFlow backend.")


@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend") in ("tensorflow", "tf")
    )
def analyze_dataset(data_or_train_loader, backend=None):
    """
    Analyze the dataset to determine its size and number of features.

    This function inspects the provided dataset or data loader to 
    estimate the total number of samples (`dataset_size`) and the number 
    of features (`num_features`) per sample. This information is crucial 
    for determining an appropriate batch size and optimizing training 
    performance.

    Parameters
    ----------
    data_or_train_loader : torch.utils.data.Dataset, tf.data.Dataset,\
        list, tuple, or np.ndarray
        The dataset or data loader to be analyzed. It can be a PyTorch Dataset, 
        TensorFlow Dataset, or standard Python iterable such as a list, tuple, 
        or NumPy array.
    
    Returns
    -------
    tuple
        A tuple containing:
        
        - `dataset_size` (int): The total number of samples in the dataset.
        - `num_features` (int): The number of features per sample.
    
    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import TensorDataset
    >>> from gofast.tools.mlutils import _analyze_dataset
    >>> dataset = TensorDataset(torch.randn(1000, 20), torch.randint(0, 2, (1000,)))
    >>> size, features = _analyze_dataset(dataset)
    >>> print(size, features)
    1000 20
    
    >>> import tensorflow as tf
    >>> from gofast.tools.mlutils import _analyze_dataset
    >>> tf_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([500, 15]), tf.random.uniform([500], maxval=2, dtype=tf.int32)))
    >>> size, features = _analyze_dataset(tf_dataset)
    >>> print(size, features)
    500 15
    
    Notes
    -----
    - This function supports both PyTorch and TensorFlow datasets, as well as 
      standard Python iterables like lists, tuples, and NumPy arrays.
    - For high-dimensional data, `num_features` is determined based on the 
      structure of the first sample.
    
    See Also
    --------
    get_batch_size : Main function to determine batch size based on dataset analysis.
    get_tf_dataset_size : Helper function to estimate the size of a TensorFlow dataset.
    
    References
    ----------
    .. [1] Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance
          Deep Learning Library," in NeurIPS, 2019.
    .. [2] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning on 
         Heterogeneous Distributed Systems," 2016.
    """
    
    backend= str(backend).lower() 

    if backend=="torch": 
        try:
            import torch
            import torch.utils.data as Tc_data
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG) 
            
        if isinstance(data_or_train_loader, (
                torch.utils.data.Dataset, Tc_data.Dataset)):
            
            dataset_size = len(data_or_train_loader)
            sample = data_or_train_loader[0]
            if isinstance(sample, (list, tuple)):
                num_features = len(sample[0])
            elif isinstance(sample, dict):
                num_features = len(sample)
            else:
                num_features = 1
                
    elif backend in ("tensorflow", "tf"): 
        try:
            import tensorflow as tf
            from tensorflow.data import Dataset as TF_Dataset
        except ImportError:
            raise DependencyError(TF_DEP_EMSG) 
    
        if isinstance(data_or_train_loader, (tf.data.Dataset, TF_Dataset)):
            dataset_size = get_tf_dataset_size(data_or_train_loader)
            for sample in data_or_train_loader.take(1):
                inputs, _ = sample
                if isinstance(inputs, (list, tuple)):
                    num_features = len(inputs)
                elif isinstance(inputs, dict):
                    num_features = len(inputs)
                else:
                    num_features = inputs.shape[-1] if len(inputs.shape) > 1 else 1
                    
    elif isinstance(data_or_train_loader, (list, tuple, np.ndarray)):
        dataset_size = len(data_or_train_loader)
        sample = data_or_train_loader[0]
        if isinstance(sample, (list, tuple)):
            num_features = len(sample)
        elif isinstance(sample, dict):
            num_features = len(sample)
        else:
            num_features = 1
    else:
        raise TypeError("Unsupported data type for dataset analysis.")
    
    return dataset_size, num_features

@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed for getting tf dataset size.",
    )
def get_tf_dataset_size(tf_dataset):
    """
    Estimate the size of a TensorFlow dataset.

    This function attempts to determine the number of samples in a 
    TensorFlow dataset. It uses TensorFlow's `cardinality` method to estimate 
    the dataset size. If the size cannot be determined, a `ValueError` is raised.

    Parameters
    ----------
    tf_dataset : tf.data.Dataset
        The TensorFlow dataset whose size is to be estimated.
    
    Returns
    -------
    int
        The estimated number of samples in the TensorFlow dataset.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from gofast.tools.mlutils import _get_tf_dataset_size
    >>> tf_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([200, 10]))
    >>> size = _get_tf_dataset_size(tf_dataset)
    >>> print(size)
    200
    
    Notes
    -----
    - This function relies on TensorFlow's `cardinality` method, which may 
      return `tf.data.experimental.INFINITE_CARDINALITY` or 
      `tf.data.experimental.UNKNOWN_CARDINALITY` for certain datasets.
    - Ensure that the dataset is fully defined and does not have infinite or 
      unknown cardinality before using this function.
    
    See Also
    --------
    tf.data.Dataset.cardinality : TensorFlow method to determine dataset size.
    _analyze_dataset : Helper function that uses this method to analyze datasets.
    
    References
    ----------
    .. [1] Abadi, M., et al., "TensorFlow: Large-Scale Machine 
      Learning on Heterogeneous Distributed Systems," 2016.
    """
        
    try:
        import tensorflow as tf
    except ImportError:
        raise DependencyError(TF_DEP_EMSG) 
    
    try:
        cardinality = tf.data.experimental.cardinality(tf_dataset).numpy()
        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError("Dataset has infinite cardinality.")
        if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
            raise ValueError("Unable to determine TensorFlow dataset size.")
        return int(cardinality)
    except Exception:
        raise ValueError("Unable to determine TensorFlow dataset size.")

def compute_smart_batch_size( 
        dataset_size=None, num_features=None, data=None, max_batch_size=512):
    """
    Compute a smart batch size based on dataset size and number of features.

    This function employs a heuristic to determine an appropriate batch 
    size by considering both the total number of samples (`dataset_size`) and 
    the number of features (`num_features`) per sample. The heuristic aims to 
    balance memory usage and training efficiency.

    Parameters
    ----------
    dataset_size : int
        The total number of samples in the dataset. Is essential if `data` 
        is ``None``.
    
    num_features : int, optional
        The number of features per sample. Should be provided
        if `data` is ``None``.
    
    data: ArrayLike or pd.DataFrame, optional 
       If given, `dataset_size` and `num_features` should be determined 
       accordingly if one or both are not provided. 
    
    max_batch_size : int, default=512
        The maximum allowed batch size. The computed batch size will not exceed 
        this value. Default value is set to ``512``.
    
    Returns
    -------
    int
        The computed smart batch size suitable for training.
    
    Examples
    --------
    >>> from gofast.tools.mlutils import compute_smart_batch_size
    >>> batch_size = compute_smart_batch_size(1000, 50, 512)
    >>> print(batch_size)
    32
    
    Notes
    -----
    - The function uses logarithmic scaling based on the dataset size to 
      determine an initial batch size estimate.
    - If the number of features exceeds the dataset size, a more conservative 
      batch size is chosen to prevent potential overfitting and memory issues.
    
    See Also
    --------
    get_batch_size : Main function that utilizes this helper for batch size computation.
    
    References
    ----------
    .. [1] LeCun, Y., et al., "Gradient-Based Learning Applied to 
       Document Recognition," 1998.
    """
    if data is not None: 
        if isinstance ( data, (np.ndarray, pd.DataFrame)): 
            dataset_size = dataset_size or data.shape [0] 
            num_features = num_features or data.shape [1]
            
    # Logarithmic scaling based on dataset size
    batch_size = int(math.log(dataset_size + 1, 2))  # Avoid log(0)
    batch_size = max(32, batch_size)
    
    # Adjust batch size based on number of features
    if num_features > dataset_size:
        batch_size = max(1, dataset_size // 4)
    else:
        batch_size = min(batch_size, max_batch_size)
    
    return batch_size

@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend")in ("tensorflow", "tf")
    )
def dynamic_batch_size(
    current_batch_size, performance_metrics, *,
    accuracy_threshold=0.90, scale_factor=2,
    default_size=None, backend='torch', verbose=0
):
    """
    Dynamically adjust the batch size based on model performance.

    This function modifies the batch size during training based on the 
    evaluation of performance metrics such as accuracy. If the performance 
    meets or exceeds a specified threshold, the batch size is scaled up by 
    a defined factor to potentially enhance training efficiency. If not, 
    the batch size remains unchanged or is set to a default value.

    Parameters
    ----------
    current_batch_size : int
        The current batch size being used in training.
    
    performance_metrics : dict
        A dictionary containing performance metrics, such as 
        {'accuracy': 0.92, 'loss': 0.25}. These metrics are used to decide 
        whether to adjust the batch size.
    
    accuracy_threshold : float, optional
        The threshold of accuracy at which to increase the batch size. If the 
        model's accuracy meets or exceeds this value, the batch size will be 
        increased. Default is 0.90.
    
    scale_factor : int, optional
        The factor by which to scale the batch size when performance improves.
        For example, a `scale_factor` of 2 will double the batch size. Default 
        is 2.
    
    default_size : int, optional
        The fallback batch size to use if no improvement in performance is 
        detected or if the backend is unsupported. Default is None, which means 
        the function will retain the `current_batch_size`.
    
    backend : str, optional
        The backend to use for adjustment. This can either be 'torch' for 
        PyTorch or 'tensorflow' for TensorFlow. Default is 'torch'.
    
    verbose : int, optional
        The verbosity level. If set to 1, the function will print information 
        about the batch size adjustment process. Default is 0 (silent).
    
    Returns
    -------
    int
        The adjusted batch size based on performance metrics.
    
    Examples
    --------
    >>> from gofast.tools.mlutils import dynamic_batch_size
    >>> performance_metrics = {'accuracy': 0.92, 'loss': 0.25}
    >>> new_batch_size = dynamic_batch_size(
    ...     current_batch_size=32,
    ...     performance_metrics=performance_metrics,
    ...     backend='torch',
    ...     verbose=1
    ... )
    Performance improved (accuracy: 0.92). Increasing batch size to 64.
    >>> print(new_batch_size)
    64
    
    Notes
    -----
    - The function supports both PyTorch and TensorFlow backends, adjusting 
      the batch size accordingly based on the specified `backend`.
    - It is recommended to monitor training performance continuously to 
      ensure that increasing the batch size does not negatively impact model 
      convergence or generalization.
    
    See Also
    --------
    get_batch_size : Main function that determines the initial batch size.
    _analyze_dataset : Helper function used for dataset analysis.
    _prepare_tf_dataset : Helper function used for preparing TensorFlow datasets.
    
    References
    ----------
    .. [1] Smith, L. N. (2018). "A disciplined approach to neural 
          network hyperparameters: Part 1 - learning rate, batch size, 
          momentum, and weight decay." arXiv preprint arXiv:1803.09820.
    """
    # Adjust batch size for PyTorch backend
    if backend.lower() == 'torch':
        try:
            import torch
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG) 
        if torch.cuda.is_available():
            accuracy = performance_metrics.get('accuracy', 0)
            if accuracy >= accuracy_threshold:
                new_batch_size = current_batch_size * scale_factor
                if verbose:
                    print(
                        f"Performance improved (accuracy: {accuracy}). "
                        f"Increasing batch size to {new_batch_size}."
                    )
                return new_batch_size
            else:
                if verbose:
                    print(
                        f"Performance not improved (accuracy: {accuracy}). "
                        f"Keeping batch size at {current_batch_size}."
                    )
                return current_batch_size
        else:
            if default_size is not None:
                if verbose:
                    print(f"Using fallback batch size: {default_size}")
                return default_size
            else:
                if verbose:
                    print(
                        "CUDA not available. Keeping batch size at "
                        f"{current_batch_size}."
                    )
                return current_batch_size

    # Adjust batch size for TensorFlow backend
    elif backend.lower() == 'tensorflow':
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError(TF_DEP_EMSG) 
    
        if tf.config.list_physical_devices('GPU'):
            accuracy = performance_metrics.get('accuracy', 0)
            if accuracy >= accuracy_threshold:
                new_batch_size = current_batch_size * scale_factor
                if verbose:
                    print(
                        f"Performance improved (accuracy: {accuracy}). "
                        f"Increasing batch size to {new_batch_size}."
                    )
                return new_batch_size
            else:
                if verbose:
                    print(
                        f"Performance not improved (accuracy: {accuracy}). "
                        f"Keeping batch size at {current_batch_size}."
                    )
                return current_batch_size
        else:
            if default_size is not None:
                if verbose:
                    print(f"Using fallback batch size: {default_size}")
                return default_size
            else:
                if verbose:
                    print(
                        "No GPU available for TensorFlow. Keeping batch size at "
                        f"{current_batch_size}."
                    )
                return current_batch_size

    # Fallback strategy for unsupported backends
    else:
        if default_size is not None:
            if verbose:
                print(f"Using fallback batch size: {default_size}")
            return default_size
        else:
            if verbose:
                print(
                    "No backend strategy available. Keeping batch size at "
                    f"{current_batch_size}."
                )
            return current_batch_size

def one_click_prep (
    data: DataFrame, 
    target_columns=None,
    columns=None, 
    impute_strategy=None,
    coerce_datetime=False, 
    seed=None, 
    **process_kws
    ):
    """
    Perform all-in-one preprocessing for beginners on a pandas DataFrame,
    simplifying common tasks like scaling, encoding, and imputation.

    This function is designed to be user-friendly, particularly for those new
    to data analysis, by automating complex preprocessing tasks with just a
    few parameters. It supports selective processing through specification
    of target and other columns and incorporates flexibility with custom
    imputation strategies and random seeds for reproducibility.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to preprocess. This should be a structured DataFrame
        with any combination of numeric and categorical variables.
    target_columns : str or list of str, optional
        Column(s) designated as target(s) for modeling. These columns will
        not be scaled or imputed to avoid data leakage. Defaults to None,
        implying no columns are treated as targets.
    columns : str or list of str, optional
        Column names to be specifically included in preprocessing. If None
        (default), all columns are processed.
    impute_strategy : dict, optional
        Defines specific strategies for imputing missing values, separately
        for 'numeric' and 'categorical' data types. The default strategy is
        {'numeric': 'median', 'categorical': 'constant'}.
    coerce_datetime : bool, default=False
        If True, tries to convert object columns to datetime data types when 
        `data` is a numpy array. 
    seed : int, optional
        Random seed for operations that involve randomization, ensuring
        reproducibility of results. Defaults to None.
    
    **process_kws : keyword arguments, optional
        Additional arguments that can be passed to preprocessing steps, such
        as parameters for scaling or encoding methods.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with preprocessed data, where numeric columns have been
        scaled, categorical columns encoded, and missing values imputed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np 
    >>> from gofast.tools import one_click_preprocess
    >>> data = pd.DataFrame({
    ...     'Age': [25, np.nan, 37, 59],
    ...     'City': ['New York', 'Paris', 'Berlin', np.nan],
    ...     'Income': [58000, 67000, np.nan, 120000]
    ... })
    >>> processed_data = one_click_preprocess(
    ...     data, seed=42)
    >>> processed_data.head()
            Age    Income  City_Berlin  City_New York  City_Paris  City_missing
    0 -1.180971 -0.815478            0              1           0             0
    1 -0.203616 -0.448513            0              0           1             0
    2 -0.203616 -0.448513            1              0           0             0
    3  1.588203  1.712504            0              0           0             1

    After preprocessing, the 'Age' and 'Income' columns will be scaled,
    the 'City' column will be one-hot encoded, and missing values in
    'Age' and 'City' will be imputed based on the specified strategies.

    Notes
    -----
    The function relies on scikit-learn's ColumnTransformer to apply
    different preprocessing steps to numeric and categorical columns.
    It is important to note that while this function aims to simplify
    preprocessing for beginners, understanding the underlying transformations
    and their implications is beneficial for more advanced data analysis.
    """
    # Convert input data to a DataFrame if it is not one already,
    # handling any issues silently.
    data = build_data_if(data, to_frame=True, force=True, input_name='col',
                         raise_warning='silence',
                         coerce_datetime=coerce_datetime 
                         )
    # Set a seed for reproducibility in operations that involve randomness.
    np.random.seed(seed)

    # Set default imputation strategies if none are provided.
    impute_strategy = impute_strategy or {
        'numeric': 'median', 'categorical': 'constant'}

    # Ensure impute_strategy is a dictionary with required keys.
    if ( not isinstance(impute_strategy, dict) 
        or 'numeric' not in impute_strategy 
        or 'categorical' not in impute_strategy
        ):
        raise ValueError("impute_strategy must be a dictionary with"
                         " 'numeric' and 'categorical' keys")

    # Pop keyword arguments or set defaults for handling missing categories,
    # fill values, and behavior of additional columns not specified in transformers.
    handle_unknown = process_kws.pop("handle_unknown", 'ignore')
    fill_value = process_kws.pop("fill_value", 'missing')
    remainder = process_kws.pop("remainder", 'passthrough')

    # If specific columns are specified, reduce the DataFrame to these columns only.
    if columns is not None:
        columns = list(is_iterable(columns, exclude_string= True, transform =True )) 
        data = data[columns] if isinstance(columns, list) else data[[columns]]

    # Convert target_columns to a list if it's a single column passed as string.
    if target_columns is not None:
        target_columns = [target_columns] if isinstance(
            target_columns, str) else target_columns

    # Identify numeric and categorical features based on their data type.
    numeric_features = data.select_dtypes(['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(['object']).columns.tolist()
    
    # Exclude target columns from the numeric features list if specified.
    if target_columns is not None:
        numeric_features = [col for col in numeric_features 
                            if col not in target_columns]

    # Define transformation pipeline for numeric features: imputation and scaling.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy['numeric'])),
        ('scaler', StandardScaler())
    ])

    # Define transformation pipeline for categorical 
    # features: imputation and one-hot encoding.
    steps = [('imputer', SimpleImputer(
        strategy=impute_strategy['categorical'], fill_value=fill_value))
        ] 
    categorical_step = ('onehot', OneHotEncoder(handle_unknown=handle_unknown))
    if categorical_features: 
        steps += [categorical_step]
        
    categorical_transformer = Pipeline(steps=steps)

    # Combine the numeric and categorical transformations with ColumnTransformer.
    transformer_steps = [('num', numeric_transformer, numeric_features)] 
    if categorical_features: 
        transformer_steps +=[('cat', categorical_transformer, categorical_features)]
        
    preprocessor = ColumnTransformer(
        transformers=transformer_steps, remainder=remainder
    )

    # Fit and transform the data using the defined ColumnTransformer.
    data_processed = preprocessor.fit_transform(data)

    # Attempt to retrieve processed column names for creating
    # a DataFrame from the transformed data.
    try:
        if categorical_features: 
            processed_columns = numeric_features + list(get_feature_names(
                preprocessor.named_transformers_['cat']['onehot'], categorical_features)
                ) + [col for col in data.columns 
                     if col not in numeric_features + categorical_features]
        else: 
            processed_columns = numeric_features  + [
                col for col in data.columns if col not in numeric_features]
    except:
        # Fallback for older versions of scikit-learn or other compatibility issues.
        if categorical_features: 
            cat_features_names = get_feature_names(preprocessor.named_transformers_['cat'])
            processed_columns = numeric_features + list(cat_features_names)
        else: 
            processed_columns = numeric_features

    # Check if the transformed data is a sparse matrix and convert to dense if necessary.
    if sparse.issparse(data_processed):
        data_processed = data_processed.toarray()

    # Create a new DataFrame with the processed data.
    data_processed = pd.DataFrame(
        data_processed, columns=processed_columns, index=data.index)

    # Attempt to use a custom transformer if available.
    try:
        from ..transformers import FloatCategoricalToInt
        data_processed = FloatCategoricalToInt(
            ).fit_transform(data_processed)
    except:
        pass
    # Return the preprocessed DataFrame.
    return data_processed

def soft_encoder(
    data: Union[DataFrame, ArrayLike], 
    columns: List[str] = None, 
    func: _F = None, 
    categories: Dict[str, List] = None, 
    get_dummies: bool = ..., 
    parse_cols: bool = ..., 
    return_cat_codes: bool = ..., 
) -> DataFrame:
    """
    Encode multiple categorical variables in a dataset.

    Function facilitates the encoding of categorical variables by applying 
    specified transformations, mapping categories to integers, or performing 
    one-hot encoding. It accepts input data in the form of pandas DataFrames, 
    array-like structures convertible to DataFrame, or dictionaries that are 
    directly convertible to DataFrame.

    Parameters
    ----------
    data : DataFrame | ArrayLike | dict
        The input data to process. Accepts a pandas DataFrame, any array-like 
        structure that can be converted to a DataFrame, or a dictionary that will
        be converted to a DataFrame. In the case of array-like or dictionary
        inputs, the structure must be suitable for DataFrame transformation.
    columns : list, optional
        A list of column names from the data that are to be encoded. If not 
        provided, all columns within the DataFrame are considered for encoding.
    func : callable, optional
        A function to apply to the data for encoding purposes. The function 
        should accept a single value and return a transformed value. It is 
        applied to each element of the columns specified by `columns`, or to all
        elements in the DataFrame if `columns` is None.
    categories : dict, optional
        A dictionary mapping column names (keys) to lists of categories (values)
        that should be used for encoding. This dictionary explicitly defines 
        the categories corresponding to each column that should be transformed.
    get_dummies : bool, default False
        When set to True, enables one-hot encoding for the specified `columns` 
        or for all columns if `columns` is unspecified. This parameter converts
        each categorical variable into multiple binary columns, one for each 
        category, which indicates the presence of the category.
    parse_cols : bool, default False
        If True and `columns` is a string, this parameter will interpret the 
        string as a list of column names, effectively parsing a single comma-
        separated string into separate column names.
    return_cat_codes : bool, default False
        When True, the function returns a tuple. The first element of the tuple 
        is the DataFrame with transformed data, and the second element is a 
        dictionary that maps the original categorical values to the new 
        numerical codes. This is useful for retaining a reference to the original
        categorical data.

    Returns
    -------
    DataFrame or (DataFrame, dict)
        The primary return is the encoded DataFrame. If `return_cat_codes` is 
        True, a dictionary mapping original categories to their new numerical 
        codes is also returned.
        
    Raises
    ------
    TypeError
        If `func` is provided but is not callable, or if `categories` 
        is not a dictionary.
        
    Examples
    --------
    >>> from gofast.tools.mlutils import soft_encoder
    >>> # Sample dataset with categorical variables
    >>> data = {'Height': [152, 175, 162, 140, 170],
    ...         'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    ...         'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small'],
    ...         'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Triangle'],
    ...         'Weight': [80, 75, 55, 61, 70]
    ...        }
    >>> # Basic encoding without additional parameters
    >>> df_encoded = soft_encoder(data)
    >>> df_encoded.head(2)
    Out[1]:
       Height  Weight  Color  Size  Shape
    0     152      80      2     2      0
    1     175      75      0     0      1
    
    >>> # Returning a map of categorical codes
    >>> df_encoded, map_codes = soft_encoder(data, return_cat_codes=True)
    >>> map_codes
    Out[2]:
    {'Color': {2: 'Red', 0: 'Blue', 1: 'Green'},
     'Size': {2: 'Small', 0: 'Large', 1: 'Medium'},
     'Shape': {0: 'Circle', 1: 'Square', 2: 'Triangle'}}
    
    >>> # Custom function to manually map categories
    >>> def cat_func(x):
    ...     if x == 'Red':
    ...         return 2
    ...     elif x == 'Blue':
    ...         return 0
    ...     elif x == 'Green':
    ...         return 1
    ...     else:
    ...         return x
    >>> df_encoded = soft_encoder(data, func=cat_func)
    >>> df_encoded.head(3)
    Out[3]:
       Height  Color    Size     Shape  Weight
    0     152      2   Small    Circle      80
    1     175      0   Large    Square      75
    2     162      1  Medium  Triangle      55
    
    >>> # Perform one-hot encoding
    >>> df_encoded = soft_encoder(data, get_dummies=True)
    >>> df_encoded.head(3)
    Out[4]:
       Height  Weight  Color_Blue  Color_Green  Color_Red  Size_Large  Size_Medium  \
    0     152      80           0            0          1           0            0   
    1     175      75           1            0          0           1            0   
    2     162      55           0            1          0           0            1   
    
       Size_Small  Shape_Circle  Shape_Square  Shape_Triangle
    0           1             1             0               0
    1           0             0             1               0
    2           0             0             0               1
    
    >>> # Specifying explicit categories
    >>> df_encoded = soft_encoder(data, categories={'Size': ['Small', 'Large', 'Medium']})
    >>> df_encoded.head()
    Out[5]:
       Height  Color     Shape  Weight  Size
    0     152    Red    Circle      80     0
    1     175   Blue    Square      75     1
    2     162  Green  Triangle      55     2
    3     140    Red    Circle      61     2
    4     170   Blue  Triangle      70     0
    
    Notes
    -----
    - The function handles various forms of input data, applying default 
      encoding if no specific encoding function or categories are provided.
    - Custom encoding functions allow for flexibility in mapping categories 
      manually.
    - One-hot encoding transforms each categorical attribute into multiple 
      binary attributes, enhancing model interpretability but increasing 
      dimensionality.
    - Specifying explicit categories helps ensure consistency in encoding, 
      especially when some categories might not appear in the training set but 
      could appear in future data.

    """
    from .datautils import nan_to_na 
    
    # Convert ellipsis inputs to False for get_dummies, parse_cols,
    # return_cat_codes if not explicitly defined
    get_dummies, parse_cols, return_cat_codes = ellipsis2false(
        get_dummies, parse_cols, return_cat_codes)

    # Convert input data to DataFrame if not already a DataFrame
    df = build_data_if(data, to_frame=True, force=True, input_name='col',
                       raise_warning='silence')

    # Recheck and convert data to numeric dtypes if possible
    # and handle NaN to fit it specified types. 
    df = nan_to_na(to_numeric_dtypes(df)) 
    # Ensure columns are iterable and parse them if necessary
    if columns is not None:
        columns = list(is_iterable(columns, exclude_string=True, 
                                   transform=True, parse_string=parse_cols))
        # Select only the specified features from the DataFrame
        df = select_features(df, features=columns)
    
    # Initialize map_codes to store mappings of categorical codes to labels
    map_codes = {}
    # Create a CategoryMap code object for nested dict collection
    mapresult=ResultSummary(name="CategoryMap", flatten_nested_dicts=False)
    
    # Perform one-hot encoding if requested
    if get_dummies:
        # Use pandas get_dummies for one-hot encoding and handle
        # return type based on return_cat_codes
        mapresult.add_results(map_codes)
        return (pd.get_dummies(df, columns=columns), mapresult
                ) if return_cat_codes else pd.get_dummies(df, columns=columns)

    # Automatically select numeric and categorical columns if not manually specified
    num_columns, cat_columns = bi_selector(df)

    # Apply provided function to categorical columns if func is given
    if func is not None:
        if not callable(func):
            raise TypeError(f"Provided func is not callable. Received: {type(func)}")
        if len(cat_columns) == 0:
            # Warn if no categorical data were found
            warnings.warn("No categorical data were detected. To transform"
                          " numeric values into categorical labels, consider"
                          " using either `gofast.tools.smart_label_classifier`"
                          " or `gofast.tools.categorize_target`.")
            return df
        
        # Apply the function to each categorical column
        for col in cat_columns:
            df[col] = df[col].apply(func)

        # Return DataFrame and mappings if required
        mapresult.add_results(map_codes)
        return (df, mapresult) if return_cat_codes else df

    # Handle automatic categorization if categories are not provided
    if categories is None:
        categories = {}
        for col in cat_columns:
            # Drop NaN values before finding unique categories
            unique_values = np.unique(df[col].dropna())
            categories[col] =  list(unique_values) 

    # Ensure categories is a dictionary
    if not isinstance(categories, dict):
        raise TypeError("Expected a dictionary with the format"
                        " {'column name': 'labels'} to categorize data.")

    # Map categories for each column and adjust DataFrame accordingly
    for col, values in categories.items():
        if col not in df.columns:
            continue
        values = is_iterable(values, exclude_string=True, transform=True)
        df[col] = pd.Categorical(df[col], categories=values, ordered=True)
        val = df[col].cat.codes
        temp_col = col + '_col'
        df[temp_col] = val
        map_codes[col] = dict(zip(val, df[col]))
        df.drop(columns=[col], inplace=True)  # Drop original column
        # Rename the temp column to original column name
        df.rename(columns={temp_col: col}, inplace=True) 

    # Return DataFrame and mappings if required
    mapresult.add_results(map_codes)
    return (df, mapresult) if return_cat_codes else df

@ensure_pkg ("imblearn", extra= (
    "`imblearn` is actually a shorthand for ``imbalanced-learn``.")
   )
def resampling( 
    X, 
    y, 
    kind ='over', 
    strategy ='auto', 
    random_state =None, 
    verbose: bool=..., 
    **kws
    ): 
    """ Combining Random Oversampling and Undersampling 
    
    Resampling involves creating a new transformed version of the training 
    dataset in which the selected examples have a different class distribution.
    This is a simple and effective strategy for imbalanced classification 
    problems.

    Applying re-sampling strategies to obtain a more balanced data 
    distribution is an effective solution to the imbalance problem. There are 
    two main approaches to random resampling for imbalanced classification; 
    they are oversampling and undersampling.

    - Random Oversampling: Randomly duplicate examples in the minority class.
    - Random Undersampling: Randomly delete examples in the majority class.
        
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y: array-like of shape (n_samples, ) 
        Target vector where `n_samples` is the number of samples.
    kind: str, {"over", "under"} , default="over"
      kind of sampling to perform. ``"over"`` and ``"under"`` stand for 
      `oversampling` and `undersampling` respectively. 
      
    strategy : float, str, dict, callable, default='auto'
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
          
    random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.

            - If int, ``random_state`` is the seed used by the random number
              generator;
            - If ``RandomState`` instance, random_state is the random number
              generator;
            - If ``None``, the random number generator is the ``RandomState``
              instance used by ``np.random``.
              
    verbose: bool, default=False 
      Display the counting samples 
      
    Returns 
    ---------
    X, y : NDarray, Arraylike 
        Arraylike sampled 
    
    Examples 
    --------- 
    >>> import gofast as gf 
    >>> from gofast.tools.mlutils import resampling 
    >>> data, target = gf.fetch_data ('bagoue analysed', as_frame =True, return_X_y=True) 
    >>> data.shape, target.shape 
    >>> data_us, target_us = resampling (data, target, kind ='under',verbose=True)
    >>> data_us.shape, target_us.shape 
    Counters: Auto      
                         Raw counter y: Counter({0: 232, 1: 112})
               UnderSampling counter y: Counter({0: 112, 1: 112})
    Out[43]: ((224, 8), (224,))
    
    """
    kind = str(kind).lower() 
    if kind =='under': 
        from imblearn.under_sampling import RandomUnderSampler
        rsampler = RandomUnderSampler(sampling_strategy=strategy, 
                                      random_state = random_state ,
                                      **kws)
    else:  
        from imblearn.over_sampling import RandomOverSampler 
        rsampler = RandomOverSampler(sampling_strategy=strategy, 
                                     random_state = random_state ,
                                     **kws
                                     )
    Xs, ys = rsampler.fit_resample(X, y)
    
    if ellipsis2false(verbose)[0]: 
        print("{:<20}".format(f"Counters: {strategy.title()}"))
        print( "{:>35}".format( "Raw counter y:") , Counter (y))
        print( "{:>35}".format(f"{kind.title()}Sampling counter y:"), Counter (ys))
        
    return Xs, ys 

def bin_counting(
    data: DataFrame, 
    bin_columns: Union[str, List[str]], 
    tname: Union[str, Series[int]], 
    odds: str = "N+", 
    return_counts: bool = ..., 
    tolog: bool = ..., 
    encode_categorical: bool = ..., 
) -> None:
    """ Bin counting categorical variable and turn it into probabilistic 
    ratio.
    
    Bin counting is one of the perennial rediscoveries in machine learning. 
    It has been reinvented and used in a variety of applications, from ad 
    click-through rate prediction to hardware branch prediction [1]_, [2]_ 
    and [3]_.
    
    Given an input variable X and a target variable Y, the odds ratio is 
    defined as:
        
    .. math:: 
        
        odds ratio = \frac{ P(Y = 1 | X = 1)/ P(Y = 0 | X = 1)}{
            P(Y = 1 | X = 0)/ P(Y = 0 | X = 0)}
          
    Probability ratios can easily become very small or very large. The log 
    transform again comes to our rescue. Anotheruseful property of the 
    logarithm is that it turns a division into a subtraction. To turn 
    bin statistic probability value to log, set ``uselog=True``.
    
    Parameters 
    -----------
    data: dataframe 
       Data containing the categorical values. 
       
    bin_columns: str or list , 
       The columns to apply the bin_counting. If 'auto', the columns are 
       extracted and categorized before applying the `bin_counting`. 
       
    tname: str, pd.Series
      The target name for which the counting is operated. If series, it 
      must have the same length as the data. 
      
    odds: str , {"N+", "N-", "log_N+"}: 
        The odds ratio of bin counting to fill the categorical. ``N+`` and  
        ``N-`` are positive and negative probabilistic computing. Whereas the
        ``log_N+`` is the logarithm odds ratio useful when value are smaller 
        or larger. 
        
    return_counts: bool, default=True 
      return the bin counting dataframes. 
  
    tolog: bool, default=False, 
      Apply the logarithm to the output data ratio. Indeed, Probability ratios 
      can easily  become very small or very large. For instance, there will be 
      users who almost never click on ads, and perhaps users who click on ads 
      much more frequently than not.) The log transform again comes to our  
      rescue. Another useful property of the logarithm is that it turns a 
      division 
      
    encode_categorical : bool, optional
        If `True`, encode categorical variables. Default is `False`.

    Returns 
    --------
    d: dataframe 
       Dataframe transformed or bin-counting data
       
    Examples 
    ---------
    >>> import gofast as gf 
    >>> from gofast.tools.mlutils import bin_counting 
    >>> X, y = gf.fetch_data ('bagoue analysed', as_frame =True) 
    >>> # target binarize 
    >>> y [y <=1] = 0;  y [y > 0]=1 
    >>> X.head(2) 
    Out[7]: 
          power  magnitude       sfi      ohmS       lwi  shape  type  geol
    0  0.191800  -0.140799 -0.426916  0.386121  0.638622    4.0   1.0   3.0
    1 -0.430644  -0.114022  1.678541 -0.185662 -0.063900    3.0   2.0   2.0
    >>>  bin_counting (X , bin_columns= 'geol', tname =y).head(2)
    Out[8]: 
          power  magnitude       sfi      ohmS  ...  shape  type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  0.386121  ...    4.0   1.0  0.656716           1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...    3.0   2.0  0.219251           0
    [2 rows x 9 columns]
    >>>  bin_counting (X , bin_columns= ['geol', 'shape', 'type'], tname =y).head(2)
    Out[10]: 
          power  magnitude       sfi  ...      type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.267241  0.656716           1
    1 -0.430644  -0.114022  1.678541  ...  0.385965  0.219251           0
    [2 rows x 9 columns]
    >>> df = pd.DataFrame ( pd.concat ( [X, pd.Series ( y, name ='flow')],
                                       axis =1))
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'], 
                      tname ="flow", tolog=True).head(2)
    Out[12]: 
          power  magnitude       sfi      ohmS  ...     shape      type      geol  flow
    0  0.191800  -0.140799 -0.426916  0.386121  ...  0.828571  0.364706  1.913043     1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...  0.364865  0.628571  0.280822     0
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'],odds ="N-", 
                      tname =y, tolog=True).head(2)
    Out[13]: 
          power  magnitude       sfi  ...      geol  flow  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.522727     1           1
    1 -0.430644  -0.114022  1.678541  ...  3.560976     0           0
    [2 rows x 10 columns]
    >>> bin_counting (df , bin_columns= "geol",tname ="flow", tolog=True,
                      return_counts= True )
    Out[14]: 
         flow  no_flow  total_flow        N+        N-     logN+     logN-
    3.0    44       23          67  0.656716  0.343284  1.913043  0.522727
    2.0    41      146         187  0.219251  0.780749  0.280822  3.560976
    0.0    18       43          61  0.295082  0.704918  0.418605  2.388889
    1.0     9       20          29  0.310345  0.689655  0.450000  2.222222

    References 
    -----------
    .. [1] Yeh, Tse-Yu, and Yale N. Patt. Two-Level Adaptive Training Branch 
           Prediction. Proceedings of the 24th Annual International 
           Symposium on Microarchitecture (1991):51–61
           
    .. [2] Li, Wei, Xuerui Wang, Ruofei Zhang, Ying Cui, Jianchang Mao, and 
           Rong Jin.Exploitation and Exploration in a Performance Based Contextual 
           Advertising System. Proceedings of the 16th ACM SIGKDD International
           Conference on Knowledge Discovery and Data Mining (2010): 27–36
           
    .. [3] Chen, Ye, Dmitry Pavlov, and John _F. Canny. “Large-Scale Behavioral 
           Targeting. Proceedings of the 15th ACM SIGKDD International 
           Conference on Knowledge Discovery and Data Mining (2009): 209–218     
    """
    return_counts, tolog, encode_categorical= ellipsis2false(
        return_counts, tolog, encode_categorical)   
    # assert everything
    if not is_frame (data, df_only =True ):
        raise TypeError(f"Expect dataframe. Got {type(data).__name__!r}")
        
    if isinstance (bin_columns, str) and bin_columns=='auto': 
        ttname = tname if isinstance (tname, str) else None # pass 
        _, bin_columns = process_data_types(
            data, target_name= ttname,
            exclude_target=True if ttname else False 
        )
        
    if encode_categorical: 
        data = soft_encoder(data) 
        
    if not _is_numeric_dtype(data, to_array= True): 
        raise TypeError ("Expect data with encoded categorical variables."
                         " Please check your data.")
    if hasattr ( tname, '__array__'): 
        check_consistent_length( data, tname )
        if not _is_arraylike_1d(tname): 
            raise TypeError (
                 "Only one dimensional array is allowed for the target.")
        # create fake bin target 
        if not hasattr ( tname, 'name'): 
            tname = pd.Series (tname, name ='bin_target')
        # concatenate target 
        data= pd.concat ( [ data, tname], axis = 1 )
        tname = tname.name  # take the name 
    
    bin_columns= is_iterable(bin_columns, exclude_string= True, transform =True )
    tname = str(tname) ; 
    target_all_counts =[]
    
    validate_feature(data, features =bin_columns + [tname] )
    d= data.copy() 
    # -convert all features dtype to float for consistency
    # except the binary target 
    feature_cols = is_in_if (d.columns , tname, return_diff= True ) 
    d[feature_cols] = d[feature_cols].astype ( float)
    # -------------------------------------------------
    for bin_column in bin_columns: 
        d, tc  = _single_counts(d , bin_column, tname, 
                           odds =odds, 
                           tolog=tolog, 
                           return_counts= return_counts
                           )
    
        target_all_counts.append (tc) 
    # lowering the computation time 
    if return_counts: 
        d = ( target_all_counts if len(target_all_counts) >1 
                 else target_all_counts [0]
                 ) 

    return d

def _single_counts ( 
        d,  bin_column, tname, odds = "N+",
        tolog= False, return_counts = False ): 
    """ An isolated part of bin counting. 
    Compute single bin_counting. """
    # polish pos_label 
    od = copy.deepcopy( odds) 
    # reconvert log and removetrailer
    odds= str(odds).upper().replace ("_", "")
    # just separted for 
    keys = ('N-', 'N+', 'lOGN+')
    msg = ("Odds ratio or log Odds ratio expects"
           f" {smart_format(('N-', 'N+', 'logN+'), 'or')}. Got {od!r}")
    # check wther log is included 
    if odds.find('LOG')>=0: 
        tolog=True # then remove log 
        odds= odds.replace ("LOG", "")

    if odds not in keys: 
        raise ValueError (msg) 
    # If tolog, then reconstructs
    # the odds_labels
    if tolog: 
        odds= f"log{odds}"
    
    target_counts= _target_counting(
        d.filter(items=[bin_column, tname]),
    bin_column , tname =tname, 
    )
    target_all, target_bin_counts = _bin_counting(target_counts, tname, odds)
    # Check to make sure we have all the devices
    target_all.sort_values(by = f'total_{tname}', ascending=False)  
    if return_counts: 
        return d, target_all 
   
    # zip index with ratio 
    lr = list(zip (target_bin_counts.index, target_bin_counts[odds])
         )
    ybin = np.array ( d[bin_column])# replace value with ratio 
    for (value , ratio) in lr : 
        ybin [ybin ==value] = ratio 
        
    d[bin_column] = ybin 
    
    return d, target_all

def _target_counting(d, bin_column, tname ):
    """ An isolated part of counting the target. 
    
    :param d: DataFrame 
    :param bin_column: str, columns to appling bincounting strategy 
    :param tname: str, target name. 

    """
    pos_action = pd.Series(d[d[tname] > 0][bin_column].value_counts(),
        name=tname)
    
    neg_action = pd.Series(d[d[tname] < 1][bin_column].value_counts(),
    name=f'no_{tname}')
     
    counts = pd.DataFrame([pos_action,neg_action]).T.fillna('0')
    counts[f'total_{tname}'] = counts[tname].astype('int64') +\
    counts[f'no_{tname}'].astype('int64')
    
    return counts

def _bin_counting (counts, tname, odds="N+" ):
    """ Bin counting application to the target. 
    :param counts: pd.Series. Target counts 
    :param tname: str target name. 
    :param odds: str, label to bin-compute
    """
    counts['N+'] = ( counts[tname]
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64')
                            )
                    )
    counts['N-'] = ( counts[f'no_{tname}']
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64'))
                    )
    
    items2filter= ['N+', 'N-']
    if str(odds).find ('log')>=0: 
        counts['logN+'] = counts['N+'].divide(counts['N-'])
        counts ['logN-'] = counts ['N-'].divide ( counts['N+'])
        items2filter.extend (['logN+', 'logN-'])
    # If we wanted to only return bin-counting properties, 
    # we would filter here
    bin_counts = counts.filter(items= items2filter)

    return counts, bin_counts  
 
def laplace_smoothing_word(word, class_,  word_counts, class_counts, V):
    """
    Apply Laplace smoothing to estimate the conditional probability of a 
    word given a class.

    Laplace smoothing (add-one smoothing) is used to handle the issue of 
    zero probability in categorical data, particularly in the context of 
    text classification with Naive Bayes.

    The mathematical formula for Laplace smoothing is:
    
    .. math:: 
        P(w|c) = \frac{\text{count}(w, c) + 1}{\text{count}(c) + |V|}

    where `count(w, c)` is the count of word `w` in class `c`, `count(c)` is 
    the total count of all words in class `c`, and `|V|` is the size of the 
    vocabulary.

    Parameters
    ----------
    word : str
        The word for which the probability is to be computed.
    class_ : str
        The class for which the probability is to be computed.
    word_counts : dict
        A dictionary containing word counts for each class. The keys should 
        be tuples of the form (word, class).
    class_counts : dict
        A dictionary containing the total count of words for each class.
    V : int
        The size of the vocabulary, i.e., the number of unique words in 
        the dataset.

    Returns
    -------
    float
        The Laplace-smoothed probability of the word given the class.

    Example
    -------
    >>> from gofast.tools.mlutils import laplace_smoothing_word
    >>> word_counts = {('dog', 'animal'): 3, ('cat', 'animal'):
                       2, ('car', 'non-animal'): 4}
    >>> class_counts = {'animal': 5, 'non-animal': 4}
    >>> V = len(set([w for (w, c) in word_counts.keys()]))
    >>> laplace_smoothing_word('dog', 'animal', word_counts, class_counts, V)
    0.5
    
    References
    ----------
    - C.D. Manning, P. Raghavan, and H. Schütze, "Introduction to Information Retrieval",
      Cambridge University Press, 2008.
    - A detailed explanation of Laplace Smoothing can be found in Chapter 13 of 
      "Introduction to Information Retrieval" by Manning et al.

    Notes
    -----
    This function is particularly useful in text classification tasks where the
    dataset may contain a large number of unique words, and some words may not 
    appear in the training data for every class.
    """
    word_class_count = word_counts.get((word, class_), 0)
    class_word_count = class_counts.get(class_, 0)
    probability = (word_class_count + 1) / (class_word_count + V)
    return probability

def laplace_smoothing_categorical(
        data, feature_col, class_col, V=None):
    """
    Apply Laplace smoothing to estimate conditional probabilities of 
    categorical features given a class in a dataset.

    This function calculates the Laplace-smoothed probabilities for each 
    category of a specified feature given each class.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing categorical features and a class label.
    feature_col : str
        The column name in the dataset representing the feature for which 
        probabilities are to be calculated.
    class_col : str
        The column name in the dataset representing the class label.
    V : int or None, optional
        The size of the vocabulary (i.e., the number of unique categories 
                                    in the feature).
        If None, it will be calculated based on the provided feature column.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the Laplace-smoothed probabilities for each 
        category of the feature across each class.

    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import laplace_smoothing_categorical
    >>> data = pd.DataFrame({'feature': ['cat', 'dog', 'cat', 'bird'],
                             'class': ['A', 'A', 'B', 'B']})
    >>> probabilities = laplace_smoothing_categorical(data, 'feature', 'class')
    >>> print(probabilities)

    Notes
    -----
    This function is useful for handling categorical data in classification
    tasks, especially when the dataset may contain categories that do not 
    appear in the training data for every class.
    """
    is_frame( data, df_only=True, raise_exception=True)
    if V is None:
        V = data[feature_col].nunique()

    class_counts = data[class_col].value_counts()
    probability_tables = []

    # Iterating over each class to calculate probabilities
    for class_value in data[class_col].unique():
        class_subset = data[data[class_col] == class_value]
        feature_counts = class_subset[feature_col].value_counts()
        probabilities = (feature_counts + 1) / (class_counts[class_value] + V)
        probabilities.name = class_value
        probability_tables.append(probabilities.to_frame().T)

    # Using pandas.concat to combine the probability tables
    probability_table = pd.concat(probability_tables, sort=False).fillna(1 / V)
    # Transpose to match expected format: features as rows, classes as columns
    probability_table = probability_table.T

    return probability_table

def laplace_smoothing(
    data: Union[ArrayLike, DataFrame], 
    alpha: float = 1.0, 
    columns: Union[list, None] = None
) -> Union[ArrayLike, DataFrame]:
    """
    Applies Laplace smoothing to  data to calculate smoothed probabilities.

    Parameters
    ----------
    data : ndarray or DataFrame
        An array-like or DataFrame object containing categorical data. Each column 
        represents a feature, and each row represents a data sample.
    alpha : float, optional
        The smoothing parameter, often referred to as 'alpha'. This is 
        added to the count for each category in each feature. 
        Default is 1 (Laplace Smoothing).
    columns: list, optional
        Columns to construct the DataFrame when `data` is an ndarray. The 
        number of columns must match the second dimension of the ndarray.
        
    Returns
    -------
    smoothed_probs : ndarray or DataFrame
        An array or DataFrame of the same shape as `data` containing the smoothed 
        probabilities for each category in each feature.

    Raises
    ------
    ValueError
        If `columns` is provided and its length does not match the number 
        of columns in `data`.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import laplace_smoothing
    >>> data = np.array([[0, 1], [1, 0], [1, 1]])
    >>> laplace_smoothing(data, alpha=1)
    array([[0.4 , 0.6 ],
           [0.6 , 0.4 ],
           [0.6 , 0.6 ]])

    >>> data_df = pd.DataFrame(data, columns=['feature1', 'feature2'])
    >>> laplace_smoothing(data_df, alpha=1)
       feature1  feature2
    0       0.4       0.6
    1       0.6       0.4
    2       0.6       0.6
    """
    if isinstance(data, np.ndarray):
        if columns:
            if len(columns) != data.shape[1]:
                raise ValueError("Length of `columns` does not match the shape of `data`.")
            data = pd.DataFrame(data, columns=columns)
        input_type = 'ndarray'
    elif isinstance(data, pd.DataFrame):
        input_type = 'dataframe'
    else:
        raise TypeError("`data` must be either a numpy.ndarray or a pandas.DataFrame.")

    smoothed_probs_list = []
    features = data.columns if input_type == 'dataframe' else range(data.shape[1])

    for feature in features:
        series = data[feature] if input_type == 'dataframe' else data[:, feature]
        counts = np.bincount(series, minlength=series.max() + 1)
        smoothed_counts = counts + alpha
        total_counts = smoothed_counts.sum()
        smoothed_probs = (series.map(lambda x: smoothed_counts[x] / total_counts)
                          if input_type == 'dataframe' else smoothed_counts[series] / total_counts)
        smoothed_probs_list.append(smoothed_probs)

    if input_type == 'dataframe':
        return pd.DataFrame({feature: probs for feature, probs in zip(
            features, smoothed_probs_list)})
    else:
        return np.column_stack(smoothed_probs_list)

def evaluate_model(
    model: Optional[_F] = None,
    X: Optional[Union[NDArray, DataFrame]] = None,
    Xt: Optional[Union[NDArray, DataFrame]] = None,
    y: Optional[Union[NDArray, Series]] = None, 
    yt: Optional[Union[NDArray, Series]] = None,
    y_pred: Optional[Union[NDArray, Series]] = None,
    scorer: Union[str, _F] = 'accuracy_score',
    eval: bool = False,
    **kws: Any
) -> Union[Tuple[Optional[Union[NDArray, Series]], Optional[float]],
           Optional[Union[NDArray, Series]]]:
    """
    Evaluates a predictive model's performance or the effectiveness of predictions 
    using a specified scoring metric.

    Parameters
    ----------
    model : Callable, optional
        A machine learning model that implements fit and predict methods.
        Required if `y_pred` is not provided.
    X : np.ndarray or pd.DataFrame, optional
        Training data features. Required if `model` is provided and `y_pred` is None.
    Xt : np.ndarray or pd.DataFrame, optional
        Test data features. Required if `model` is provided and `y_pred` is None.
    y : np.ndarray or pd.Series, optional
        Training data labels. Required if `model` is provided and `y_pred` is None.
    yt : np.ndarray or pd.Series, optional
        Test data labels. Required if `eval` is True.
    y_pred : np.ndarray or pd.Series, optional
        Predictions for test data. Required if `model` is None.
    scorer : str or Callable, default='accuracy'
        The scoring metric name or a scorer callable object/function with signature 
        scorer(y_true, y_pred, **kws). 
    eval : bool, default=False
        If True, performs evaluation using `scorer` on `yt` and `y_pred`.
    **kws : Any
        Additional keyword arguments to pass to the scoring function.

    Returns
    -------
    predictions : np.ndarray or pd.Series
        The predicted labels or probabilities.
    score : float, optional
        The score of the predictions based on `scorer`. Only returned if `eval` is True.

    Raises
    ------
    ValueError
        If required arguments are missing or if the provided arguments are invalid.
    TypeError
        If `scorer` is not a recognized scoring function.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.tools.mlutils import evaluate_model
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> model = LogisticRegression()
    >>> y_pred, score = evaluate_model(model=model, X=X_train, Xt=X_test,
    ...                                y=y_train, yt=y_test, eval=True)
    >>> print(f'Score: {score:.2f}')
    
    >>> # Providing predictions directly
    >>> y_pred, _ = evaluate_model(y_pred=y_pred, yt=y_test, scorer='accuracy',
    ...                            eval=True)
    >>> print(f'Accuracy: {score:.2f}')
    """
    from ..metrics import fetch_scorers
    
    if y_pred is None:
        if model is None or X is None or y is None or Xt is None:
            raise ValueError("Model, X, y, and Xt must be provided when y_pred"
                             " is not provided.")
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise TypeError("The provided model does not implement fit and "
                            "predict methods.")
        
        # Check model if is fitted
        try: check_is_fitted(model)
        except: 
            # If the model is not fitted, then fit it with X and y
            if X is not None and y is not None:
                if hasattr(X, 'ndim') and X.ndim == 1:
                    X = X.reshape(-1, 1)
                model.fit(X, y)
            else:
                raise ValueError("Model is not fitted, and no training data"
                                 " (X, y) were provided.")
        y_pred = model.predict(Xt)

    if eval:
        if yt is None:
            raise ValueError("yt must be provided when eval is True.")
        if not isinstance(scorer, (str, callable)):
            raise TypeError("scorer must be a string or a callable,"
                            f" got {type(scorer).__name__}.")
        if isinstance (scorer, str): 
            scorer= fetch_scorers (scorer) 
        # score_func = get_scorer(scorer, include_sklearn= True )
        score = scorer(yt, y_pred, **kws)
        return y_pred, score

    return y_pred

def get_correlated_features(
    data:DataFrame ,
    corr:str ='pearson', 
    threshold: float=.95 , 
    fmt: bool= False 
    )-> DataFrame: 
    """Find the correlated features/columns in the dataframe. 
    
    Indeed, highly correlated columns don't add value and can throw off 
    features importance and interpretation of regression coefficients. If we  
    had correlated columns, choose to remove either the columns from  
    level_0 or level_1 from the features data is a good choice. 
    
    Parameters 
    -----------
    data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
        Dataframe containing samples M  and features N
    corr: str, ['pearson'|'spearman'|'covariance']
        Method of correlation to perform. Note that the 'person' and 
        'covariance' don't support string value. If such kind of data 
        is given, turn the `corr` to `spearman`. *default* is ``pearson``
        
    threshold: int, default is ``0.95``
        the value from which can be considered as a correlated data. Should not 
        be greater than 1. 
        
    fmt: bool, default {``False``}
        format the correlated dataframe values 
        
    Returns 
    ---------
    df: `pandas.DataFrame`
        Dataframe with cilumns equals to [level_0, level_1, pearson]
        
    Examples
    --------
    >>> from gofast.tools.mlutils import get_correlated_features 
    >>> df_corr = get_correlated_features (data , corr='spearman',
                                     fmt=None, threshold=.95
                                     )
    """
    data = build_data_if(data, to_frame=True, raise_exception= True, 
                         input_name="col")
    
    th= copy.deepcopy(threshold) 
    threshold = str(threshold)  
    try : 
        threshold = float(threshold.replace('%', '')
                          )/1e2  if '%' in threshold else float(threshold)
    except: 
        raise TypeError (
            f"Threshold should be a float value, got: {type(th).__name__!r}")
          
    if threshold >= 1 or threshold <= 0 : 
        raise ValueError (
            f"threshold must be ranged between 0 and 1, got {th!r}")
      
    if corr not in ('pearson', 'covariance', 'spearman'): 
        raise ValueError (
            f"Expect ['pearson'|'spearman'|'covariance'], got{corr!r} ")
    # collect numerical values and exclude cat values
    
    df = select_features(data, None, 'number')
        
    # use pipe to chain different func applied to df 
    c_df = ( 
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril (df1, k=-1 ), # low triangle zeroed 
                columns = df.columns, 
                index =df.columns, 
                )
            )
            .stack ()
            .rename(corr)
            .pipe(
                lambda s: s[
                    s.abs()> threshold 
                    ].reset_index()
                )
                .query("level_0 not in level_1")
        )

    return  c_df.style.format({corr :"{:2.f}"}) if fmt else c_df 
                      
def get_global_score(
    cvres: Dict[str, ArrayLike],
    ignore_convergence_problem: bool = False
) -> Tuple[float, float]:
    """
    Retrieve the global mean and standard deviation of test scores from 
    cross-validation results.

    This function computes the overall mean and standard deviation of test 
    scores from the results of cross-validation. It can also handle situations 
    where convergence issues might have occurred during model training, 
    depending on the `ignore_convergence_problem` flag.

    Parameters
    ----------
    cvres : Dict[str, np.ndarray]
        A dictionary containing the cross-validation results. Expected to have 
        keys 'mean_test_score' and 'std_test_score', with each key mapping to 
        an array of scores.
    ignore_convergence_problem : bool, default=False
        If True, ignores NaN values that might have resulted from convergence 
        issues during model training while calculating the mean. If False, NaN 
        values contribute to the final mean as NaN.

    Returns
    -------
    Tuple[float, float]
        A tuple containing two float values:
        - The first element is the mean of the test scores across all 
          cross-validation folds.
        - The second element is the mean of the standard deviations of the 
          test scores across all cross-validation folds.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> clf = DecisionTreeClassifier()
    >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5,
    ...                          scoring='accuracy', return_train_score=True)
    >>> cvres = {'mean_test_score': scores, 'std_test_score': np.std(scores)}
    >>> mean_score, mean_std = get_global_score(cvres)
    >>> print(f"Mean score: {mean_score}, Mean standard deviation: {mean_std}")

    Notes
    -----
    - The function is primarily designed to be used with results obtained from 
      scikit-learn's cross-validation functions like `cross_val_score`.
    - It is assumed that `cvres` contains keys 'mean_test_score' and 
      'std_test_score'.
    """
    if ignore_convergence_problem:
        mean_score = np.nanmean(cvres.get('mean_test_score'))
        mean_std = np.nanmean(cvres.get('std_test_score'))
    else:
        mean_score = np.mean( cvres.get('mean_test_score'))
        mean_std = np.mean(cvres.get('std_test_score'))

    return mean_score, mean_std

def format_model_score(
    model_score: Union[float, Dict[str, float]] = None,
    selected_estimator: Optional[str] = None
) -> None:
    """
    Formats and prints model scores.

    Parameters
    ----------
    model_score : float or Dict[str, float], optional
        The model score or a dictionary of model scores with estimator 
        names as keys.
    selected_estimator : str, optional
        Name of the estimator to format the score for. Used only if 
        `model_score` is a float.

    Example
    -------
    >>> from gofast.tools.mlutils import format_model_score
    >>> format_model_score({'DecisionTreeClassifier': 0.26, 'BaggingClassifier': 0.13})
    >>> format_model_score(0.75, selected_estimator='RandomForestClassifier')
    """

    print('-' * 77)
    if isinstance(model_score, dict):
        for estimator, score in model_score.items():
            formatted_score = round(score * 100, 3)
            print(f'> {estimator:<30}:{"Score":^10}= {formatted_score:^10} %')
    elif isinstance(model_score, float):
        estimator_name = selected_estimator if selected_estimator else 'Unknown Estimator'
        formatted_score = round(model_score * 100, 3)
        print(f'> {estimator_name:<30}:{"Score":^10}= {formatted_score:^10} %')
    else:
        print('Invalid model score format. Please provide a float or'
              ' a dictionary of scores.')
    print('-' * 77)
    
def stats_from_prediction(y_true, y_pred, verbose=False):
    """
    Generate statistical summaries and accuracy metrics from actual values (y_true)
    and predicted values (y_pred).

    Parameters
    ----------
    y_true : list or numpy.array
        Actual values.
    y_pred : list or numpy.array
        Predicted values.
    verbose : bool, optional
        If True, print the statistical summary and accuracy metrics.
        Default is False.

    Returns
    -------
    dict
        A dictionary containing statistical measures such 
        as MAE, MSE, RMSE, 
        and accuracy (if applicable).

    Examples
    --------
    >>> from gofast.tools.mlutils import stats_from_prediction 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> stats_from_prediction(y_true, y_pred, verbose=True)
    """
    from sklearn.metrics import ( 
        mean_absolute_error, mean_squared_error, accuracy_score) 
    # Calculating statistics
    check_consistent_length(y_true, y_pred )
    stats = {
        'mean': np.mean(y_pred),
        'median': np.median(y_pred),
        'std_dev': np.std(y_pred),
        'min': np.min(y_pred),
        'max': np.max(y_pred)
    }
    # add the metric stats 
    stats =dict ({
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        }, **stats, 
        )
    # Adding accuracy for classification tasks
    # Check if y_true and y_pred are categories task 
    if is_classification_task(y_true, y_pred ): 
    # if all(map(lambda x: x in [0, 1], y_true + y_pred)): #binary 
        stats['Accuracy'] = accuracy_score(y_true, y_pred)

    # Printing the results if verbose is True
    summary = MetricFormatter(
        title="Prediction Summary", descriptor="PredictStats", 
        **stats)
    if verbose:
        print(summary)
       
    return summary

def save_dataframes(
    *data: Union[DataFrame, Any],
    file_name_prefix: str = 'data',
    output_format: str = 'excel',
    sep: str = ',',
    start_index: int = 1
    ) -> None:
    """
    Saves multiple dataframes to Excel or CSV files, with each dataframe in a
    separate file.
    
    The files are named using a specified prefix and an index.

    Parameters
    ----------
    *data : Union[pd.DataFrame, Any]
        Variable number of arguments, where each argument is a dataframe or
        data that can be converted to a dataframe.
    file_name_prefix : str, optional
        Prefix for the output file names. Default is 'data'.
    output_format : str, optional
        Output format of the files. Can be 'excel' or 'csv'. Default is 'excel'.
    sep : str, optional
        Separator character for CSV output. Default is ','.
    start_index : int, optional
        Starting index for numbering the output files. Default is 1.

    Examples
    --------
    >>> from gofast.tools.mlutils import save_dataframes
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    >>> save_dataframes(df1, df2, file_name_prefix='mydata', output_format='csv')
    # This will create 'mydata_1.csv' for df1 and 'mydata_2.csv' for df2

    >>> save_dataframes(df1, output_format='excel', file_name_prefix='test')
    # This will create 'test_1.xlsx' containing df1
    """
    for index, df in enumerate(data, start=start_index):
        # Ensure the argument is a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Determine the file name
        file_name = f"{file_name_prefix}_{index}"
        if output_format == 'csv':
            df.to_csv(f"{file_name}.csv", sep=sep, index=False)
        elif output_format == 'excel':
            with pd.ExcelWriter(f"{file_name}.xlsx") as writer:
                df.to_excel(writer, index=False)
        else:
            raise ValueError("Unsupported output format. Choose 'excel' or 'csv'.")

def fetch_tgz(
    data_url: str,
    tgz_filename: str,
    data_path: Optional[str] = None,
    show_progress: bool = False
) -> None:
    """
    Fetches and extracts a 'tgz' file from a specified URL, optionally into 
    a target directory.

    If `data_path` is provided and does not exist, it is created. If `data_path`
    is not provided, a default directory named 'tgz_data' in the current working 
    directory is used and created if necessary.

    Parameters
    ----------
    data_url : str
        The URL where the .tgz file is located.
    tgz_filename : str
        The filename of the .tgz file to download.
    data_path : Optional[str], optional
        The absolute path to the directory where the .tgz file will be extracted.
        If None, uses a directory named ``'tgz_data'`` in the current working 
        directory.
    show_progress : bool, optional
        If True, displays a progress bar during the file download. Default is False.

    Examples
    --------
    >>> from gofast.tools.mlutils import fetch_tgz
    >>> fetch_tgz(
    ...     data_url="http://example.com/data",
    ...     tgz_filename="data.tgz",
    ...     show_progress=True
    ... )

    >>> fetch_tgz(
    ...     data_url="http://example.com/data",
    ...     tgz_filename="data.tgz",
    ...     data_path="/path/to/custom/data",
    ...     show_progress=True
    ... )

    Note
    ----
    The function requires `tqdm` library for showing the progress bar. Ensure
    that `tqdm` is installed if `show_progress` is set to True.
    """
    # Use a default data directory if none is provided
    data_path = data_path or os.path.join(os.getcwd(), 'tgz_data')
    
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

    tgz_path = os.path.join(data_path, tgz_filename)
    
    # Define a simple progress function, if needed
    def _progress(block_num, block_size, total_size):
        if show_progress:
            if tqdm is None:
                raise ImportError("`tqdm` library is required for progress output.")
            progress = tqdm(total=total_size, unit='iB', unit_scale=True, ascii=True, 
                            ncols= 100) 
            progress.n = block_num * block_size
            progress.last_print_n = progress.n
            progress.update()

    # Download the .tgz file
    urllib.request.urlretrieve(
        data_url, tgz_path, _progress if show_progress else None)

    # Extract the .tgz file
    with tarfile.open(tgz_path) as data_tgz:
        data_tgz.extractall(path=data_path)
    
    if show_progress:
        print("Download and extraction complete.")

def process_data_types(
    data,
    target_name=None, 
    exclude_target=False, 
    return_frame=False, 
    include_datetime=True,
    is_numeric_datetime=True, 
    return_target=False, 
    encode_cat_columns=False, 
    error='warn', 
    ):
    """
    Processes a DataFrame to separate numeric and categorical data.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    target_name : str or list of str, optional
        The name(s) of the target column(s) to exclude. If a single 
        target column is provided, it can be given as a string. 
        If multiple target columns are provided, they should be given 
        as a list of strings.
    exclude_target : bool, optional
        If `True`, exclude the target column(s) from the returned 
        numeric and categorical data. Default is `False`.
    return_frame : bool, optional
        If `True`, return DataFrames for numeric and categorical data. 
        If `False`, return lists of column names. Default is `False`.
    include_datetime : bool, optional
        If `True`, include datetime columns in the processing. 
        Default is `True`.
    is_numeric_datetime : bool, optional
        If `True`, consider datetime columns as numeric. Default is `True`.
    return_target : bool, optional
        If `True`, return the target data along with numeric and 
        categorical data. Default is `False`.
    encode_cat_columns : bool, optional
        If `True`, encode categorical variables. Default is `False`.
    error : str, optional
        How to handle errors when target columns are not found. Options 
        are ``'warn'``, ``'raise'``, ``'ignore'``. Default is ``'warn'``.

    Returns
    -------
    tuple
        If `return_frame` is `True`, returns a tuple of DataFrames 
        (`numeric_df`, `categorical_df`, `target_df`). 
        If `return_frame` is `False`, returns a tuple of lists 
        (`numeric_columns`, `categorical_columns`, `target_columns`).

    Raises
    ------
    ValueError
        If `exclude_target` is `True` and `target_name` is not provided.
        If specified target columns do not exist in the DataFrame 
        and ``error`` is set to ``'raise'``.

    Notes
    -----
    The function processes a DataFrame to separate numeric and 
    categorical data, with optional inclusion of datetime columns. 
    If `target_name` is provided, it can be excluded from the data 
    or returned separately. The function handles missing target 
    columns based on the ``error`` parameter.

    Mathematically, the separation of numeric and categorical columns 
    can be represented as:

    .. math:: 
        X_{\text{numeric}} = \{ x_i \mid x_i \in \mathbb{R}, \forall i \}
    
    .. math:: 
        X_{\text{categorical}} = \{ x_i \mid x_i \notin \mathbb{R}, \forall i \}

    where :math:`X` is the set of all columns in the DataFrame, 
    :math:`X_{\text{numeric}}` is the set of numeric columns, and 
    :math:`X_{\text{categorical}}` is the set of categorical columns.

    Examples
    --------
    >>> from gofast.tools.mlutils import process_data_types 
    >>> df = pd.DataFrame({'A': [1, 2, 3], 
    ...                    'B': ['a', 'b', 'c'], 
    ...                    'C': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])})
    >>> numeric_columns, categorical_columns = process_data_types(df)
    >>> numeric_columns
    ['A']
    >>> categorical_columns
    ['B']

    >>> numeric_df, categorical_df = process_data_types(df, return_frame=True)
    >>> numeric_df
       A
    0  1
    1  2
    2  3
    >>> categorical_df
       B
    0  a
    1  b
    2  c

    >>> numeric_columns, categorical_columns, target_columns = process_data_types(
    ...     df, target_name='A', exclude_target=True, return_target=True)
    >>> numeric_columns
    []
    >>> categorical_columns
    ['B']
    >>> target_columns
    ['A']

    See Also
    --------
    pandas.DataFrame.select_dtypes : Select columns by data type.
    pandas.concat : Concatenate pandas objects along a particular axis.

    References
    ----------
    .. [1] McKinney, Wes. "pandas: a foundational Python library for data 
       analysis and statistics." Python for High Performance and Scientific 
       Computing (2011): 1-9.
    """

    is_frame(data, df_only =True, raise_exception=True, objname='data')
    if exclude_target and not target_name:
        raise ValueError("If exclude_target is True, target_name must be provided.")
    
    if isinstance(target_name, str):
        target_name = [target_name]
    
    if target_name:
        missing_targets = [target for target in target_name if target not in data.columns]
        if missing_targets:
            if error == 'warn':
                warnings.warn(f"Target columns {missing_targets} not found in DataFrame.")
            elif error == 'raise':
                raise ValueError(f"Target columns {missing_targets} not found in DataFrame.")
            # Filter out missing targets
            target_name = [target for target in target_name if target in data.columns]
    
    numeric_df = data.select_dtypes([np.number])
    categorical_df = data.select_dtypes(None, [np.number])
    
    if include_datetime:
        datetime_df = data.select_dtypes(['datetime'])
        if is_numeric_datetime:
            numeric_df = pd.concat([numeric_df, datetime_df], axis=1)
        else:
            categorical_df = pd.concat([categorical_df, datetime_df], axis=1)
    
    target_df = pd.DataFrame()
    if target_name:
        target_df = data[target_name]
    
    if exclude_target and target_name:
        numeric_df = numeric_df.drop(columns=target_name, errors='ignore')
        categorical_df = categorical_df.drop(columns=target_name, errors='ignore')
    
    if encode_cat_columns and not categorical_df.empty: 
        categorical_df = soft_encoder ( categorical_df )
        
    if return_frame:
        if return_target:
            return numeric_df, categorical_df, target_df
        else:
            return numeric_df, categorical_df
    else:
        if return_target:
            return ( 
                numeric_df.columns.tolist(), categorical_df.columns.tolist(),
                target_df.columns.tolist()
                )
        else:
            return numeric_df.columns.tolist(), categorical_df.columns.tolist()

def discretize_categories(
    data: Union[DataFrame, Series],
    in_cat: str,
    new_cat: Optional[str] = None,
    divby: float = 1.5,
    higherclass: int = 5
) -> DataFrame:
    """
    Discretizes a numerical column in the DataFrame into categories. 
    
    Creating a new categorical column based on ceiling division and 
    an upper class limit.

    Parameters
    ----------
    data : DataFrame or Series
        Input data containing the column to be discretized.
    in_cat : str
        Column name in `data` used for generating the new categorical attribute.
    new_cat : str, optional
        Name for the newly created categorical column. If not provided, 
        a default name 'new_category' is used.
    divby : float, default=1.5
        The divisor used in the ceiling division to discretize the column values.
    higherclass : int, default=5
        The upper bound for the discretized categories. Values reaching this 
        class or higher are grouped into this single upper class.

    Returns
    -------
    DataFrame
        A new DataFrame including the newly created categorical column.

    Examples
    --------
    >>> from gofast.tools.mlutils import discretize_categories
    >>> df = pd.DataFrame({'age': [23, 45, 18, 27]})
    >>> discretized_df = discretize_categories(df, 'age', 'age_cat', divby=10, higherclass=3)
    >>> print(discretized_df)
       age  age_cat
    0   23      3.0
    1   45      3.0
    2   18      2.0
    3   27      3.0

    Note: The 'age_cat' column contains discretized categories based on the 
    'age' column.
    """
    if isinstance (data, pd.Series): 
        data = data.to_frame() 
    is_frame(data, df_only =True, raise_exception=True, objname='data')
    
    if new_cat is None:
        new_cat = 'new_category'
    
    # Discretize the specified column
    data[new_cat] = np.ceil(data[in_cat] / divby)
    # Apply upper class limit
    data[new_cat] = data[new_cat].where(data[new_cat] < higherclass, other=higherclass)
    
    return data

def stratify_categories(
    data: Union[DataFrame, ArrayLike],
    cat_name: str, 
    n_splits: int = 1, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[Union[DataFrame, ArrayLike], Union[DataFrame, ArrayLike]]: 
    """
    Perform stratified sampling on a dataset based on a specified categorical column.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        The dataset to be split. Can be a Pandas DataFrame or a NumPy ndarray.
        
    cat_name : str
        The name of the categorical column in 'data' used for stratified sampling. 
        This column must exist in 'data' if it's a DataFrame.
        
    n_splits : int, optional
        Number of re-shuffling & splitting iterations. Defaults to 1.
        
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to 0.2.
        
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
        Defaults to 42.

    Returns
    -------
    Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]
        A tuple containing the training and testing sets.

    Raises
    ------
    ValueError
        If 'cat_name' is not found in 'data' when 'data' is a DataFrame.
        If 'test_size' is not between 0 and 1.
        If 'n_splits' is less than 1.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100)
    ... })
    >>> train_set, test_set = stratify_categories(df, 'category')
    >>> train_set.shape, test_set.shape
    ((80, 3), (20, 3))
    """

    if isinstance(data, pd.DataFrame) and cat_name not in data.columns:
        raise ValueError(f"Column '{cat_name}' not found in the DataFrame.")

    if not (0 < test_size < 1):
        raise ValueError("Test size must be between 0 and 1.")

    if n_splits < 1:
        raise ValueError("Number of splits 'n_splits' must be at least 1.")

    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                   random_state=random_state)
    for train_index, test_index in split.split(data, data[cat_name] if isinstance(
            data, pd.DataFrame) else data[:, cat_name]):
        if isinstance(data, pd.DataFrame):
            strat_train_set = data.iloc[train_index]
            strat_test_set = data.iloc[test_index]
        else:  # Handle numpy arrays
            strat_train_set = data[train_index]
            strat_test_set = data[test_index]

    return strat_train_set, strat_test_set

def fetch_model(
    file: str,
    path: Optional[str] = None,
    default: bool = False,
    name: Optional[str] = None,
    verbose: int = 0
    ) -> Union[Dict[str, Any], List[Tuple[Any, Dict[str, Any], Any]]]:
    """
    Fetches a model saved using the Python pickle module or joblib module.

    Parameters
    ----------
    file : str
        The filename of the dumped model, saved using `joblib` or Python
        `pickle` module.
    path : Optional[str], optional
        The directory path containing the model file. If None, `file` is assumed
        to be the full path to the file.
    default : bool, optional
        If True, returns a list of tuples (model, best parameters, best scores)
        for each model in the file. If False, returns the entire contents of the
        file.
    name : Optional[str], optional
        The name of the specific model to retrieve from the file. If specified,
        only the named model and its parameters are returned.
    verbose : int, optional
        Verbosity level. More messages are displayed for values greater than 0.

    Returns
    -------
    Union[Dict[str, Any], List[Tuple[Any, Dict[str, Any], Any]]]
        Depending on the `default` flag:
        - If `default` is True, returns a list of tuples containing the model,
          best parameters, and best scores for each model in the file.
        - If `default` is False, returns the entire contents of the file, which
          could include multiple models and their respective information.

    Raises
    ------
    FileNotFoundError
        If the specified model file is not found.
    KeyError
        If `name` is specified but not found in the loaded model file.

    Examples
    --------
    >>> model_info = fetch_model('model.pkl', path='/models',
                                 name='RandomForest', default=True)
    >>> model, best_params, best_scores = model_info[0]
    """
    full_path = os.path.join(path, file) if path else file

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File {full_path!r} not found.")

    is_joblib = full_path.endswith('.pkl') or full_path.endswith('.joblib')
    load_func = joblib.load if is_joblib else pickle.load
    with open(full_path, 'rb') as f:
        model_data = load_func(f)

    if verbose > 0:
        lib_used = "joblib" if is_joblib else "pickle"
        print(f"Model loaded from {full_path!r} using {lib_used}.")

    if name:
        try:
            specific_model_data = model_data[name]
        except KeyError:
            available_models = list(model_data.keys())
            raise KeyError(f"Model name '{name}' not found. Available models: {available_models}")
        
        if default:
            if not isinstance(specific_model_data, dict):
                warnings.warn(
                    "The retrieved model data does not follow the expected structure. "
                    "Each model should be represented as a dictionary, with the model's "
                    "name as the key and its details (including 'best_params_' and "
                    "'best_scores_') as nested dictionaries. For instance: "
                    "`model_data = {'ModelName': {'best_params_': <parameters>, "
                    "'best_scores_': <scores>}}`. As the structure is unexpected, "
                    "returning the raw model data instead of the processed tuple."
                )
                return specific_model_data
            # Assuming model data structure for specific named model when default is True
            return [(specific_model_data, specific_model_data.get('best_params_', {}),
                     specific_model_data.get('best_scores_', {}))]
        return specific_model_data

    if default:
        # Assuming model data structure contains 'best_model', 'best_params_', and 'best_scores'
        return [(model, info.get('best_params_', {}), info.get('best_scores_', {})) 
                for model, info in model_data.items()]

    return model_data

def fetch_tgz2(
    tgz_file: str, 
    filename: str, 
    savefile: str = 'tgz', 
    rename_outfile: Optional[str] = None
) -> str:
    """
    Extracts a specified file from a tar archive and saves it to a given directory.

    Parameters
    ----------
    tgz_file : str
        The full path to the tar file.
    filename : str
        The specific file to extract from the tar archive.
    savefile : str, optional
        Directory to save the extracted file, by default 'tgz'.
    rename_outfile : str, optional
        New name for the extracted file, if renaming is desired.

    Returns
    -------
    str
        The full path to the extracted (and possibly renamed) file.

    Raises
    ------
    FileNotFoundError
        If the specified `tgz_file` or `filename` within the tar archive does not exist.

    Examples
    --------
    >>> from gofast.tools.mlutils import fetch_tgz
    >>> fetch_tgz('data/__tar.tgz/fmain.bagciv.data.tar.gz', 'dataset.csv',
    ...           'extracted', rename_outfile='main.bagciv.data.csv')
    """
    tgz_path = Path(tgz_file)
    save_path = Path(savefile)
    save_path.mkdir(parents=True, exist_ok=True)

    if not tgz_path.is_file():
        raise FileNotFoundError(f"Source {tgz_file!r} is not a valid file.")

    with tarfile.open(tgz_path) as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(filename)), None)
        if member:
            tar.extract(member, path=save_path)
            extracted_file_path = save_path / member.name
            final_file_path = save_path / (rename_outfile if rename_outfile else filename)
            if extracted_file_path != final_file_path:
                extracted_file_path.rename(final_file_path)
                if extracted_file_path.parent != save_path:
                    shutil.rmtree(extracted_file_path.parent, ignore_errors=True)
        else:
            raise FileNotFoundError(f"File {filename} not found in {tgz_file}.")

    print(f"--> '{final_file_path}' was successfully extracted from '{tgz_path.name}' "
          f"and saved to '{save_path}'.")
    return str(final_file_path)


     
def _assert_sl_target (target,  df=None, obj=None): 
    """ Check whether the target name into the dataframe for supervised 
    learning.
    
    :param df: dataframe pandas
    :param target: str or index of the supervised learning target name. 
    
    :Example: 
        
        >>> from gofast.tools.mlutils import _assert_sl_target
        >>> from gofast.datasets import fetch_data
        >>> data = fetch_data('Bagoue original').get('data=df')  
        >>> _assert_sl_target (target =12, obj=prepareObj, df=data)
        ... 'flow'
    """
    is_dataframe = isinstance(df, pd.DataFrame)
    is_ndarray = isinstance(df, np.ndarray)
    if is_dataframe :
        targets = smart_format(
            df.columns if df.columns is not None else [''])
    else:targets =''
    
    if target is None:
        nameObj=f'{obj.__class__.__name__}'if obj is not None else 'Base class'
        msg =''.join([
            f"{nameObj!r} {'basically' if obj is not None else ''}"
            " works with surpervised learning algorithms so the",
            " input target is needed. Please specify the target", 
            f" {'name' if is_dataframe else 'index' if is_ndarray else ''}", 
            " to take advantage of the full functionalities."
            ])
        if is_dataframe:
            msg += f" Select the target among {targets}."
        elif is_ndarray : 
            msg += f" Max columns size is {df.shape[1]}"

        warnings.warn(msg, UserWarning)
        _logger.warning(msg)
        
    if target is not None: 
        if is_dataframe: 
            if isinstance(target, str):
                if not target in df.columns: 
                    msg =''.join([
                        f"Wrong target value {target!r}. Please select "
                        f"the right column name: {targets}"])
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg)
                    target =None
            elif isinstance(target, (float, int)): 
                is_ndarray =True 
  
        if is_ndarray : 
            _len = len(df.columns) if is_dataframe else df.shape[1] 
            m_=f"{'less than' if target >= _len  else 'greater than'}" 
            if not isinstance(target, (float,int)): 
                msg =''.join([f"Wrong target value `{target}`!"
                              f" Object type is {type(df)!r}. Target columns", 
                              " index should be given instead."])
                warnings.warn(msg, category= UserWarning)
                _logger.warning(msg)
                target=None
            elif isinstance(target, (float,int)): 
                target = int(target)
                if not 0 <= target < _len: 
                    msg =f" Wrong target index. Should be {m_} {str(_len-1)!r}."
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg) 
                    target =None
                    
            if df is None: 
                wmsg = ''.join([
                    f"No data found! `{target}` does not fit any data set.", 
                      "Could not fetch the target name.`df` argument is None.", 
                      " Need at least the data `numpy.ndarray|pandas.dataFrame`",
                      ])
                warnings.warn(wmsg, UserWarning)
                _logger.warning(wmsg)
                target =None
                
            target = list(df.columns)[target] if is_dataframe else target
            
    return target

def _extract_target(
        X, target: Union[ArrayLike, int, str, List[Union[int, str]]]):
    """
    Extracts and validates the target variable(s) from the dataset.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        The dataset from which to extract the target variable(s).
    target : ArrayLike, int, str, or list of int/str
        The target variable(s) to be used. If an array-like or DataFrame, 
        it's directly used as `y`. If an int or str (or list of them), it 
        indicates the column(s) in `X` to be used as `y`.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        The dataset without the target column(s).
    y : pd.Series, np.ndarray, pd.DataFrame
        The target variable(s).
    target_names : list of str
        The names of the target variable(s) for labeling purposes.
    """
    target_names = []

    if isinstance(target, (list, pd.DataFrame)) or (
            isinstance(target, pd.Series) and not isinstance(X, np.ndarray)):
        if isinstance(target, list):  # List of column names or indexes
            if all(isinstance(t, str) for t in target):
                y = X[target]
                target_names = target
            elif all(isinstance(t, int) for t in target):
                y = X.iloc[:, target]
                target_names = [X.columns[i] for i in target]
            X = X.drop(columns=target_names)
        elif isinstance(target, pd.DataFrame):
            y = target
            target_names = target.columns.tolist()
            # Assuming target DataFrame is not part of X
        elif isinstance(target, pd.Series):
            y = target
            target_names = [target.name] if target.name else ["target"]
            if target.name and target.name in X.columns:
                X = X.drop(columns=target.name)
                
    elif isinstance(target, (int, str)):
        if isinstance(target, str):
            y = X.pop(target)
            target_names = [target]
        elif isinstance(target, int):
            y = X.iloc[:, target]
            target_names = [X.columns[target]]
            X = X.drop(columns=X.columns[target])
    elif isinstance(target, np.ndarray) or (
            isinstance(target, pd.Series) and isinstance(X, np.ndarray)):
        y = np.array(target)
        target_names = ["target"]
    else:
        raise ValueError("Unsupported target type or target does not match X dimensions.")
    
    check_consistent_length(X, y)
    
    return X, y, target_names

def smart_split(
    X, 
    target: Optional[Union[ArrayLike, int, str, List[Union[int, str]]]] = None,
    test_size: float = 0.2, 
    random_state: int = 42,
    stratify: bool = False,
    shuffle: bool = True,
    return_df: bool = False,
    **skws
) -> Union[
    Tuple[DataFrame, DataFrame], 
    Tuple[ArrayLike, ArrayLike],
    Tuple[DataFrame, DataFrame, Series, Series], 
    Tuple[DataFrame, DataFrame, DataFrame, DataFrame], 
    Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]
    ]:
    """
    Splits data into training and test sets, with the option to extract and 
    handle multiple target variables. 
    
    Function supports both single and multi-label targets and maintains 
    compatibility with pandas DataFrame and numpy ndarray.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        The input data to be split. This can be either feature data alone or 
        include the target column(s) if the `target` parameter is used to specify 
        target column(s) for extraction.
    target : int, str, list of int/str, pd.Series, pd.DataFrame, optional
        Specifies the target variable(s) for supervised learning problems. 
        It can be:
        - An integer or string specifying the column index or name in `X` to 
          be used as the target variable.
        - A list of integers or strings for multi-label targets.
        - A pandas Series or DataFrame directly specifying the target variable(s).
        If `target` is provided as an array-like object or DataFrame, its 
        length must match the number of samples in `X`.
    test_size : float, optional
        Represents the proportion of the dataset to include in the test split. 
        Must be between 0.0 and 1.0.
    random_state : int, optional
        Sets the seed for random operations, ensuring reproducible splits.
    stratify : bool, optional
        Ensures that the train and test sets have approximately the same 
        percentage of samples of each target class if set to True.
    shuffle : bool, optional
        Determines whether to shuffle the dataset before splitting. 
    return_df : bool, optional
        If True and `X` is a DataFrame, returns the splits as pandas DataFrames/Series. 
        Otherwise, returns numpy ndarrays.
    skws : dict
        Additional keyword arguments for `train_test_split`, allowing customization 
        of the split beyond the parameters explicitly mentioned here.

    Returns
    -------
    Depending on the inputs and `return_df`:
    - If `target` is not specified: X_train, X_test
    - If `target` is specified: X_train, X_test, y_train, y_test
    `X_train` and `X_test` are the splits of the input data, while `y_train` and 
    `y_test` are the splits of the target variable(s) if provided.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.mlutils import smart_split
    >>> data = pd.DataFrame({
    ...     'Feature1': [1, 2, 3, 4],
    ...     'Feature2': [4, 3, 2, 1],
    ...     'Target': [0, 1, 0, 1]
    ... })
    >>> # Single target specified as a column name
    >>> X_train, X_test, y_train, y_test = smart_split(
    ... data, target='Target', return_df=True)
    >>> print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    """
    if target is not None:
        X, y, target_names = _extract_target(X, target)
    else:
        y, target_names = None, []

    stratify_param = y if stratify and y is not None else None
    if y is not None: 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle, 
            stratify=stratify_param, **skws)
    else: 
        X_train, X_test= train_test_split(
            X, test_size=test_size, random_state=random_state, shuffle=shuffle, 
            stratify=stratify_param, **skws)

    if return_df and isinstance(X, pd.DataFrame):
        X_train, X_test = pd.DataFrame(X_train, columns=X.columns
                                       ), pd.DataFrame(X_test, columns=X.columns)
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y_train, y_test = pd.DataFrame(
                    y_train, columns=target_names), pd.DataFrame(
                        y_test, columns=target_names)
            else:
                y_train, y_test = pd.Series(
                    y_train, name=target_names[0]), pd.Series(
                        y_test, name=target_names[0])

    return (X_train, X_test, y_train, y_test) if y is not None else (X_train, X_test)

def handle_imbalance(
    X, y=None, strategy='oversample', 
    random_state=42, 
    target_col='target'
    ):
    """
    Handles imbalanced datasets by either oversampling the minority class or 
    undersampling the majority class.
    
    It supports inputs as pandas DataFrame/Series, numpy arrays, and allows 
    specifying the target variable either as a separate argument or as part 
    of the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame, np.ndarray
        The features of the dataset. If `y` is None, `X` is expected to include 
        the target variable.
    y : pd.Series, np.ndarray, optional
        The target variable of the dataset. If None, `target_col` must be 
        specified, and `X` must be a DataFrame containing the target.
    strategy : str, optional
        The strategy to apply for handling imbalance: 'oversample' or 
        'undersample'. Default is 'oversample'.
    random_state : int, optional
        The random state for reproducible results. Default is 42.
    target_col : str, optional
        The name of the target column in `X` if `X` is a DataFrame and 
        `y` is None. Default is 'target'.
    
    Returns
    -------
    X_resampled, y_resampled : Resampled features and target variable.
        The types of `X_resampled` and `y_resampled` match the input types.

    Examples
    --------
    Using with numpy arrays:
    
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import handle_imbalance
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([0, 1, 0])
    >>> X_resampled, y_resampled = handle_imbalance(X, y)
    >>> print(X_resampled.shape, y_resampled.shape)
    (3, 2) (3,)
    Using with pandas DataFrame (including target column):
    
    >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [2, 3, 4], 'target': [0, 1, 0]})
    >>> X_resampled, y_resampled = handle_imbalance(df, target_col='target')
    >>> print(X_resampled.shape, y_resampled.value_counts())

    Using with pandas DataFrame and Series:
    
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [2, 3, 4]})
    >>> y = pd.Series([0, 1, 0], name='target')
    >>> X_resampled, y_resampled = handle_imbalance(X, y)
    >>> print(X_resampled.shape, y_resampled.value_counts())
    """
    if y is None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` must be a DataFrame when `y` is None.")
        exist_features(X, target_col, name =target_col)
        y = X[target_col]
        X = X.drop(target_col, axis=1)

    if not _is_arraylike_1d(y): 
        raise TypeError ("Check `y`. Expect one-dimensional array.")

    if not isinstance (y, pd.Series) : 
        # squeeze y to keep 1d array for skipping value error 
        # when constructing pd.Series and 
        # ensure `y` is a Series with the correct name for 
        # easy concatenation and manipulation
        y =pd.Series (np.squeeze (y), name =target_col ) 
    else : target_col = y.name # reset the default target_col 
 
    # Check consistent length 
    check_consistent_length(X, y )
    
    if isinstance(X, pd.DataFrame):
        data = pd.concat([X, y], axis=1)
    elif isinstance (X, np.ndarray ): 
        # Ensure `data` from `X` is a DataFrame with correct 
        # column names for subsequent operations
        data = pd.DataFrame(
            np.column_stack([X, y]), columns=[*X.columns, y.name]
            if isinstance(X, pd.DataFrame) else [
                    *[f"feature_{i}" for i in range(X.shape[1])], y.name])
    else: 
        TypeError("Unsupported type for X. Must be np.ndarray or pd.DataFrame.")
        
    # Identify majority and minority classes
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()

    # Correctly determine the number of samples for resampling
    num_majority = y.value_counts()[majority_class]
    num_minority = y.value_counts()[minority_class]

    # Apply resampling strategy
    if strategy == 'oversample':
        minority_upsampled = resample(
            data[data[target_col] == minority_class],
            replace=True,
            n_samples=num_majority,
            random_state=random_state
            )
        resampled = pd.concat(
            [data[data[target_col] == majority_class], minority_upsampled])
    elif strategy == 'undersample':
        majority_downsampled = resample(
            data[data[target_col] == majority_class],
            replace=False,
            n_samples=num_minority,
            random_state=random_state
            )
        resampled = pd.concat(
            [data[data[target_col] == minority_class], majority_downsampled])

    # Prepare the output
    X_resampled = resampled.drop(target_col, axis=1)
    y_resampled = resampled[target_col]

    # Convert back to the original input type if necessary
    if isinstance(X, np.ndarray):
        X_resampled = X_resampled.to_numpy()
        y_resampled = y_resampled.to_numpy()

    return X_resampled, y_resampled

def soft_data_split(
    X, y=None, *,
    test_size=0.2,
    target_column=None,
    random_state=42,
    extract_target=False,
    **split_kwargs
):
    """
    Splits data into training and test sets, optionally extracting a 
    target column.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data to split. If `extract_target` is True, a target column can be
        extracted from `X`.
    y : array-like, optional
        Target variable array. If None and `extract_target` is False, `X` is
        split without a target variable.
    test_size : float, optional
        Proportion of the dataset to include in the test split. Should be
        between 0.0 and 1.0. Default is 0.2.
    target_column : int or str, optional
        Index or column name of the target variable in `X`. Used only if
        `extract_target` is True.
    random_state : int, optional
        Controls the shuffling for reproducible output. Default is 42.
    extract_target : bool, optional
        If True, extracts the target variable from `X`. Default is False.
    split_kwargs : dict, optional
        Additional keyword arguments to pass to `train_test_split`.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split data arrays.

    Raises
    ------
    ValueError
        If `target_column` is not found in `X` when `extract_target` is True.

    Example
    -------
    >>> from gofast.datasets import fetch_data
    >>> data = fetch_data('Bagoue original')['data']
    >>> X, XT, y, yT = split_data(data, extract_target=True, target_column='flow')
    """

    if extract_target:
        if isinstance(X, pd.DataFrame) and target_column in X.columns:
            y = X[target_column]
            X = X.drop(columns=target_column)
        elif hasattr(X, '__array__') and isinstance(target_column, int):
            y = X[:, target_column]
            X = np.delete(X, target_column, axis=1)
        else:
            raise ValueError(f"Target column {target_column!r} not found in X.")

    if y is not None:
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, **split_kwargs)
    else:
        return  train_test_split(
            X, test_size=test_size,random_state=random_state, **split_kwargs)
 
def load_model(
    file_path: str, *,
    retrieve_default: bool = True,
    model_name: Optional[str] = None,
    storage_format: Optional[str] = None
    ) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """
    Loads a saved model or data using Python's pickle or joblib module.

    Parameters
    ----------
    file_path : str
        The path to the saved model file. Supported formats are `.pkl` and `.joblib`.
    retrieve_default : bool, optional, default=True
        If True, returns the model along with its best parameters. If False,
        returns the entire contents of the saved file.
    model_name : Optional[str], optional
        The name of the specific model to retrieve from the saved file. If None,
        the entire file content is returned.
    storage_format : Optional[str], optional
        The format used for saving the file. If None, the format is inferred
        from the file extension. Supported formats are 'joblib' and 'pickle'.

    Returns
    -------
    Union[Any, Tuple[Any, Dict[str, Any]]]
        The loaded model or a tuple of the model and its parameters, depending
        on the `retrieve_default` value.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If the specified model name is not found in the file.
    ValueError
        If the storage format is not supported or if the loaded data is not
        a dictionary when a model name is specified.

    Example
    -------
    >>> model, params = load_model('path_to_file.pkl', model_name='SVC')
    >>> print(model)
    >>> print(params)
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    storage_format = storage_format or os.path.splitext(file_path)[-1].lower().lstrip('.')
    if storage_format not in {"joblib", "pickle"}:
        raise ValueError(f"Unsupported storage format '{storage_format}'. "
                         "Use 'joblib' or 'pickle'.")

    load_func = joblib.load if storage_format == 'joblib' else pickle.load
    with open(file_path, 'rb') as file:
        loaded_data = load_func(file)

    if model_name:
        if not isinstance(loaded_data, dict):
            warnings.warn(
                f"Expected loaded data to be a dictionary for model name retrieval. "
               f"Received type '{type(loaded_data).__name__}'. Returning loaded data.")
            return loaded_data

        model_info = loaded_data.get(model_name)
        if model_info is None:
            available = ', '.join(loaded_data.keys())
            raise KeyError(f"Model '{model_name}' not found. Available models: {available}")

        if retrieve_default:
            if not isinstance(model_info, dict):
                # Check if 'best_model_' and 'best_params_' are among the keys
                main_keys = [key for key in loaded_data if key in (
                    'best_model_', 'best_params_')]
                if len(main_keys) == 0:
                    warnings.warn(
                    "The structure of the default model data is not correctly "
                    "formatted. Expected 'best_model_' and 'best_params_' to be "
                    "present within a dictionary keyed by the model's name. Each key "
                    "should map to a dictionary containing the model itself and its "
                    "parameters, for example: `{'ModelName': {'best_model_': <Model>, "
                    "'best_params_': <Parameters>}}`. Since the expected keys were "
                    "not found, returning the unprocessed model data."
                    )
                    return model_info
                else:
                    # Extract 'best_model_' and 'best_params_' from loaded_data
                    best_model = loaded_data.get('best_model_', None)
                    best_params = loaded_data.get('best_params_', {})
            else:
                # Direct extraction from model_info if it's properly structured
                best_model = model_info.get('best_model_', None)
                best_params = model_info.get('best_params_', {})

            return best_model, best_params

        return model_info

    return loaded_data
     
def bi_selector (d,  features =None, return_frames = False,
                 parse_features:bool=... ):
    """ Auto-differentiates the numerical from categorical attributes.
    
    This is usefull to select the categorial features from the numerical 
    features and vice-versa when we are a lot of features. Enter features 
    individually become tiedous and a mistake could probably happenned. 
    
    Parameters 
    ------------
    d: pandas dataframe 
        Dataframe pandas 
    features : list of str
        List of features in the dataframe columns. Raise error is feature(s) 
        does/do not exist in the frame. 
        Note that if `features` is ``None``, it returns the categorical and 
        numerical features instead. 
        
    return_frames: bool, default =False 
        return the difference columns (features) from the given features  
        as a list. If set to ``True`` returns bi-frames composed of the 
        given features and the remaining features. 
        
    Returns 
    ----------
    - Tuple ( list, list)
        list of features and remaining features 
    - Tuple ( pd.DataFrame, pd.DataFrame )
        List of features and remaing features frames.  
            
    Example 
    --------
    >>> from gofast.tools.mlutils import bi_selector 
    >>> from gofast.datasets import load_hlogs 
    >>> data = load_hlogs().frame # get the frame 
    >>> data.columns 
    >>> Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter', 'aquifer_group',
           'pumping_level', 'aquifer_thickness', 'hole_depth_before_pumping',
           'hole_depth_after_pumping', 'hole_depth_loss', 'depth_starting_pumping',
           'pumping_depth_at_the_end', 'pumping_depth', 'section_aperture', 'k',
           'kp', 'r', 'rp', 'remark'],
          dtype='object')
    >>> num_features, cat_features = bi_selector (data)
    >>> num_features
    ...['gamma_gamma',
         'depth_top',
         'aquifer_thickness',
         'pumping_depth_at_the_end',
         'section_aperture',
         'remark',
         'depth_starting_pumping',
         'hole_depth_before_pumping',
         'rp',
         'hole_depth_after_pumping',
         'hole_depth_loss',
         'depth_bottom',
         'sp',
         'pumping_depth',
         'kp',
         'resistivity',
         'short_distance_gamma',
         'r',
         'natural_gamma',
         'layer_thickness',
         'k',
         'well_diameter']
    >>> cat_features 
    ... ['hole_id', 'strata_name', 'rock_name', 'aquifer_group', 
         'pumping_level']
    """
    parse_features, = ellipsis2false(parse_features )
    _assert_all_types( d, pd.DataFrame, objname=" unfunc'bi-selector'")
    if features is None: 
        d, diff_features, features = to_numeric_dtypes(
            d,  return_feature_types= True ) 
    if features is not None: 
        features = is_iterable(features, exclude_string= True, transform =True, 
                               parse_string=parse_features )
        diff_features = is_in_if( d.columns, items =features, return_diff= True )
        if diff_features is None: diff_features =[]
    return  ( diff_features, features ) if not return_frames else  (
        d [diff_features] , d [features ] ) 

def make_pipe(
    X, 
    y =None, *,   
    num_features = None, 
    cat_features=None, 
    label_encoding='LabelEncoder', 
    scaler = 'StandardScaler' , 
    missing_values =np.nan, 
    impute_strategy = 'median', 
    sparse_output=True, 
    for_pca =False, 
    transform =False, 
    ): 
    """ make a pipeline to transform data at once. 
    
    make a quick pipeline is usefull to fast preprocess the data at once 
    for quick prediction. 
    
    Work with a pandas dataframe. If `None` features is set, the numerical 
    and categorial features are automatically retrieved. 
    
    Parameters
    ---------
    X : pandas dataframe of shape (n_samples, n_features)
        The input samples. Use ``dtype=np.float32`` for maximum
        efficiency. Sparse matrices are also supported, use sparse
        ``csc_matrix`` for maximum efficiency.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    num_features: list or str, optional 
        Numerical features put on the list. If `num_features` are given  
        whereas `cat_features` are ``None``, `cat_features` are figured out 
        automatically.
    cat_features: list of str, optional 
        Categorial features put on the list. If `num_features` are given 
        whereas `num_features` are ``None``, `num_features` are figured out 
        automatically.
    label_encoding: callable or str, default='sklearn.preprocessing.LabelEncoder'
        kind of encoding used to encode label. This assumes 'y' is supplied. 
    scaler: callable or str , default='sklearn.preprocessing.StandardScaler'
        kind of scaling used to scaled the numerical data. Note that for 
        the categorical data encoding, 'sklearn.preprocessing.OneHotEncoder' 
        is implemented  under the hood instead. 
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    
    impute_strategy : str, default='mean'
        The imputation strategy.
    
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
    
           strategy="constant" for fixed value imputation.
           
    sparse_output : bool, default=False
        Is used when label `y` is given. Binarize labels in a one-vs-all 
        fashion. If ``True``, returns array from transform is desired to 
        be in sparse CSR format.
        
    for_pca:bool, default=False, 
        Transform data for principal component ( PCA) analysis. If set to 
        ``True``, :class:`sklearn.preprocessing.OrdinalEncoder`` is used insted 
        of :class:sklearn.preprocessing.OneHotEncoder``. 
        
    transform: bool, default=False, 
        Tranform data inplace rather than returning the naive pipeline. 
        
    Returns
    ---------
    full_pipeline: :class:`gofast.exlib.sklearn.FeatureUnion`
        - Full pipeline composed of numerical and categorical pipes 
    (X_transformed &| y_transformed):  {array-like, sparse matrix} of \
        shape (n_samples, n_features)
        - Transformed data. 
        
        
    Examples 
    ---------
    >>> from gofast.tools.mlutils import make_naive_pipe 
    >>> from gofast.datasets import load_hlogs 
    
    (1) Make a naive simple pipeline  with RobustScaler, StandardScaler 
    >>> from gofast.exlib.sklearn import RobustScaler 
    >>> X_, y_ = load_hlogs (as_frame=True )# get all the data  
    >>> pipe = make_naive_pipe(X_, scaler =RobustScaler ) 
    
    (2) Transform X in place with numerical and categorical features with 
    StandardScaler (default). Returned CSR matrix 
    
    >>> make_naive_pipe(X_, transform =True )
    ... <181x40 sparse matrix of type '<class 'numpy.float64'>'
    	with 2172 stored elements in Compressed Sparse Row format>

    """
    
    from ..transformers import DataFrameSelector
    
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not hasattr (X, '__array__'):
        raise TypeError(f"'make_naive_pipe' not supported {type(X).__name__!r}."
                        " Expects X as 'pandas.core.frame.DataFrame' object.")
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="Array for transforming X or making naive pipeline"
        )
    if not hasattr (X, "columns"):
        # create naive column for 
        # Dataframe selector 
        X = pd.DataFrame (
            X, columns = [f"naive_{i}" for i in range (X.shape[1])]
            )
    #-> Encode y if given
    if y is not None: 
        # if (label_encoding =='labelEncoder'  
        #     or get_estimator_name(label_encoding) =='LabelEncoder'
        #     ): 
        #     enc =LabelEncoder()
        if  ( label_encoding =='LabelBinarizer' 
                or get_estimator_name(label_encoding)=='LabelBinarizer'
               ): 
            enc =LabelBinarizer(sparse_output=sparse_output)
        else: 
            label_encoding =='labelEncoder'
            enc =LabelEncoder()
            
        y= enc.fit_transform(y)
    #set features
    if num_features is not None: 
        cat_features, num_features  = bi_selector(
            X, features= num_features 
            ) 
    elif cat_features is not None: 
        num_features, cat_features  = bi_selector(
            X, features= cat_features 
            )  
    if ( cat_features is None 
        and num_features is None 
        ): 
        num_features , cat_features = bi_selector(X ) 
    # assert scaler value 
    if get_estimator_name (scaler)  in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler )) 
    elif ( any ( [v.lower().find (str(scaler).lower()) >=0
                  for v in sc.keys()])
          ):  
        for k, v in sc.items () :
            if k.lower().find ( str(scaler).lower() ) >=0: 
                scaler = v ; break 
    else : 
        msg = ( f"Supports {smart_format( sc.keys(), 'or')} or "
                "other scikit-learn scaling objects, got {!r}" 
                )
        if hasattr (scaler, '__module__'): 
            name = getattr (scaler, '__module__')
            if getattr (scaler, '__module__') !='sklearn.preprocessing._data':
                raise ValueError (msg.format(name ))
        else: 
            name = scaler.__name__ if callable (scaler) else (
                scaler.__class__.__name__ ) 
            raise ValueError (msg.format(name ))
    # make pipe 
    npipe = [
            ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=impute_strategy)),                
            ('scalerObj', scaler() if callable (scaler) else scaler ), 
            ]
    
    if len(num_features)!=0 : 
       npipe.insert (
            0,  ('selectorObj', DataFrameSelector(columns= num_features))
            )

    num_pipe=Pipeline(npipe)
    
    if for_pca : 
        encoding=  ('OrdinalEncoder', OrdinalEncoder())
    else:  encoding =  (
        'OneHotEncoder', OneHotEncoder())
        
    cpipe = [
        encoding
        ]
    if len(cat_features)!=0: 
        cpipe.insert (
            0, ('selectorObj', DataFrameSelector(columns= cat_features))
            )

    cat_pipe = Pipeline(cpipe)
    # make transformer_list 
    transformer_list = [
        ('num_pipeline', num_pipe),
        ('cat_pipeline', cat_pipe), 
        ]

    #remove num of cat pipe if one of them is 
    # missing in the data 
    if len(cat_features)==0: 
        transformer_list.pop(1) 
    if len(num_features )==0: 
        transformer_list.pop(0)
        
    full_pipeline =FeatureUnion(transformer_list=transformer_list) 
    
    return  ( full_pipeline.fit_transform (X) if y is None else (
        full_pipeline.fit_transform (X), y ) 
             ) if transform else full_pipeline

@ensure_pkg (
    "imblearn", 
    partial_check=True, 
    condition="balance_classes", 
    extra= (
        "Synthetic Minority Over-sampling Technique (SMOTE) cannot be used."
        " Note,`imblearn` is actually a shorthand for ``imbalanced-learn``."
        ), 
   )
def build_data_preprocessor(
    X: Union [NDArray, DataFrame], 
    y: Optional[ArrayLike] = None, *,  
    num_features: Optional[List[str]] = None, 
    cat_features: Optional[List[str]] = None, 
    custom_transformers: Optional[List[Tuple[str, TransformerMixin]]] = None,
    label_encoding: Union[str, TransformerMixin] = 'LabelEncoder', 
    scaler: Union[str, TransformerMixin] = 'StandardScaler', 
    missing_values: Union[int, float, str, None] = np.nan, 
    impute_strategy: str = 'median', 
    feature_interaction: bool = False,
    dimension_reduction: Optional[Union[str, TransformerMixin]] = None,
    feature_selection: Optional[Union[str, TransformerMixin]] = None,
    balance_classes: bool = False,
    advanced_imputation: Optional[TransformerMixin] = None,
    verbose: bool = False,
    output_format: str = 'array',
    transform: bool = False,
    **kwargs: Any
) -> Any:
    """
    Create a preprocessing pipeline for data transformation and feature engineering.

    Function constructs a pipeline to preprocess data for machine learning tasks, 
    accommodating a variety of transformations including scaling, encoding, 
    and dimensionality reduction. It supports both numerical and categorical data, 
    and can incorporate custom transformations.

    Parameters
    ----------
    X : np.ndarray or DataFrame
        Input features dataframe or arraylike. Must be two dimensional array.
    y : array-like, optional
        Target variable. Required for supervised learning tasks.
    num_features : list of str, optional
        List of numerical feature names. If None, determined automatically.
    cat_features : list of str, optional
        List of categorical feature names. If None, determined automatically.
    custom_transformers : list of tuples, optional
        Custom transformers to be included in the pipeline. Each tuple should 
        contain ('name', transformer_instance).
    label_encoding : str or transformer, default 'LabelEncoder'
        Encoder for the target variable. Accepts standard scikit-learn encoders 
        or custom encoder objects.
    scaler : str or transformer, default 'StandardScaler'
        Scaler for numerical features. Accepts standard scikit-learn scalers 
        or custom scaler objects.
    missing_values : int, float, str, np.nan, None, default np.nan
        Placeholder for missing values for imputation.
    impute_strategy : str, default 'median'
        Imputation strategy. Options: 'mean', 'median', 'most_frequent', 'constant'.
    feature_interaction : bool, default False
        If True, generate polynomial and interaction features.
    dimension_reduction : str or transformer, optional
        Dimensionality reduction technique. Accepts 'PCA', 't-SNE', or custom object.
    feature_selection : str or transformer, optional
        Feature selection method. Accepts 'SelectKBest', 'SelectFromModel', or custom object.
    balance_classes : bool, default False
        If True, balance classes in classification tasks.
    advanced_imputation : transformer, optional
        Advanced imputation technique like KNNImputer or IterativeImputer.
    verbose : bool, default False
        Enable verbose output.
    output_format : str, default 'array'
        Desired output format: 'array' or 'dataframe'.
    transform : bool, default False
        If True, apply the pipeline to the data immediately and return transformed data.

    Returns
    -------
    full_pipeline : Pipeline or (X_transformed, y_transformed)
        The constructed preprocessing pipeline, or transformed data if `transform` is True.

    Examples
    --------
    >>> from gofast.tools.mlutils import build_data_preprocessor
    >>> from gofast.datasets import load_hlogs
    >>> X, y = load_hlogs(as_frame=True, return_X_y=True)
    >>> pipeline = build_data_preprocessor(X, y, scaler='RobustScaler')
    >>> X_transformed = pipeline.fit_transform(X)
    
    """
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not isinstance (X, pd.DataFrame): 
        # create fake dataframe for handling columns features 
        X= pd.DataFrame(X)
    # assert scaler value 
    if get_estimator_name (scaler) in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler ))() 
        
    # Define numerical and categorical pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy, missing_values=missing_values)),
        ('scaler', StandardScaler() if scaler == 'StandardScaler' else scaler)
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', missing_values=missing_values)),
        ('encoder', OneHotEncoder() if label_encoding == 'LabelEncoder' else label_encoding)
    ])

    # Determine automatic feature selection if not provided
    if num_features is None and cat_features is None:
        num_features = make_column_selector(dtype_include=['int', 'float'])(X)
        cat_features = make_column_selector(dtype_include='object')(X)

    # Feature Union for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_features),
            ('cat', categorical_pipeline, cat_features)
        ])

    # Add custom transformers if any
    if custom_transformers:
        for name, transformer in custom_transformers:
            preprocessor.named_transformers_[name] = transformer

    # Feature interaction, selection, and dimension reduction
    steps = [('preprocessor', preprocessor)]
    
    if feature_interaction:
        steps.append(('interaction', PolynomialFeatures()))
    if feature_selection:
        steps.append(('feature_selection', SelectKBest() 
                      if feature_selection == 'SelectKBest' else feature_selection))
    if dimension_reduction:
        steps.append(('dim_reduction', PCA() if dimension_reduction == 'PCA' 
                      else dimension_reduction))

    # Final pipeline
    pipeline = Pipeline(steps)

   # Advanced imputation logic if required
    if advanced_imputation:
        from sklearn.experimental import enable_iterative_imputer # noqa
        from sklearn.impute import IterativeImputer
        if advanced_imputation == 'IterativeImputer':
            steps.insert(0, ('advanced_imputer', IterativeImputer(
                estimator=RandomForestClassifier(), random_state=42)))
        else:
            steps.insert(0, ('advanced_imputer', advanced_imputation))

    # Final pipeline before class balancing
    pipeline = Pipeline(steps)

    # Class balancing logic if required
    if balance_classes and y is not None:
        if str(balance_classes).upper() == 'SMOTE':
            from imblearn.over_sampling import SMOTE
            # Note: SMOTE works on numerical data, so it's applied after initial pipeline
            pipeline = Pipeline([('preprocessing', pipeline), (
                'smote', SMOTE(random_state=42))])

    # Transform data if transform flag is set
    # if transform:
    output_format = output_format or 'array' # force none to hold array
    if str(output_format) not in ('array', 'dataframe'): 
        raise ValueError(f"Invalid '{output_format}', expect 'array' or 'dataframe'.")
        
    return _execute_transformation(
        pipeline, X, y, transform, output_format, label_encoding)

def _execute_transformation(
        pipeline, X, y, transform, output_format, label_encoding):
    """ # Transform data if transform flag is set or return pipeline"""
    if transform:
        X_transformed = pipeline.fit_transform(X)
        if y is not None:
            y_transformed = _transform_target(y, label_encoding) if label_encoding else y
            return (X_transformed, y_transformed) if output_format == 'array' else (
                pd.DataFrame(X_transformed), pd.Series(y_transformed))
        
        return X_transformed if output_format == 'array' else pd.DataFrame(X_transformed)
    
    return pipeline

def _transform_target(y, label_encoding:Union [BaseEstimator, TransformerMixin] ):
    if label_encoding == 'LabelEncoder':
        encoder = LabelEncoder()
        return encoder.fit_transform(y)
    elif isinstance(label_encoding, (BaseEstimator, TransformerMixin)):
        return label_encoding.fit_transform(y)
    else:
        raise ValueError("Unsupported label_encoding value: {}".format(label_encoding)) 
         
def select_feature_importances(
        clf, X, y=None, *, threshold=0.1, prefit=True, 
        verbose=0, return_selector=False, **kwargs
        ):
    """
    Select features based on importance thresholds after model fitting.
    
    Parameters
    ----------
    clf : estimator object
        The estimator from which the feature importances are derived. Must have
        either `feature_importances_` or `coef_` attributes after fitting, unless
        `importance_getter` is specified in `kwargs`.
        
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples.
        
    y : array-like of shape (n_samples,), default=None
        The target values (class labels) as integers or strings.
        
    threshold : float, default=0.1
        The threshold value to use for feature selection. Features with importance
        greater than or equal to this value are retained.
        
    prefit : bool, default=True
        Whether the estimator is expected to be prefit. If `True`, `clf` should
        already be fitted; otherwise, it will be fitted on `X` and `y`.
        
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    return_selector : bool, default=False
        Whether to return the selector object instead of the transformed data.
        
    **kwargs : additional keyword arguments
        Additional arguments passed to `SelectFromModel`.
    
    Returns
    -------
    X_selected or selector : array or SelectFromModel object
        The selected features in `X` if `return_selector` is False, or the
        selector object itself if `return_selector` is True.
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.tools.mlutils import select_feature_importances
    >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=3)
    >>> clf = RandomForestClassifier()
    >>> X_selected = select_feature_importances(clf, X, y, threshold="mean", prefit=False)
    >>> X_selected.shape
    (1000, n_selected_features)
    
    Using `return_selector=True` to get the selector object:
    
    >>> selector = select_feature_importances(
        clf, X, y, threshold="mean", prefit=False, return_selector=True)
    >>> selector.get_support()
    array([True, False, ..., True])
    """
    # Check if the classifier is fitted based on the presence of attributes
    if not prefit and (hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_')):
        warnings.warn(f"The estimator {clf.__class__.__name__} appears to be fitted. "
                      "Consider setting `prefit=True` or refit the estimator.",UserWarning)
    try:threshold = float(threshold ) 
    except: pass 

    selector = SelectFromModel(clf, threshold=threshold, prefit=prefit, **kwargs)
    
    if not prefit:
        selector.fit(X, y)
    
    if verbose:
        n_features = selector.transform(X).shape[1]
        print(f"Number of features meeting the threshold={threshold}: {n_features}")
    
    return selector if return_selector else selector.transform(X)

@SmartProcessor(fail_silently=True, param_name ="skip_columns")
def soft_imputer(
    X, 
    strategy='mean', 
    missing_values=np.nan, 
    fill_value=None, 
    drop_features=False, 
    mode=None, 
    copy=True, 
    verbose=0, 
    add_indicator=False,
    keep_empty_features=False, 
    skip_columns=None, 
    **kwargs
    ):
    """
    Impute missing values in a dataset, optionally dropping features and handling 
    both numerical and categorical data.

    This function extends the functionality of scikit-learn's SimpleImputer to 
    support dropping specified features and a ``bi-impute`` mode for handling 
    both numerical and categorical data. It ensures API consistency with 
    scikit-learn's transformers and allows for flexible imputation strategies.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        The input data to impute.
        
    strategy : str, default='mean'
        The imputation strategy:
        - 'mean': Impute using the mean of each column. Only for numeric data.
        - 'median': Impute using the median of each column. Only for numeric data.
        - 'most_frequent': Impute using the most frequent value of each column. 
          For numeric and categorical data.
        - 'constant': Impute using the specified `fill_value`.
        
    missing_values : int, float, str, np.nan, None, or pd.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
        
    fill_value : str or numerical value, default=None
        When `strategy` == 'constant', `fill_value` is used to replace all
        occurrences of `missing_values`. If left to the default, `fill_value` 
        will be 0 when imputing numerical data and 'missing_value' for strings 
        or object data types.
        For 'constant' strategy, specifies the value used for replacement.
        In 'bi-impute' mode, allows specifying separate fill values for
        numerical and categorical data using a delimiter from the set
        {"__", "--", "&", "@", "!"}. For example, "0.5__missing" indicates
        0.5 as fill value for numerical data and "missing" for categorical.
        
    drop_features : bool or list, default=False
        If True, drops all categorical features before imputation. If a list, 
        drops specified features.
        
    mode : str, optional
        If set to 'bi-impute', imputes both numerical and categorical features 
        and returns a single imputed dataframe. Only 'bi-impute' is supported.
        
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
        
    verbose : int, default=0
        Controls the verbosity of the imputer.
        
    add_indicator : bool, default=False
        If True, a `MissingIndicator` transform will be added to the output 
        of the imputer's transform.
        
    keep_empty_features : bool, default=False
        If True, features that are all missing when `fit` is called are 
        included in the transform output.
        
    skip_columns : list of str or int, optional
        Specifies the columns to exclude from processing when the decorator is
        applied to a function. If the input data `X` is a pandas DataFrame, 
        `skip_columns` should contain the names of the columns to be skipped. 
        If `X` is a numpy array, `skip_columns`  should be a list of column 
        indices (integers) indicating the positions of the columns to be excluded.
        This allows selective processing of data, avoiding alterations to the
        specified columns.
        
    **kwargs : dict
        Additional fitting parameters.

    Returns
    -------
    Xi : array-like or sparse matrix of shape (n_samples, n_features)
        The imputed dataset.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.mlutils import soft_imputer
    >>> X = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, np.nan, 9]])
    >>> soft_imputer(X, strategy='mean')
    array([[ 1. ,  5. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 2.5,  5. ,  9. ]])
    
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['a', np.nan, 'b']})
    >>> soft_imputer(df, strategy='most_frequent', mode='bi-impute')
               A  B
        0    1.0  a
        1    2.0  a
        2    1.5  b

    Notes
    -----
    The 'bi-impute' mode requires categorical features to be explicitly indicated
    as such by using pandas Categorical dtype or by specifying features to drop.
    """
    X, is_frame  = _convert_to_dataframe(X)
    X = _drop_features(X, drop_features)
    
    if mode == 'bi-impute':
        fill_values, strategies = _enabled_bi_impute_mode (strategy, fill_value)
        try: 
            num_imputer = SimpleImputer(
                strategy=strategies[0], missing_values=missing_values,
                fill_value=fill_values[0])
        except ValueError as e: 
            msg= (" Consider using the {'__', '--', '&', '@', '!'} delimiter"
                  " for mixed numeric and categorical fill values.")
            # Improve the error message 
            raise ValueError(
                f"Imputation failed due to: {e}." +
                msg if check_mixed_data_types(X) else ''
                )
        cat_imputer = SimpleImputer(strategy= strategies[1], 
                                    missing_values=missing_values,
                                    fill_value = fill_values [1]
                                    )
        num_imputed, cat_imputed, num_columns, cat_columns = _separate_and_impute(
            X, num_imputer, cat_imputer)
        Xi = np.hstack((num_imputed, cat_imputed))
        new_columns = num_columns + cat_columns
        Xi = pd.DataFrame(Xi, index=X.index, columns=new_columns)
    else:
        try:
            Xi, imp = _impute_data(
                X, strategy, missing_values, fill_value, add_indicator, copy)
        except Exception as e : 
            raise ValueError("Imputation failed. Consider using the"
                             " 'bi-impute' mode for mixed data types.") from e 
        if isinstance(X, pd.DataFrame):
            Xi = pd.DataFrame(Xi, index=X.index, columns=imp.feature_names_in_)
            
    if not is_frame: # revert back to array
        Xi = np.array ( Xi )
    return Xi

def _convert_to_dataframe(X):
    """Ensure input is a pandas DataFrame."""
    is_frame=True 
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(check_array(X, dtype=None, force_all_finite='allow-nan'), 
                         columns=[f'feature_{i}' for i in range(X.shape[1])])
        is_frame=False 
    return X, is_frame 

def _drop_features(X, drop_features):
    """Drop specified features from the DataFrame."""
    if isinstance(drop_features, bool) and drop_features:
        X = X.select_dtypes(exclude=['object', 'category'])
    elif isinstance(drop_features, list):
        X = X.drop(columns=drop_features, errors='ignore')
    return X

def _impute_data(X, strategy, missing_values, fill_value, add_indicator, copy):
    """Impute the dataset using SimpleImputer."""
    imp = SimpleImputer(strategy=strategy, missing_values=missing_values, 
                        fill_value=fill_value, add_indicator=add_indicator, 
                        copy=copy)
    Xi = imp.fit_transform(X)
    return Xi, imp

def _separate_and_impute(X, num_imputer, cat_imputer):
    """Separate and impute numerical and categorical features."""
    X, num_columns, cat_columns= to_numeric_dtypes(X, return_feature_types=True )

    if len(num_columns) > 0:
        num_imputed = num_imputer.fit_transform(X[num_columns])
    else:
        num_imputed = np.array([]).reshape(X.shape[0], 0)
    
    if len(cat_columns) > 0:
        cat_imputed = cat_imputer.fit_transform(X[cat_columns])
    else:
        cat_imputed = np.array([]).reshape(X.shape[0], 0)
    return num_imputed, cat_imputed, num_columns, cat_columns

def _enabled_bi_impute_mode(
    strategy: str, fill_value: Union[str, float, None]
     ) -> Tuple[List[Union[None, float, str]], List[str]]:
    """
    Determines strategies and fill values for numerical and categorical data
    in bi-impute mode based on the provided strategy and fill value.

    Parameters
    ----------
    strategy : str
        The imputation strategy to use.
    fill_value : Union[str, float, None]
        The fill value to use for imputation, which can be a float, string, 
        or None. When a string containing delimiters is provided, it indicates
        separate fill values for numerical and categorical data.

    Returns
    -------
    Tuple[List[Union[None, float, str]], List[str]]
        A tuple containing two lists: the first with fill values for numerical
        and categorical data, and the second with strategies for numerical and
        categorical data.

    Examples
    --------
    >>> from gofast.tools.mlutils import _enabled_bi_impute_mode
    >>> enabled_bi_impute_mode('mean', None)
    ([None, None], ['mean', 'most_frequent'])

    >>> _enabled_bi_impute_mode('constant', '0__missing')
    ([0.0, 'missing'], ['constant', 'constant'])
    
    >>> _enabled_bi_impute_mode (strategy='constant', fill_value="missing")
    ([0.0, 'missing'], ['constant', 'constant'])
    
    >>> _enabled_bi_impute_mode('constant', 9.) 
    ([9.0, None], ['constant', 'most_frequent'])
    
    >>> _enabled_bi_impute_mode(strategy='constant', fill_value="mean__missing",)
    ([None, 'missing'], ['mean', 'constant'])
    """
    num_strategy, cat_strategy = 'most_frequent', 'most_frequent'
    fill_values = [None, None]
    
    if fill_value is None or isinstance(fill_value, (float, int)):
        if strategy in ["mean", 'median', 'constant']:
            num_strategy = strategy
            fill_values[0] = ( 
                0.0 if strategy == 'constant' 
                and fill_value is None else fill_value
                )
        return fill_values, [num_strategy, cat_strategy]

    if contains_delimiter(fill_value,{"__", "--", "&", "@", "!"} ):
        fill_values, strategies = _manage_fill_value(fill_value, strategy)
    else:
        fill_value = (
            f"{strategy}__{fill_value}" if strategy in ['mean', 'median'] 
            else ( f"0__{fill_value}" if strategy =="constant" else fill_value )
        )
        fill_values, strategies = _manage_fill_value(fill_value, strategy)
    
    return fill_values, strategies

def _manage_fill_value(
    fill_value: str, strategy: str
    ) -> Tuple[List[Union[None, float, str]], List[str]]:
    """
    Parses and manages fill values for bi-impute mode, supporting mixed types.

    Parameters
    ----------
    fill_value : str
        The fill value string potentially containing mixed types for numerical
        and categorical data.
    strategy : str
        The imputation strategy to determine how to handle numerical fill values.

    Returns
    -------
    Tuple[List[Union[None, float, str]], List[str]]
        A tuple containing two elements: the first is a list with numerical and
        categorical fill values, and the second is a list of strategies for 
        numerical and categorical data.

    Raises
    ------
    ValueError
        If the fill value does not contain a proper separator or if the numerical
        fill value is incompatible with the specified strategy.

    Examples
    --------
    >>> from gofast.tools.mlutils import _manage_fill_value
    >>> _manage_fill_value("0__missing", "constant")
    ([0.0, 'missing'], ['constant', 'constant'])

    >>> _manage_fill_value("mean__missing", "mean")
    ([None, 'missing'], ['mean', 'constant'])
    """
    regex = re.compile(r'(__|--|&|@|!)')
    parts = regex.split(fill_value)
    if len(parts) < 3:
        raise ValueError("Fill value must contain a separator (__|--|&|@|!)"
                         " between numerical and categorical fill values.")

    num_fill, cat_fill = parts[0], parts[-1]
    num_fill_value = None if strategy in ['mean', 'median'] and num_fill.lower(
        ) in ['mean', 'median'] else num_fill

    try:
        num_fill_value = float(num_fill) if num_fill.replace('.', '', 1).isdigit() else num_fill
    except ValueError:
        raise ValueError(f"Numerical fill value '{num_fill}' must be a float"
                         f" for strategy '{strategy}'.")
    strategies =[ strategy if strategy in ['mean', 'median', 'constant']
                 else 'most_frequent', 'constant'] 
    if num_fill.lower() in ['mean', 'median'] and strategy=='constant': 
        # Permutate the strategy and fill value. 
        strategies [0]= num_fill.lower()
        num_fill_value =None ; 
    
    return [num_fill_value, cat_fill], strategies

@SmartProcessor(fail_silently= True, param_name="skip_columns")
def soft_scaler(
    X, *, 
    kind=StandardScaler, 
    copy=True, 
    with_mean=True, 
    with_std=True, 
    feature_range=(0, 1), 
    clip=False, 
    norm='l2',
    skip_columns=None, 
    verbose=0, 
    **kwargs
    ):
    """
    Scale data using specified scaling strategy from scikit-learn. 
    
    Function excludes categorical features from scaling and provides 
    feedback via verbose parameter.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        The data to scale, can contain both numerical and categorical features.
    kind : str, default='StandardScaler'
        The kind of scaling to apply to numerical features. One of 'StandardScaler', 
        'MinMaxScaler', 'Normalizer', or 'RobustScaler'.
    copy : bool, default=True
        If False, avoid a copy and perform inplace scaling instead.
    with_mean : bool, default=True
        If True, center the data before scaling. Only applicable when kind is
        'StandardScaler' or 'RobustScaler'.
    with_std : bool, default=True
        If True, scale the data to unit variance. Only applicable when kind is
        'StandardScaler' or 'RobustScaler'.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data. Only applicable when kind 
        is 'MinMaxScaler'.
    clip : bool, default=False
        Set to True to clip transformed values to the provided feature range.
        Only applicable when kind is 'MinMaxScaler'.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non-zero sample or feature.
        Only applicable when kind is 'Normalizer'.
    skip_columns : list of str or int, optional
        Specifies the columns to exclude from processing when the decorator is
        applied to a function. If the input data `X` is a pandas DataFrame, 
        `skip_columns` should contain the names of the columns to be skipped. 
        If `X` is a numpy array, `skip_columns`  should be a list of column 
        indices (integers) indicating the positions of the columns to be excluded.
        This allows selective processing of data, avoiding alterations to the
        specified columns.
    verbose : int, default=0
        If > 0, warning messages about the processing.
        
    **kwargs : additional keyword arguments
        Additional fitting parameters to pass to the scaler.
        
    Returns
    -------
    X_scaled : {ndarray, sparse matrix, dataframe} of shape (n_samples, n_features)
        The scaled data. The scaled data with numerical features scaled according
        to the specified kind, and categorical features returned unchanged. 
        The return type matches the input type.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.tools.mlutils import soft_scaler
    >>> X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    >>> X_scaled = soft_scaler(X, kind='StandardScaler')
    >>> print(X_scaled)
    [[ 0.  -1.22474487  1.33630621]
     [ 1.22474487  0.  -0.26726124]
     [-1.22474487  1.22474487 -1.06904497]]

    >>> df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    >>> df_scaled = soft_scaler(df, kind='RobustScaler', with_centering=True,
                                with_scaling=True)
    >>> print(df_scaled)
              a    b    c
        0 -0.5 -1.0  1.0
        1  0.5  0.0  0.0
        2 -0.5  1.0 -1.0
    """
    X= to_numeric_dtypes(X)
    input_is_dataframe = isinstance(X, pd.DataFrame)
    cat_features = X.select_dtypes(
        exclude=['number']).columns if input_is_dataframe else []

    if verbose > 0 and len(cat_features) > 0:
        warnings.warn(
            "Note: Categorical data detected and excluded from scaling.")

    kind= kind if isinstance(kind, str) else kind.__name__
    scaler = _determine_scaler(
        kind, copy=copy, with_mean=with_mean, with_std=with_std, norm=norm, 
        feature_range=feature_range, clip=clip, **kwargs)

    if input_is_dataframe:
        num_features = X.select_dtypes(['number']).columns
        X_scaled_numeric = _scale_numeric_features(X, scaler, num_features)
        X_scaled = _concat_scaled_numeric_with_categorical(
            X_scaled_numeric, X, cat_features)
    else:
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def _determine_scaler(kind, **kwargs):
    """
    Determines the scaler based on the kind parameter.
    """
    scaler_classes = {
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler,
        'Normalizer': Normalizer,
        'RobustScaler': RobustScaler
    }
    scaler_class = scaler_classes.get(kind, None)
    if scaler_class is None:
        raise ValueError(f"Unsupported scaler kind: {kind}. Supported scalers"
                         f" are: {', '.join(scaler_classes.keys())}.")
    kwargs= get_valid_kwargs(scaler_class, **kwargs)
    return scaler_class(**kwargs)

def _scale_numeric_features(X, scaler, num_features):
    """
    Scales numerical features of the DataFrame X using the provided scaler.
    """
    return scaler.fit_transform(X[num_features])

def _concat_scaled_numeric_with_categorical(X_scaled_numeric, X, cat_features):
    """
    Concatenates scaled numerical features with original categorical features.
    """
    X_scaled = pd.concat([pd.DataFrame(X_scaled_numeric, index=X.index, 
                        columns=X.select_dtypes(['number']).columns),
                          X[cat_features]], axis=1)
    return X_scaled[X.columns]  # Maintain original column order

@ensure_pkg(
    "shap", extra="SHapley Additive exPlanations (SHAP) is needed.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("pkg")=="shap"
    )
def display_feature_contributions(
        X, y=None, view=False, pkg=None):
    """
    Trains a RandomForest model to determine the importance of features in
    the dataset and optionally displaysthese importances visually using SHAP.

    Parameters
    ----------
    X : ndarray or DataFrame
        The feature matrix from which to determine feature importances. This 
        should not include the target variable.
    y : ndarray, optional
        The target variable array. If provided, it will be used for supervised 
        learning. If None, an unsupervised approach will be used 
        (feature importances based on feature permutation).
    view : bool, optional
        If True, display feature importances using SHAP's summary plot.
        Defaults to False.

    Returns
    -------
    dict
        A dictionary where keys are feature names and values are their 
        corresponding importances as determined by the RandomForest model.

    Examples
    --------
    >>> from sklearn.datasets import load_iris 
    >>> from gofast.tools.mlutils import display_feature_contributions
    >>> data = load_iris()
    >>> X = data['data']
    >>> feature_names = data['feature_names']
    >>> display_feature_contributions(X, view=True)
    {'sepal length (cm)': 0.112, 'sepal width (cm)': 0.032, 
     'petal length (cm)': 0.423, 'petal width (cm)': 0.433}
    """
    pkg ='shap' if str(pkg).lower() =='shap' else 'matplotlib'
    
    validate_data_types(X, nan_policy="raise", error ="raise")
    
    # Initialize the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the provided features, with or without a target variable
    if y is not None:
        model.fit(X, y)
    else:
        # Fit a dummy target if y is None, assuming unsupervised setup
        model.fit(X, range(X.shape[0]))

    # Extract feature importances
    importances = model.feature_importances_
    
    # Optionally, display the feature importances using the chosen visualization package
    feature_names = model.feature_names_in_ if hasattr(
            model, 'feature_names_in_') else [f'feature_{i}' for i in range(X.shape[1])]
    if view:
        if pkg.lower() == "shap":
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, X, feature_names=feature_names)
            
        elif pkg.lower() == "matplotlib":
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            indices = range(len(importances))
            plt.title('Feature Importances')
            plt.bar(indices, importances, color='skyblue', align='center')
            plt.xticks(indices, feature_names, rotation=45)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()

    # Map feature names to their importances
    feature_importance_dict = dict(zip(feature_names, importances))
    summary = ReportFactory(title="Feature Contributions Table",).add_mixed_types(
        feature_importance_dict)
    
    print(summary)

@ensure_pkg ("shap", extra = ( 
    "`get_feature_contributions` needs SHapley Additive exPlanations (SHAP)"
    " package to be installed. Instead, you can use"
    " `gofast.tools.display_feature_contributions` for contribution scores" 
    " and `gofast.analysis.get_feature_importances` for PCA quick evaluation."
    )
 )
def get_feature_contributions(X, model=None, view=False):
    """
    Calculate the SHAP (SHapley Additive exPlanations) values to determine 
    the contribution of each feature to the model's predictions for each 
    instance in the dataset and optionally display a visual summary.

    Parameters
    ----------
    X : ndarray or DataFrame
        The feature matrix for which to calculate feature contributions.
    model : sklearn.base.BaseEstimator, optional
        A pre-trained tree-based machine learning model from scikit-learn (e.g.,
        RandomForest). If None, a new RandomForestClassifier will be trained on `X`.
    view : bool, optional
        If True, displays a visual summary of feature contributions using SHAP's
        visualization tools. Default is False.

    Returns
    -------
    ndarray
        A matrix of SHAP values where each row corresponds to an instance and
        each column corresponds to a feature's contribution to that instance's 
        prediction.

    Notes
    -----
    The function defaults to creating and using a RandomForestClassifier if no 
    model is provided. It is more efficient to pass a pre-trained model if 
    available.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.tools.mlutils import get_feature_contributions
    >>> data = load_iris()
    >>> X = data['data']
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, data['target'])
    >>> contributions = get_feature_contributions(X, model, view=True)
    """
    import shap
    
    # If no model is provided, train a RandomForestClassifier
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Dummy target, assuming unsupervised setup for example
        model.fit(X, np.zeros(X.shape[0]))  

    # Create the Tree explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP returns a list for multi-class, we sum across all classes for overall importance
    if isinstance(shap_values, list):
        shap_values = np.sum(np.abs(shap_values), axis=0)

    # Visualization if view is True
    if view:
        shap.summary_plot(shap_values, X, feature_names=model.feature_names_in_ if hasattr(
            model, 'feature_names_in_') else None)

    return shap_values

          
def smart_label_classifier(
    y: ArrayLike, *,
    values: Union[float, List[float], None] = None,
    labels: Union[int, str, List[str]] = None,
    order: str = 'soft',
    func: Optional[Callable[[float], Union[int, str]]] = None,
    raise_warn: bool = True
) -> np.ndarray:
    """
    Maps a numeric array into class labels based on specified thresholds or
    a custom mapping function. The `smart_label_classifier` function 
    categorizes an array of continuous values into distinct classes, either
    by using predefined threshold values (`values`) or by applying a custom
    function (`func`). Optional `labels` can be used to name the categories.

    .. math::
        Y_i = 
        \begin{cases} 
            L_1, & \text{if } y_i \leq v_1 \\
            L_2, & \text{if } v_1 < y_i \leq v_2 \\
            \vdots \\
            L_{n+1}, & \text{if } y_i > v_n \\
        \end{cases}

    where :math:`y_i` represents the value of the `i`-th item in `y`, 
    and :math:`L` denotes the class labels corresponding to thresholds 
    :math:`v`.

    Parameters
    ----------
    y : ArrayLike
        One-dimensional array of numeric values to be categorized.

    values : float, list of float, optional
        Threshold values for categorization. If `values` is provided,
        items in `y` are mapped based on these thresholds. For instance,
        if `values = [1.0, 2.5]`, three classes will be generated: one
        for items less than or equal to 1.0, one for items between 1.0
        and 2.5, and one for items greater than 2.5.

    labels : int, str, or list of str, optional
        Labels for the resulting categories. If an integer is provided, 
        it specifies the number of classes to generate in `y` 
        automatically when `func` and `values` are `None`. For example, 
        if `labels=3`, the function divides `y` into three classes. If 
        `labels` is a list, each element should correspond to a class 
        created by `values` + 1. Mismatches raise an error in strict mode.

    order : {'soft', 'strict'}, default='soft'
        Mode to control the handling of `values`. If `order='strict'`,
        items in `y` must match `values` exactly; otherwise, approximate
        values are substituted. A warning is issued in soft mode if a 
        mismatch occurs.

    func : Callable, optional
        Custom function to categorize values in `y`. If `func` is provided,
        it takes precedence, and `values` are ignored. `func` should accept
        a single numeric input and return a category.

    raise_warn : bool, default=True
        If `True`, raises a warning when `order='soft'` and `values` 
        cannot be matched exactly or if `labels` do not match the 
        number of classes derived from `values`.

    Returns
    -------
    np.ndarray
        Array of the same length as `y`, with categorized values or
        labels if provided.

    Notes
    -----
    - This function requires either `values` or `func` to categorize `y`.
      If neither is provided, `labels` must be an integer to specify the
      number of classes.
    - `labels` should match the number of classes created by `values` + 1.
      If they do not, a `ValueError` is raised if `order` is `'strict'`.

    Examples
    --------
    >>> from gofast.tools.mlutils import smart_label_classifier
    >>> import numpy as np
    >>> y = np.arange(0, 7, 0.5)
    
    Basic classification with values:
    >>> smart_label_classifier(y, values=[1.0, 3.2])
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    Assign custom labels:
    >>> smart_label_classifier(y, values=[1.0, 3.2], labels=['low', 'mid', 'high'])
    array(['low', 'low', 'low', 'mid', 'mid', 'mid', 'mid', 'high', 'high', 
           'high', 'high', 'high', 'high', 'high'], dtype=object)

    Using a custom function:
    >>> def custom_func(v):
    ...     if v <= 1: return 'low'
    ...     elif 1 < v <= 3.2: return 'mid'
    ...     else: return 'high'
    >>> smart_label_classifier(y, func=custom_func)
    array(['low', 'low', 'low', 'mid', 'mid', 'mid', 'mid', 'high', 'high', 
           'high', 'high', 'high', 'high', 'high'], dtype=object)

    Auto-generate classes:
    >>> smart_label_classifier(y, labels=3)
    array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])

    See Also
    --------
    _validate_func_values_labels : Helper function for validating `func` 
                                   and `values` parameters.
    _assert_labels_from_values : Helper function to validate `labels` 
                                 against `values`.
    _smart_mapper : Helper function for mapping continuous values to 
                    categorical classes based on thresholds.

    References
    ----------
    .. [1] Johnson, T., & Brown, A. (2021). *Categorical Data Mapping*.
       Data Science Journal, 17(4), 123-145.
    .. [2] Lee, K., & Singh, P. (2019). *Threshold-Based Classification
       Techniques*. Journal of Machine Learning, 9(2), 67-80.
    """

    name = None
    if isinstance(y, pd.Series) and hasattr(y, "name"):
        name = y.name

    arr = np.asarray(y).squeeze()

    if not _is_arraylike_1d(arr):
        raise TypeError(
            "Expected a one-dimensional array,"
            f" got array with shape {arr.shape}"
        )

    if isinstance(values, str):
        values = str2columns(values)

    if values is not None:
        values = is_iterable(values, parse_string=True, transform=True)
        approx_values: List[Tuple[float, float]] = []
        processed_values = np.zeros(len(values), dtype=float)

        for i, v in enumerate(values):
            try:
                v = float(v)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Value '{v}' must be a valid number.") from e

            non_nan_arr = arr[~np.isnan(arr)]
            diff = np.abs(non_nan_arr - v)
            min_idx = np.argmin(diff)

            if order == 'strict' and diff[min_idx] != 0.0:
                raise ValueError(
                    f"Value {v} is missing in the array. It must be present "
                    "when order is set to 'strict', or set order to 'soft'"
                    " to allow approximate matching."
                )

            matched_value = non_nan_arr[min_idx]
            processed_values[i] = matched_value

            if diff[min_idx] != 0.0:
                approx_values.append((v, matched_value))

        if approx_values and raise_warn:
            original_vals, substituted_vals = zip(*approx_values)
            verb = "are" if len(original_vals) > 1 else "is"
            warnings.warn(
                f"Values {original_vals} {verb} missing in the array. "
                f"Substituted with {substituted_vals}."
            )

    arr_copied = arr.copy()

    if func is None and values is None:
        return _validate_func_values_labels(
            func=func, 
            values= values, 
            labels= labels, 
            y=y, 
            order =order, 
        )

    mapper_func: Optional[Callable[[float], Union[int, str]]] = func
    if values is not None and func is None:
        mapper_func = lambda k: _smart_mapper(k, kr=processed_values)

    arr_mapped = pd.Series(arr_copied, name='temp').map(mapper_func).values

    label_mapping: Dict[Union[int, float], Union[int, str]] = {}
    if labels is not None:
        labels = is_iterable(labels, parse_string=True, transform=True)
        labels, label_mapping = _assert_labels_from_values(
            arr_mapped,
            processed_values,
            labels,
            label_mapping,
            raise_warn=raise_warn,
            order=order
        )

    if labels is not None:
        arr_mapped = pd.Series(
            arr_mapped, name=name or 'temp'
        ).map(label_mapping).values
    else:
        arr_mapped = arr_mapped if name is None else pd.Series(
            arr_mapped, name=name
        )

    return arr_mapped

def _validate_func_values_labels(
    func: Optional[Callable],
    values: Optional[Union[float, List[float]]],
    labels: Optional[Union[int, str, List[str]]],
    y: np.ndarray,
    order: str
) -> np.ndarray:
    """
    Validates that either `func` or `values` is provided, and handles cases 
    where `labels` is provided as an integer when `func` and `values` are None.
    """

    # Raise an error if labels are not provided
    # when both func and values are None
    if labels is None:
        raise TypeError(
            "'func' cannot be None when 'values' are not provided."
        )
    
    # Handle the case where labels is an integer
    if isinstance(labels, int):
        if order == 'strict':
            raise TypeError(
                "'func' cannot be None when 'values' are not provided. "
                "To heuristically create `labels` classes, set `order='soft'`."
            )
        
        # Ensure `y` is a 1-dimensional array
        y = np.squeeze(y)
        if y.ndim != 1:
            raise ValueError(
                "Input array `y` must be one-dimensional for"
                " automatic class generation."
            )
        
        try:
            # Automatically create `labels` number of classes in `y`
            y_min, y_max = np.min(y), np.max(y)
            thresholds = np.linspace(y_min, y_max, labels + 1)[1:-1]
            categorized_y = np.digitize(y, bins=thresholds)
            return categorized_y
        except Exception as e:
            raise ValueError(
                "An error occurred while attempting to categorize `y`. "
                "Ensure `y` is numeric and contains valid values for"
                " thresholding."
            ) from e
    
    else:
        raise TypeError(
            "When `func` and `values` are None, `labels` should be "
            "an integer specifying the number of classes to generate."
        )
    
    return y

def _assert_labels_from_values(
    arr: np.ndarray,
    values: np.ndarray,
    labels: Union[int, str, List[str]],
    label_mapping: Dict,
    raise_warn: bool = True,
    order: str = 'soft'
) -> Tuple[List[Union[int, str]], Dict[Union[int, float], Union[int, str]]]:
    unique_labels = list(np.unique(arr))
    if not is_iterable(labels):
        labels = [labels]

    if not _check_consistency_size(unique_labels, labels, error='ignore'):
        if order == 'strict':
            verb = "were" if len(labels) > 1 else "was"
            raise TypeError(
                f"Expected {len(unique_labels)} labels for the {len(values)}"
                f" values renaming. {len(labels)} {verb} given."
            )

        expected_labels_count = len(values) + 1
        actual_labels_count = len(labels)
        if actual_labels_count != expected_labels_count:
            verb = "s are" if len(values) > 1 else " is"
            msg = (
                f"{len(values)} value{verb} passed. Labels for renaming "
                f"values expect to be composed of {expected_labels_count}"
                f" items ('number of values + 1') for pure categorization."
            )
            undefined_classes = unique_labels[len(labels):]
            labels = list(labels) + list(undefined_classes)
            labels = labels[:len(unique_labels)]
            msg += ( 
                f" Classes {smart_format(undefined_classes)}"
                " cannot be renamed."
                )

            if raise_warn:
                warnings.warn(msg)

    label_mapping = dict(zip(unique_labels, labels))
    return labels, label_mapping

def _smart_mapper(
    k: float,
    kr: np.ndarray,
    return_dict_map: bool = False
) -> Union[int, Dict[int, bool], float]:
    if len(kr) == 1:
        conditions = {
            0: k <= kr[0],
            1: k > kr[0]
        }
    elif len(kr) == 2:
        conditions = {
            0: k <= kr[0],
            1: kr[0] < k <= kr[1],
            2: k > kr[1]
        }
    else:
        conditions = {}
        for idx in range(len(kr) + 1):
            if idx == 0:
                conditions[idx] = k <= kr[idx]
            elif idx == len(kr):
                conditions[idx] = k > kr[-1]
            else:
                conditions[idx] = kr[idx - 1] < k <= kr[idx]

    if return_dict_map:
        return conditions

    for class_label, condition in conditions.items():
        if condition:
            return class_label if not math.isnan(k) else np.nan

    return np.nan
        
        
        
        

