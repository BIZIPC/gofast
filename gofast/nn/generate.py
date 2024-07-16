# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Module for data generation and NN processing functions.

Includes functions for creating sequences and generating data suitable for 
training neural networks.
"""
import numpy as np

from ..api.types import  Optional,  Tuple
from ..api.types import ArrayLike , Callable, Generator

from ..tools.coreutils import is_iterable
from ..tools.validator import check_X_y
from ..tools.validator import check_array

__all__ = ['create_sequences', 'data_generator']

def create_sequences(
    input_data: ArrayLike, 
    n_features: int, 
    n_forecast: int = 1, 
    n_lag: int = 12, 
    step: int = 1, 
    output_features: list[int] = None, 
    shuffle: bool = False, 
    normalize: bool = False
 ) -> tuple[ArrayLike, ArrayLike]:
    """
    Create sequences from the data for LSTM model training. 
    
    Generates sequences, including specific output features selection, 
    optional shuffling, and normalization.

    Parameters
    ----------
    input_data : np.ndarray
        Normalized dataset as a NumPy array.
        This array contains the input data for sequence generation.

    n_features : int
        Total number of features in the dataset.
        This is the number of columns in `input_data`.

    n_forecast : int, optional
        Number of steps to forecast; defaults to 1 (for the next time step).
        This parameter defines how many future steps are predicted.

    n_lag : int, optional
        Number of past time steps to use for prediction; defaults to 12.
        This parameter defines the length of the input sequences.

    step : int, optional
        Step size for iterating through the `input_data`; defaults to 1.
        This defines the stride of the sliding window over the `input_data`.

    output_features : list[int], optional
        Indices of features to be used for output sequences; defaults to None,
        using the last feature.
        This specifies which columns of `input_data` are the target variables.

    shuffle : bool, optional
        Whether to shuffle the sequences; defaults to False. Useful for training.
        Shuffling can help in making the training process more robust.

    normalize : bool, optional
        Whether to apply normalization on sequences; defaults to False.
        Normalization can improve model performance by standardizing the input data.

    Returns
    -------
    X : np.ndarray
        Input sequences shaped for LSTM, with dimensions [samples, time steps, features].
        These are the sequences used as input for the LSTM model.

    y : np.ndarray
        Output sequences (targets), shaped according to selected output features.
        These are the sequences used as targets for the LSTM model.

    Notes
    -----
 
    The input sequences :math:`X` are generated using a sliding window approach. 
    For each time step :math:`t`, the sequence is constructed as:

    .. math::

        X_t = [x_{t-n_{\text{lag}}}, x_{t-n_{\text{lag}}+1}, \ldots, x_{t-1}]

    The corresponding output sequences :math:`y` are generated as:

    .. math::

        y_t = [x_{t}, x_{t+1}, \ldots, x_{t+n_{\text{forecast}}-1}]

    If normalization is applied, each sequence is normalized to have zero mean 
    and unit variance.

    This function is useful for preparing time series data for LSTM models. By 
    generating input-output pairs with specified lag and forecast steps, it helps 
    in creating datasets suitable for sequence prediction tasks.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.models.deep_search import create_sequences
    >>> input_data = np.random.rand(100, 5)  # Example data
    >>> X, y = create_sequences(input_data, n_features=5, n_forecast=1, n_lag=12,
    ...                         step=1, output_features=[-1], shuffle=True, normalize=True)
    >>> print(X.shape, y.shape)

    See Also
    --------
    tf.keras.preprocessing.sequence.TimeseriesGenerator :
        Utility for creating time series batches.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    """

    input_data = check_array( input_data )
    output_features= is_iterable(output_features, transform =True )
    if output_features is None:
        output_features = [-1]  # Default to the last feature if none specified
    
    X, y = [], []
    for i in range(0, len(input_data) - n_lag, step):
        end_ix = i + n_lag
        if end_ix + n_forecast > len(input_data):
            break
        seq_x = input_data[i:end_ix, :]
        seq_y = input_data[end_ix:end_ix + n_forecast, output_features].squeeze()
        
        if normalize:
            seq_x = (seq_x - np.mean(seq_x, axis=0)) / np.std(seq_x, axis=0)
            seq_y = (seq_y - np.mean(seq_y)) / np.std(seq_y)
        
        X.append(seq_x)
        y.append(seq_y)
    
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = np.array(X)[indices]
        y = np.array(y)[indices]
    
    return np.array(X), np.array(y)

def data_generator(
    X: ArrayLike, 
    y_actual: ArrayLike, 
    y_estimated: Optional[ArrayLike] = None, 
    batch_size: int = 32,
    shuffle: bool = True,
    preprocess_fn: Optional[Callable[
        [ArrayLike, ArrayLike, Optional[ArrayLike]], Tuple[
            ArrayLike, ArrayLike, Optional[ArrayLike]]]] = None
) -> Generator[Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]], None, None]:
    r"""
    Generates batches of data for training or validation. Optionally shuffles 
    the data each epoch and applies a custom preprocessing function to the 
    batches.

    Parameters
    ----------
    X : ArrayLike
        The input features, expected to be a NumPy array or similar.
        These features are the independent variables used for model training.

    y_actual : ArrayLike
        The actual target values, expected to be a NumPy array or similar.
        These are the true values that the model aims to predict.

    y_estimated : Optional[ArrayLike], default=None
        The estimated target values, if applicable. None if not used.
        These values are used in conjunction with a custom loss function 
        for additional domain-specific information.

    batch_size : int, default=32
        The number of samples per batch.
        This defines how many samples are processed in each iteration.

    shuffle : bool, default=True
        Whether to shuffle the indices of the data before creating batches.
        Shuffling is useful to ensure that the model does not learn the 
        order of the data.

    preprocess_fn : Optional[Callable], default=None
        A function that takes a batch of `X`, `y_actual`, and `y_estimated`,
        and returns preprocessed batches. If None, no preprocessing is applied.
        This allows for custom preprocessing steps such as normalization or 
        augmentation.

    Yields
    ------
    Generator[Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]], None, None]
        A generator yielding tuples containing a batch of input features, 
        actual target values, and optionally estimated target values.
        Each batch is a tuple (`x_batch`, `y_actual_batch`, `y_estimated_batch`).

    Notes
    -----
    The `data_generator` function is designed to streamline the process of 
    feeding data into a neural network model for training or validation. 
    By optionally shuffling the data and allowing for custom preprocessing, 
    it provides flexibility and ensures that the data fed into the model is 
    both randomized and tailored to specific requirements.
    
    The generator yields batches of data defined as:

    .. math::

        \text{Batch}_i = (X[i \cdot \text{batch_size} : (i+1) \cdot \text{batch_size}],
        y_{\text{actual}}[i \cdot \text{batch_size} : (i+1) \cdot \text{batch_size}],
        y_{\text{estimated}}[i \cdot \text{batch_size} : (i+1) \cdot \text{batch_size}])

    if `preprocess_fn` is provided, each batch is processed through:

    .. math::

        X_{\text{batch}}, y_{\text{actual}_{\text{batch}}},\\
            y_{\text{estimated}_{\text{batch}}} = 
        \text{preprocess_fn}(X_{\text{batch}}, y_{\text{actual}_{\text{batch}}},\\
                             y_{\text{estimated}_{\text{batch}}})

    Examples
    --------
    >>> from gofast.models.deep_search import data_generator
    >>> import numpy as np
    >>> X = np.random.rand(100, 10)  # 100 samples, 10 features
    >>> y_actual = np.random.rand(100, 1)
    >>> y_estimated = np.random.rand(100, 1)
    >>> generator = data_generator(X, y_actual, y_estimated, batch_size=10, shuffle=True)
    >>> for x_batch, y_actual_batch, y_estimated_batch in generator:
    ...     # Process the batch
    ...     pass

    See Also
    --------
    tf.data.Dataset : A TensorFlow class for building efficient data pipelines.
    custom_loss : Defines the custom loss function that may use `y_estimated`.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """

    X, y_actual = check_X_y(X, y_actual)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        x_batch = X[batch_indices]
        y_actual_batch = y_actual[batch_indices]
        y_estimated_batch = y_estimated[batch_indices] if\
            y_estimated is not None else None

        if preprocess_fn is not None:
            x_batch, y_actual_batch, y_estimated_batch = preprocess_fn(
                x_batch, y_actual_batch, y_estimated_batch)

        yield (x_batch, y_actual_batch, y_estimated_batch)