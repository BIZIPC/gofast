# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Module for training and evaluating neural network models.

Includes functions for calculating validation loss, cross-validating LSTM models, 
evaluating models, making future predictions, plotting errors, training and 
evaluating models, and more.
"""
import os
import json
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..api.types import _Tensor,  _Optimizer
from ..api.types import _History, _Callback, _Model
from ..api.types import List, Optional, Union, Dict, Tuple
from ..api.types import ArrayLike, Callable, Any, Generator
from ..tools.coreutils import denormalize
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import check_X_y, check_consistent_length
from ..tools.validator import validate_keras_model

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    Adam = KERAS_DEPS.Adam
    RMSprop = KERAS_DEPS.RMSprop
    SGD = KERAS_DEPS.SGD
    load_model = KERAS_DEPS.load_model
    mnist = KERAS_DEPS.mnist
    Loss = KERAS_DEPS.Loss
    Sequential = KERAS_DEPS.Sequential
    Dense = KERAS_DEPS.Dense
    reduce_mean = KERAS_DEPS.reduce_mean
    GradientTape = KERAS_DEPS.GradientTape
    square = KERAS_DEPS.square
    
__all__=[
     'calculate_validation_loss',
     'cross_validate_lstm',
     'evaluate_model',
     'make_future_predictions',
     'plot_errors',
     'plot_history',
     'plot_predictions',
     'train_and_evaluate',
     'train_and_evaluate2',
     'train_epoch',
     'train_model'
 ]

DEP_MSG=dependency_message('train')

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def plot_history(
    history: _History, 
    title: str = 'Model Learning Curve',
    color_scheme: Optional[Dict[str, str]] = None,
    xlabel: str = 'Epoch',
    ylabel_acc: str = 'Accuracy',
    ylabel_loss: str = 'Loss',
    figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plots the learning curves of the training process for both accuracy and loss,
    providing visual insights into the model's performance over epochs. This function
    generates two plots: one for accuracy and another for loss, each against the number
    of epochs. The function returns matplotlib axes to allow further customization.

    Parameters
    ----------
    history : `History`
        The `History` object captures the training history of a model, typically
        returned by the ``fit`` method of models in machine learning frameworks
        like Keras. It contains metrics collected at each epoch during training,
        such as training and validation accuracy and loss.
    title : str, optional
        Title for the plots. Defaults to 'Model Learning Curve'. It is used to label
        the generated plots and can be customized to reflect specific details of
        the training session or model configuration.
    color_scheme : dict of str, optional
        Custom color settings for the plot lines. Keys should include
        'train_acc', 'val_acc', 'train_loss', and 'val_loss'. If not provided,
        default colors are used for each plot line.
    xlabel : str, optional
        Label for the x-axis, which represents the number of epochs. Defaults
        to 'Epoch', but can be customized if the training epochs are described
        with a different terminology or unit.
    ylabel_acc : str, optional
        Label for the y-axis on the accuracy plot. Defaults to 'Accuracy'.
        This label can be tailored if a different metric for model performance
        is utilized.
    ylabel_loss : str, optional
        Label for the y-axis on the loss plot. Defaults to 'Loss'. As with
        the accuracy label, this can be adjusted to suit alternate loss
        functions or descriptions.
    figsize : tuple of int, optional
        The dimensions of the figure (width, height) in inches. Defaults to
        (12, 6). Adjusting this parameter affects the size and appearance
        of the output plots.

    Returns
    -------
    ax1, ax2 : tuple of `matplotlib.axes._subplots.AxesSubplot`
        The matplotlib axes for the accuracy and loss plots. These can be
        used to further customize the plots after the function call.

    Examples
    --------
    >>> from gofast.models.deep_search import plot_history
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    >>> ax_acc, ax_loss = plot_history(history)
    >>> ax_acc.set_title('Updated Accuracy Title')
    >>> plt.show()

    Notes
    -----
    The visualization of model training history helps in diagnosing issues
    with model learning, such as overfitting or underfitting, and in making
    decisions about further model training adjustments or hyperparameter tuning.

    See Also
    --------
    Model.fit : The function used to train the model that produces the `History`
                object required by this plotting function.

    References
    ----------
    .. [1] Chollet, François. "Deep Learning with Python." Manning Publications Co.,
           2017. Insights into model evaluation and history plotting.
    """
    if color_scheme is None:
        color_scheme = {
            'train_acc': 'blue', 'val_acc': 'green',
            'train_loss': 'orange', 'val_loss': 'red'
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', 
             color=color_scheme.get('train_acc', 'blue'))
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy',
                 color=color_scheme.get('val_acc', 'green'))
    ax1.set_title(title + ' - Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_acc)
    ax1.legend()

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss',
             color=color_scheme.get('train_loss', 'orange'))
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss',
                 color=color_scheme.get('val_loss', 'red'))
    ax2.set_title(title + ' - Loss')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_loss)
    ax2.legend()
    return ax1, ax2

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def train_and_evaluate2(
    model_config: Dict[str, Any], 
    resource: int, 
    dataset: Optional[tuple] = None, 
    batch_size: int = 32, 
    optimizer: str = 'Adam', 
    callbacks: List[_Callback] = None
    ) -> float:
    """
    Trains and evaluates a TensorFlow/Keras model based on the provided 
    configuration and training parameters. This function manages resource 
    allocation, dataset handling, and optimization processes to train the model
    and measure its performance on a validation set.

    Parameters
    ----------
    model_config : Dict[str, Any]
        A dictionary specifying the model's configuration, including
        hyperparameters like layer sizes or activation functions. Keys are
        parameter names (e.g., 'units', 'activation'), and values are the
        settings for those parameters.
    resource : int
        The allocated resource for training the model, typically measured in
        the number of epochs to train.
    dataset : tuple, optional
        Specifies the dataset to train and validate the model. Should be a
        tuple of tuples ((x_train, y_train), (x_val, y_val)). If not provided,
        the MNIST dataset is automatically loaded and used.
    batch_size : int, optional
        The number of samples per batch during training. Affects the gradient
        estimation and update steps. Default is 32.
    optimizer : str, optional
        The name of the optimizer to use for adjusting the weights of the model.
        Common options include 'Adam', 'sgd', 'rmsprop'. Default is 'Adam'.
    callbacks : List[tf.keras.callbacks.Callback], optional
        A list of callback functions to apply during training, such as early
        stopping or learning rate schedulers.

    Returns
    -------
    float
        The best validation accuracy achieved during the training process.

    Examples
    --------
    >>> from gofast.models.deep_search import train_and_evaluate2
    >>> model_config = {'units': 64, 'activation': 'relu', 'learning_rate': 0.001}
    >>> resource = 10  # Number of epochs
    >>> best_val_accuracy = train_and_evaluate2(model_config, resource)
    >>> print(f"Best Validation Accuracy: {best_val_accuracy:.2f}")

    Notes
    -----
    This function abstracts away the complexities of model training and
    evaluation, providing a high-level interface to configure, train, and
    evaluate neural network models. The use of callbacks and different optimizers
    can significantly influence training dynamics and outcomes.

    See Also
    --------
    tf.keras.Model : The base class in TensorFlow for constructing neural networks.
    tf.keras.optimizers.Adam : The Adam optimizer, commonly used for training models.

    References
    ----------
    .. [1] Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic
           optimization." arXiv preprint arXiv:1412.6980 (2014).
    .. [2] LeCun, Yann, et al. "Gradient-based learning applied to document
           recognition." Proceedings of the IEEE, 1998.
    """
    # Load and prepare dataset
    if dataset is None:
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype('float32') / 255
        x_val = x_val.reshape(-1, 784).astype('float32') / 255
    else:
        (x_train, y_train), (x_val, y_val) = dataset

    # Define model architecture
    model = Sequential([
        Dense(model_config['units'], activation='relu', 
                              input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    # Select optimizer
    if optimizer.lower() == 'adam':
        opt = Adam(learning_rate=model_config['learning_rate'])
    elif optimizer.lower() == 'sgd':
        opt = SGD(learning_rate=model_config['learning_rate'])
    elif optimizer.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=model_config['learning_rate'])
    else:
        raise ValueError(f"Optimizer '{optimizer}' is not supported.")
    
    # Compile model
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=resource, 
                        validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)
    
    # Extract and return the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    return best_val_accuracy

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def train_and_evaluate(model_config: Dict[str, Any], resource: int) -> float:
    """
    Trains and evaluates a Keras model using the specified configuration
    and resources. The function configures a model based on provided hyperparameters,
    trains it for a given number of epochs, and evaluates its performance on a
    validation set.

    Parameters
    ----------
    model_config : Dict[str, Any]
        A dictionary containing the model's configuration. This includes any
        hyperparameters that define the model's architecture such as the number of 
        units in each layer, activation functions, etc. For example, 
        `{'units': 64, 'activation': 'relu'}`.
    resource : int
        The allocated resources for training the model, typically quantified
        as the number of epochs to train the model. This dictates how many
        complete passes through the training dataset the model will make.

    Returns
    -------
    float
        The highest validation accuracy achieved by the model after training
        is completed.

    Examples
    --------
    >>> from gofast.models.deep_search import train_and_evaluate
    >>> model_config = {'units': 128, 'activation': 'relu', 'layers': 3}
    >>> resource = 20  # Train for 20 epochs
    >>> val_accuracy = train_and_evaluate(model_config, resource)
    >>> print(f"Best Validation Accuracy: {val_accuracy:.2%}")

    Notes
    -----
    The function simplifies the process of configuring, training, and evaluating
    a model by abstracting away the details of data handling and model updates.
    It is crucial to provide a well-formed model configuration and appropriate
    resources to ensure effective training and reliable evaluation.

    See Also
    --------
    tf.keras.models.Model : The generic model instance in Keras which is used
                            to construct models.
    tf.keras.optimizers : Optimizers available in Keras, such as 'Adam', 'SGD',
                          which are used to update model weights based on the 
                          calculated gradients.

    References
    ----------
    .. [1] Chollet, François. "Deep Learning with Python." Manning Publications Co.,
           2017. Describes how to use Keras to build and train deep learning models.
    """

    # Define a simple model based on the configuration
    model = Sequential([
        Dense(model_config['units'], 
                              activation='relu', input_shape=(784,)),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(
        learning_rate=model_config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_val = x_val.reshape(-1, 784).astype('float32') / 255
    
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        epochs=resource, verbose=0)
    
    # Extract and return the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    
    return best_val_accuracy

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def train_epoch(
    model: _Model, 
    optimizer: _Optimizer, 
    x_train: ArrayLike, 
    y_train_actual: ArrayLike, 
    y_train_estimated: Optional[ArrayLike] = None, 
    lambda_value: float = 0.0, 
    batch_size: int = 32, 
    use_custom_loss: bool = False
) -> Tuple[float, float]:
    """
    Trains the model for one epoch over the provided training data and 
    calculates the training and validation loss.

    Parameters
    ----------
    model : tf.keras.Model
        The neural network model to train.
        This is the model that will be trained using the training data.

    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use for training.
        Optimizer applies the gradients to the model's parameters.

    x_train : np.ndarray
        Input features for training.
        These are the features on which the model will be trained.

    y_train_actual : np.ndarray
        Actual target values for training.
        These are the true values that the model is trying to predict.

    y_train_estimated : Optional[np.ndarray], default=None
        Estimated target values for training, used with custom loss.
        These values can be used to incorporate additional domain-specific 
        knowledge into the loss function.

    lambda_value : float, default=0.0
        The weighting factor for the custom loss component, if used.
        This parameter controls the influence of the additional term in 
        the custom loss function.

    batch_size : int, default=32
        The size of the batches to use for training.
        This defines the number of samples that will be propagated 
        through the network at once.

    use_custom_loss : bool, default=False
        Whether to use a custom loss function or default to MSE.
        If set to True, the custom loss function incorporating `lambda_value` 
        and `y_train_estimated` will be used.

    Returns
    -------
    Tuple[float, float]
        The mean training loss and validation loss for the epoch.
        The first value is the average training loss and the second is 
        the validation loss.
        
    Examples
    --------
    >>> from gofast.models.deep_search import train_epoch
    >>> import tensorflow as tf
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> optimizer = tf.keras.optimizers.Adam()
    >>> # Assume x_train, y_train_actual, and y_train_estimated are prepared
    >>> train_loss, val_loss = train_epoch(
    ...     model, optimizer, x_train, y_train_actual,
    ...     y_train_estimated=None, lambda_value=0.5, batch_size=64,
    ...     use_custom_loss=True)
    >>> print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")

    Notes
    -----
    This function is designed to train a neural network model for one epoch, 
    updating its weights using the specified optimizer. If `use_custom_loss` 
    is set to True, a custom loss function that incorporates `lambda_value` 
    and `y_train_estimated` will be used, allowing for more flexibility 
    in training.

    If `use_custom_loss` is True, the loss function used is:
    
    .. math::
    
        L(y_{\text{true}}, y_{\text{pred}}, y_{\text{estimated}}, \lambda) = 
        \text{MSE}(y_{\text{true}}, y_{\text{pred}}) + 
        \lambda \cdot \text{MSE}(y_{\text{true}}, y_{\text{estimated}})
    
    Otherwise, the standard mean squared error (MSE) loss is used.
    
    See Also
    --------
    custom_loss : Defines the custom loss function.
    tf.keras.Model : Base class for Keras models.
    tf.keras.optimizers.Optimizer : Base class for Keras optimizers.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """

    validate_keras_model(model, raise_exception=True)
    epoch_train_loss = []
    for x_batch, y_actual_batch, y_estimated_batch in _data_generator(
        x_train, y_train_actual, y_train_estimated, batch_size):
        
        with GradientTape() as tape:
            y_pred_batch = model(x_batch, training=True)
            if use_custom_loss and y_train_estimated is not None:
                loss = _custom_loss(
                    y_actual_batch, y_pred_batch, 
                    y_estimated_batch, lambda_value)
            else:
                loss = mean_squared_error(
                    y_actual_batch, y_pred_batch)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_train_loss.append(reduce_mean(loss).numpy())

    # Assume calculate_validation_loss is a predefined function
    val_loss = calculate_validation_loss(
        model, x_train, y_train_actual, 
        y_train_estimated, lambda_value, use_custom_loss)
    
    return np.mean(epoch_train_loss), val_loss

def _custom_loss(
    y_true: _Tensor, 
    y_pred: _Tensor, 
    y_estimated: _Tensor, 
    lambda_value: float, 
    reduction: str = 'auto', 
    loss_name: str = 'custom_loss'
    ) -> Callable:
    """
    Computes a custom loss value which is a combination of mean squared 
    error between true and predicted values, and an additional term 
    weighted by a lambda value.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth values.
        These are the actual values that the model is trying to predict.
        
    y_pred : tf.Tensor
        The predicted values.
        These are the values predicted by the model.

    y_estimated : tf.Tensor
        An estimated version of the ground truth values, used for an 
        additional term in the loss.
        This tensor allows for incorporating additional domain-specific 
        knowledge into the loss function, which might improve model 
        performance in certain contexts.

    lambda_value : float
        The weight of the additional term in the loss calculation.
        This value determines the importance of the additional term 
        relative to the mean squared error term in the total loss.

    reduction : str, optional
        Type of `tf.keras.losses.Reduction` to apply to loss.
        Default value is 'auto', which means the reduction option will be 
        determined by the current Keras backend. Other possible values 
        include 'sum_over_batch_size', 'sum', and 'none'.

    loss_name : str, optional
        Name to use for the loss.
        This allows the custom loss function to be referred to by a 
        specific name, which can be useful for logging and debugging.

    Returns
    -------
    Callable
        A callable that takes `y_true` and `y_pred` as inputs and returns 
        the loss value as an output.

    Examples
    --------
    >>> from gofast.models.deep_search import custom_loss
    >>> import tensorflow as tf
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(
    ...     y_true, y_pred, y_estimated=tf.zeros_like(y_pred), lambda_value=0.5))
    >>> # Assume X_train, y_train are prepared data
    >>> # model.fit(X_train, y_train, epochs=10)


    See Also
    --------
    tf.keras.losses.MeanSquaredError : Mean squared error loss function.
    tf.keras.Model : Base class for Keras models.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """
    check_consistent_length(y_true,y_pred )
    
    def loss(y_true, y_pred):
        mse = reduce_mean(square(y_true - y_pred), axis=-1)
        additional_term = reduce_mean(square(y_true - y_estimated), axis=-1)
        return mse + lambda_value * additional_term
    
    return Loss(name=loss_name, reduction=reduction)(loss)

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def calculate_validation_loss(
    model: _Model, 
    x_val: ArrayLike, 
    y_val_actual: ArrayLike, 
    y_val_estimated: Optional[ArrayLike] = None, 
    lambda_value: float = 0.0, 
    use_custom_loss: bool = False
) -> float:
    """
    Calculates the loss on the validation dataset using either a custom 
    loss function or mean squared error.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model for which to calculate the validation loss.
        This model has already been trained on the training data and will 
        be evaluated on the validation data.

    x_val : np.ndarray
        The input features of the validation dataset.
        These are the features used to predict the target values in the 
        validation set.

    y_val_actual : np.ndarray
        The actual target values of the validation dataset.
        These are the true values that the model is trying to predict in 
        the validation set.

    y_val_estimated : Optional[np.ndarray], default=None
        The estimated target values for the validation dataset, used with 
        custom loss.
        These values can incorporate additional domain-specific knowledge 
        and are used in the custom loss function.

    lambda_value : float, default=0.0
        The weighting factor for the custom loss component, if used.
        This parameter controls the influence of the additional term in 
        the custom loss function.

    use_custom_loss : bool, default=False
        Indicates whether to use the custom loss function or default to 
        MSE.
        If set to True, the custom loss function incorporating `lambda_value` 
        and `y_val_estimated` will be used.

    Returns
    -------
    float
        The mean validation loss calculated over all validation samples.
        This value represents the average loss of the model on the 
        validation data.

    Notes
    -----
    If `use_custom_loss` is True, the loss function used is:

    .. math::

        L(y_{\text{true}}, y_{\text{pred}}, y_{\text{estimated}}, \lambda) = 
        \text{MSE}(y_{\text{true}}, y_{\text{pred}}) + 
        \lambda \cdot \text{MSE}(y_{\text{true}}, y_{\text{estimated}})

    Otherwise, the standard mean squared error (MSE) loss is used. 
    The loss function helps in evaluating how well the model performs on 
    unseen data, guiding further model tuning and training adjustments.

    Examples
    --------
    >>> from gofast.models.deep_search import calculate_validation_loss
    >>> import tensorflow as tf
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> # Assume x_val, y_val_actual, and y_val_estimated are prepared
    >>> val_loss = calculate_validation_loss(
    ...     model, x_val, y_val_actual, 
    ...     y_val_estimated=None, lambda_value=0.5, 
    ...     use_custom_loss=True)
    >>> print(f"Validation Loss: {val_loss}")

    See Also
    --------
    custom_loss : Defines the custom loss function.
    tf.keras.Model : Base class for Keras models.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """
    validate_keras_model(model, raise_exception=True)
    val_preds = model.predict(x_val)
    if use_custom_loss and y_val_estimated is not None:
        loss = _custom_loss(y_val_actual, val_preds, y_val_estimated, lambda_value)
    else:
        loss = mean_squared_error(y_val_actual, val_preds)
    return np.mean(loss.numpy())


def _data_generator(
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

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)        
def evaluate_model(
    model_path: str, 
    Xt: ArrayLike, 
    yt: ArrayLike, 
    batch_size: int = 32, 
    denorm_range: Tuple[float, float] = (0.0, 1.0), 
    save_metrics: bool = False, 
    metrics: Optional[List[str]] = None
) -> dict:
    """
    Evaluates a saved model on a test dataset, denormalizes predictions and
    actual values using the provided denormalization range, calculates 
    specified metrics, and optionally saves the metrics to a JSON file.

    Parameters
    ----------
    model_path : str
        Path to the saved model.
        This is the file path where the model is saved, typically in a 
        format such as HDF5.

    Xt : np.ndarray
        Test dataset features.
        These are the input features for the test dataset.

    yt : np.ndarray
        Actual target values for the test dataset.
        These are the true values that the model aims to predict on the 
        test dataset.

    batch_size : int, default=32
        Batch size to use for prediction.
        This defines how many samples are processed in each iteration 
        during prediction.

    denorm_range : Tuple[float, float], default=(0.0, 1.0)
        A tuple containing the minimum and maximum values used for 
        denormalizing the dataset.
        This should match the range used to normalize the dataset during 
        training.

    save_metrics : bool, default=False
        Whether to save the evaluation metrics to a JSON file.
        If True, the calculated metrics will be saved to a file.

    metrics : Optional[List[str]], default=None
        List of evaluation metrics to calculate. Supported metrics are 
        'mse', 'mae', and 'r2_score'. If None, all supported metrics are 
        calculated.
        This allows for flexible evaluation based on specific requirements.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics.
        The keys are the metric names and the values are the calculated 
        metric values.

    Notes 
    -----
    The function assumes that `yt` and model predictions need to be 
    denormalized before calculating metrics. The `denorm_range` parameter 
    should match the normalization applied to your dataset.
    
    The denormalization of predictions and actual values is performed as:

    .. math::

        y_{\text{denorm}} = y_{\text{norm}} \cdot \\
            (y_{\text{max}} - y_{\text{min}}) + y_{\text{min}}

    where :math:`y_{\text{norm}}` is the normalized value, and 
    :math:`y_{\text{min}}` and :math:`y_{\text{max}}` are the minimum and 
    maximum values used for normalization.

    The metrics calculated can include:
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - R-squared (R2) score

    Examples
    --------
    >>> from gofast.models.deep_search import evaluate_model
    >>> model_path = 'path/to/your/model.h5'
    >>> x_test, y_test = load_your_test_data()  # Assume this loads your test data
    >>> test_metrics = evaluate_model(model_path, x_test, y_test, 
    ...                               denorm_range=(0, 100), 
    ...                               save_metrics=True, metrics=['mse', 'mae'])
    >>> print(test_metrics)


    See Also
    --------
    tf.keras.models.load_model : Loads a model saved via `model.save()`.
    sklearn.metrics.mean_squared_error : Calculates the MSE metric.
    sklearn.metrics.mean_absolute_error : Calculates the MAE metric.
    sklearn.metrics.r2_score : Calculates the R2 score metric.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """
    Xt, yt= check_X_y(Xt, yt)
    min_value, max_value = denorm_range
    model = load_model(model_path)
    predictions = model.predict(Xt, batch_size=batch_size)
    predictions_denormalized = denormalize(predictions, min_value, max_value)
    y_test_denormalized = denormalize(yt, min_value, max_value)
    
    test_metrics = {}
    if metrics is None or 'mse' in metrics:
        test_metrics['mse'] = mean_squared_error(y_test_denormalized,
                                                 predictions_denormalized)
    if metrics is None or 'mae' in metrics:
        test_metrics['mae'] = mean_absolute_error(y_test_denormalized,
                                                  predictions_denormalized)
    if metrics is None or 'r2_score' in metrics:
        test_metrics['r2_score'] = r2_score(y_test_denormalized, 
                                            predictions_denormalized)
    
    test_metrics['model_size_bytes'] = os.path.getsize(model_path)
    
    if save_metrics:
        with open(model_path + '_test_metrics.json', 'w') as json_file:
            json.dump(test_metrics, json_file)
    
    return test_metrics

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def train_model(
    model: '_Model', 
    x_train: ArrayLike, 
    y_train_actual: ArrayLike, 
    y_train_estimated: Optional[ArrayLike] = None, 
    lambda_value: float = 0.1, 
    num_epochs: int = 100, 
    batch_size: int = 32, 
    initial_lr: float = 0.001, 
    checkpoint_dir: str = './checkpoints', 
    patience: int = 10, 
    use_custom_loss: bool = True
) -> Tuple[List[float], List[float], str]:
    """
    Trains a TensorFlow/Keras model using either a custom loss function or 
    mean squared error, with early stopping based on validation loss improvement.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained.
        This is the neural network model built using the Keras API.

    x_train : np.ndarray
        Input features for training.
        These are the independent variables used to train the model.

    y_train_actual : np.ndarray
        Actual target values for training.
        These are the true values that the model is trying to predict.

    y_train_estimated : Optional[np.ndarray], default=None
        Estimated target values for training, used with custom loss. If `None`,
        the custom loss will ignore this component.
        These values can provide additional domain-specific knowledge for the 
        custom loss function.

    lambda_value : float, default=0.1
        Weighting factor for the custom loss component.
        This parameter determines the influence of the additional term in the 
        custom loss function.

    num_epochs : int, default=100
        Total number of epochs to train the model.
        This defines how many times the training algorithm will work through 
        the entire training dataset.

    batch_size : int, default=32
        Number of samples per batch of computation.
        This determines how many samples are processed before the model's 
        internal parameters are updated.

    initial_lr : float, default=0.001
        Initial learning rate for the Adam optimizer.
        This parameter controls how much to change the model in response to 
        the estimated error each time the model weights are updated.

    checkpoint_dir : str, default='./checkpoints'
        Directory where the model checkpoints will be saved.
        Checkpoints allow the model to be saved periodically during training, 
        which can be useful for long training sessions.

    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
        This is used to implement early stopping to avoid overfitting.

    use_custom_loss : bool, default=True
        Indicates whether to use the custom loss function.
        If set to True, the custom loss function will be used, otherwise, 
        mean squared error (MSE) will be used.

    Returns
    -------
    Tuple[List[float], List[float], str]
        A tuple containing the list of training losses, validation losses, and
        the checkpoint directory path.
        The training and validation losses provide insight into the model's 
        performance over time.

    Mathematical Formulation
    ------------------------
    The custom loss function, if used, is defined as:

    .. math::

        L(y_{\text{true}}, y_{\text{pred}}, y_{\text{estimated}}, \lambda) = 
        \text{MSE}(y_{\text{true}}, y_{\text{pred}}) + 
        \lambda \cdot \text{MSE}(y_{\text{true}}, y_{\text{estimated}})

    The training process includes an early stopping mechanism based on the 
    validation loss, which stops training if no improvement is observed for 
    `patience` epochs.

    Examples
    --------
    >>> from gofast.models.deep_search import train_model
    >>> model = build_your_model()  # Assume a function to build your model
    >>> x_train, y_train_actual = load_your_data()  # Assume a function to load your data
    >>> train_losses, val_losses, checkpoint_dir = train_model(
    ...     model, x_train, y_train_actual, num_epochs=50, batch_size=64,
    ...     initial_lr=0.01, checkpoint_dir='/tmp/model_checkpoints',
    ...     patience=5, use_custom_loss=False)
    >>> print(f"Training complete. Checkpoints saved to {checkpoint_dir}")

    Notes
    -----
    The function assumes that the model, data, and other parameters are properly 
    configured before training starts. The early stopping mechanism helps prevent 
    overfitting by stopping training when the validation loss stops improving.

    See Also
    --------
    custom_loss : Defines the custom loss function.
    tf.keras.callbacks.ModelCheckpoint : Callback to save the Keras model or 
                                         model weights at some frequency.
    tf.keras.callbacks.EarlyStopping : Callback to stop training when a monitored 
                                       metric has stopped improving.

    References
    ----------
    .. [1] Chollet, F. et al. "Deep Learning with Python." 
           Manning Publications, 2017.
    """
    validate_keras_model(model, raise_exception=True)
    optimizer = Adam(learning_rate=initial_lr)
    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        # Assume train_epoch is a function that trains the model for one 
        # epoch and returns the training and validation loss
        train_loss, val_loss = train_epoch(
            model, optimizer, x_train, y_train_actual, y_train_estimated, 
            lambda_value, batch_size, use_custom_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(checkpoint_dir, 'best_model'), save_format='tf')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return train_losses, val_losses, checkpoint_dir

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def plot_predictions(
    predicted_custom: ArrayLike, 
    predicted_mse: ArrayLike, 
    actual: ArrayLike, 
    filename: Optional[str] = None, 
    figsize: tuple = (12, 6), 
    title: str = 'Actual vs Predicted',
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None, 
    **kws
) -> 'matplotlib.axes.Axes':
    """
    Plots predicted values using custom loss and MSE against actual values,
    optionally saving the plot to a file.

    Parameters
    ----------
    predicted_custom : np.ndarray
        Predicted values using a custom loss function.
        These are the values predicted by the model that was trained using a custom 
        loss function.

    predicted_mse : np.ndarray
        Predicted values using mean squared error (MSE).
        These are the values predicted by the model that was trained using the mean 
        squared error loss function.

    actual : np.ndarray
        Actual values.
        These are the ground truth values against which the predictions are compared.

    filename : Optional[str], default=None
        Path and filename where the plot will be saved. If None, the plot is not saved.
        This allows for saving the plot to a file for later review or presentation.

    figsize : tuple, default=(12, 6)
        Figure size.
        This determines the size of the plot in inches (width, height).

    title : str, default='Actual vs Predicted'
        Title of the plot.
        This provides a title for the plot, giving context to what is being displayed.

    xlabel : Optional[str]
        Label for the X-axis. If None, the X-axis will not be labeled.
        This provides a label for the X-axis, explaining what is being measured along 
        the X-axis.

    ylabel : Optional[str]
        Label for the Y-axis. If None, the Y-axis will not be labeled.
        This provides a label for the Y-axis, explaining what is being measured along 
        the Y-axis.

    **kws
        Additional keyword arguments to pass to `plt.plot`.
        These can include any valid keyword arguments for matplotlib's plot function, 
        such as `linestyle`, `color`, etc.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the plot.
        This allows for further customization or manipulation of the plot after it is 
        created.
        
    Notes
    -----

    The plot compares the predicted values from two models (one trained with custom 
    loss and the other with MSE) against the actual values:

    .. math::

        \text{Predicted}_{\text{custom}} = f_{\text{custom}}(X)
        \text{Predicted}_{\text{mse}} = f_{\text{mse}}(X)
        \text{Actual} = y

    where :math:`f_{\text{custom}}` and :math:`f_{\text{mse}}` are the prediction 
    functions of the models trained with custom loss and MSE, respectively, and :math:`y` 
    are the actual values.


    This function is useful for visualizing the performance of two different models 
    against the actual data, allowing for a visual comparison of their predictions.

    Examples
    --------
    >>> from gofast.models.deep_search import plot_predictions
    >>> import numpy as np
    >>> predicted_custom = np.random.rand(100)
    >>> predicted_mse = np.random.rand(100)
    >>> actual = np.random.rand(100)
    >>> axes = plot_predictions(predicted_custom, predicted_mse, actual,
    ...                         filename='predictions.png',
    ...                         title='Wind Speed Prediction Comparison',
    ...                         xlabel='Time', ylabel='Speed',
    ...                         linestyle='--')
    >>> plt.show()

    See Also
    --------
    matplotlib.pyplot.plot : Plot lines and/or markers to the Axes.

    References
    ----------
    .. [1] Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment." 
           Computing in Science & Engineering, 9(3), 90-95.
    """
    plt.figure(figsize=figsize)
    plt.plot(predicted_custom, label='Predicted (Custom Loss)', 
             color='blue', **kws)
    plt.plot(predicted_mse, label='Predicted (MSE)', color='green', **kws)
    plt.plot(actual, label='Actual', color='orange', **kws)
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
    return plt.gca()

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def plot_errors(
    predicted_custom: ArrayLike, 
    predicted_mse: ArrayLike, 
    actual: ArrayLike, 
    filename: Optional[str] = None, 
    figsize: tuple = (12, 6), 
    title: str = 'Error in Predictions', 
    xlabel: Optional[str] = 'Time Steps', 
    ylabel: Optional[str] = 'Error',
    **kws
) -> 'matplotlib.axes.Axes':
    """
    Plots the absolute errors in predictions using custom loss and MSE against
    actual values. 

    Parameters
    ----------
    predicted_custom : np.ndarray
        Predicted values using a custom loss function.
        These are the values predicted by the model that was trained using a custom 
        loss function.

    predicted_mse : np.ndarray
        Predicted values using mean squared error (MSE).
        These are the values predicted by the model that was trained using the mean 
        squared error loss function.

    actual : np.ndarray
        Actual values to compare against predictions.
        These are the ground truth values against which the predictions are compared.

    filename : Optional[str], default=None
        Path and filename where the plot will be saved. If None, the plot is 
        not saved.
        This allows for saving the plot to a file for later review or presentation.

    figsize : tuple, default=(12, 6)
        Dimensions of the figure.
        This determines the size of the plot in inches (width, height).

    title : str, default='Error in Predictions'
        Title of the plot.
        This provides a title for the plot, giving context to what is being displayed.

    xlabel : Optional[str], default='Time Steps'
        Label for the X-axis. If None, the X-axis will not be labeled.
        This provides a label for the X-axis, explaining what is being measured along 
        the X-axis.

    ylabel : Optional[str], default='Error'
        Label for the Y-axis. If None, the Y-axis will not be labeled.
        This provides a label for the Y-axis, explaining what is being measured along 
        the Y-axis.

    **kws
        Additional keyword arguments to pass to `plt.plot`.
        These can include any valid keyword arguments for matplotlib's plot function, 
        such as `linestyle`, `color`, etc.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the plot.
        This allows for further customization or manipulation of the plot after it is 
        created.
        
    Notes
    -----
    The absolute error for each prediction is calculated as:

    .. math::

        \text{Error}_{\text{custom}} = |\text{Actual} - \text{Predicted}_{\text{custom}}|
        \text{Error}_{\text{mse}} = |\text{Actual} - \text{Predicted}_{\text{mse}}|

    where :math:`\text{Predicted}_{\text{custom}}` and :math:`\text{Predicted}_{\text{mse}}` 
    are the predictions from the models trained with custom loss and MSE, respectively, and 
    :math:`\text{Actual}` are the ground truth values.


    This function is useful for visualizing the prediction errors of two different models 
    against the actual data, allowing for a visual comparison of their performance.

    Examples
    --------
    >>> from gofast.models.deep_search import plot_errors
    >>> import numpy as np
    >>> predicted_custom = np.random.rand(100)
    >>> predicted_mse = np.random.rand(100)
    >>> actual = np.random.rand(100)
    >>> axes = plot_errors(predicted_custom, predicted_mse, actual,
    ...                    filename='errors_plot.png',
    ...                    title='Prediction Error Comparison',
    ...                    xlabel='Time', ylabel='Prediction Error',
    ...                    linestyle='--')
    >>> plt.show()

    See Also
    --------
    matplotlib.pyplot.plot : Plot lines and/or markers to the Axes.

    References
    ----------
    .. [1] Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment." 
           Computing in Science & Engineering, 9(3), 90-95.
    """

    error_custom = np.abs(actual - predicted_custom)
    error_mse = np.abs(actual - predicted_mse)
    
    plt.figure(figsize=figsize)
    plt.plot(error_custom, label='Error (Custom Loss)', color='blue', **kws)
    plt.plot(error_mse, label='Error (MSE)', color='green', **kws)
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
    return plt.gca()

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def make_future_predictions(
    model: Any, 
    last_known_sequence: ArrayLike, 
    n_features: int, 
    n_periods: int = 1, 
    output_feature_index: int = 0, 
    update_sequence: bool = True, 
    scaler: Optional[Any] = None
) -> ArrayLike:
    """
    Generate future predictions for a specified number of periods using 
    a trained model. 
    
    Function updates the sequence with each prediction and applies inverse 
    transformation using a provided scaler.

    Parameters
    ----------
    model : Any
        The trained predictive model that has a `predict` method. This model is 
        used to generate future predictions based on the last known sequence.
        Example:
        ```python
        model = tf.keras.models.load_model('model.h5')
        ```

    last_known_sequence : ArrayLike
        A numpy array representing the last known data sequence. This sequence is 
        the starting point for generating future predictions.
        Example:
        ```python
        last_known_sequence = np.array([...])
        ```

    n_features : int
        The total number of features in the dataset. This is required to 
        properly format the predictions and input sequence.

    n_periods : int, optional, default=1
        The number of future periods for which predictions are to be made, 
        defaulting to 1. This can represent any unit of time as specified by 
        (e.g., days, months, years).

    output_feature_index : int, optional, default=0
        Index of the target feature within the dataset for which predictions 
        are made. Defaults to 0, indicating the first feature.

    update_sequence : bool, optional, default=True
        Indicates whether the sequence should be updated with each new prediction. 
        If True, each new prediction is added to the sequence for subsequent 
        predictions.

    scaler : Optional[Any], optional
        An optional scaler object used for inverse transforming the predictions
        to their original scale. If None, predictions are returned without 
        scaling. This is useful when the data has been previously scaled for 
        training the model.

    Returns
    -------
    ArrayLike
        An array of predictions for the specified future periods, potentially  
        inverse transformed to their original scale if a scaler is provided.

    Notes
    -----

    The prediction process iteratively updates the input sequence with each new 
    prediction. At each step, the model generates a new prediction:

    .. math::

        y_t = f(X_{t-1})

    where :math:`y_t` is the prediction at time step `t`, and :math:`X_{t-1}` is the 
    sequence of input data up to time step `t-1`.

    If `update_sequence` is True, the new prediction is appended to the input 
    sequence for generating the next prediction:

    .. math::

        X_t = [X_{t-1}, y_t]

    This function is useful for generating future predictions from time series 
    data. The ability to update the sequence with each new prediction allows for 
    more accurate forecasting over multiple periods.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.models.deep_search import make_future_predictions
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> model = YourModel()
    >>> scaler = MinMaxScaler()
    >>> last_known_sequence = np.random.rand(1, 12, 5)  # Example sequence
    >>> predictions = make_future_predictions(model, last_known_sequence, 
    ...                                       n_features=5, n_periods=3, 
    ...                                       scaler=scaler)
    >>> print(predictions)

    See Also
    --------
    tf.keras.Model.predict : 
        Generates output predictions for the input samples.
    sklearn.preprocessing.MinMaxScaler : 
        Transforms features by scaling each feature to a given range.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    """

    validate_keras_model(model, raise_exception=True)
    input_sequence = np.copy(last_known_sequence)
    
    future_predictions = []
    for _ in range(n_periods):
        predicted = model.predict(input_sequence)[0, output_feature_index]
        future_predictions.append(predicted)
        
        if update_sequence:
            input_sequence = np.roll(input_sequence, -1, axis=1)
            new_row = np.zeros(n_features)
            new_row[output_feature_index] = predicted
            input_sequence[0, -1, :] = new_row

    predictions_full_features = np.zeros((len(future_predictions), n_features))
    predictions_full_features[:, output_feature_index] = future_predictions
    
    if scaler is not None:
        future_predictions = scaler.inverse_transform(
            predictions_full_features)[:, output_feature_index]
    else:
        # Ensuring return type consistency
        future_predictions = np.array(future_predictions)  
    
    return future_predictions

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def cross_validate_lstm(
    model: _Model,
    X: ArrayLike,
    y: ArrayLike, 
    tscv: Optional[Union[TimeSeriesSplit, int]] = 4,
    metric: Union[str, Callable] = "mse",
    scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None,
    epochs: int = 100,
    verbose: int = 0,
    **metric_kwargs
) -> list:
    """
    Performs cross-validation on an LSTM model with specified metrics 
    and scaling.

    Parameters
    ----------
    model : Sequential
        The LSTM model to evaluate.
        This model should be compiled and ready for training.

    X : np.ndarray
        The input data for the model, structured as sequences.
        Shape should be (samples, timesteps, features).

    y : np.ndarray
        The target data corresponding to the input sequences.
        Shape should be (samples,).

    tscv : Optional[Union[TimeSeriesSplit, int]], default=4
        The cross-validation splitting strategy as a `TimeSeriesSplit` instance 
        or an integer specifying the number of splits.
        Example: `TimeSeriesSplit(n_splits=5)` or `5`.

    metric : Union[str, Callable], default="mse"
        The metric for evaluating model performance. Can be a string 
        ('mse', 'accuracy') or a callable function.
        Example: `'accuracy'` for classification or a custom metric function.

    scaler : Optional[Union[MinMaxScaler, StandardScaler]], default=None
        The scaler instance used for inverse transforming the model predictions. 
        If None, predictions are not scaled.
        Example: `MinMaxScaler()` or `StandardScaler()`.

    epochs : int, default=100
        Number of training epochs for the LSTM model.
        This defines how many times the model will be trained on the entire dataset.

    verbose : int, default=0
        Verbosity mode for model training. 0 = silent, 1 = progress bar,
        2 = one line per epoch.
        Controls the amount of information displayed during training.

    **metric_kwargs
        Additional keyword arguments to be passed to the metric function.
        Example: `sample_weight`, `multioutput`, etc.

    Returns
    -------
    list
        Scores from each cross-validation fold based on the specified metric.
        The list contains the performance metrics for each fold.
        
    Notes
    -----

    Cross-validation involves splitting the data into training and validation sets 
    multiple times and evaluating the model on each split. The metric is calculated 
    for each fold and the average performance is used to assess the model.

    .. math::

        \text{CV Score} = \frac{1}{k} \sum_{i=1}^{k} \text{Metric}(\hat{y}_i, y_i)

    where :math:`k` is the number of splits, :math:`\hat{y}_i` are the predictions 
    for fold `i`, and :math:`y_i` are the true values for fold `i`.


    This function is useful for evaluating the performance of an LSTM model on 
    time series data using cross-validation. It allows for different metrics, 
    scalers, and verbosity levels to be used during the evaluation process.

    Examples
    --------
    >>> from keras.models import Sequential
    >>> from keras.layers import Dense, LSTM
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from gofast.models.deep_search import cross_validate_lstm
    >>> model = Sequential([
    ...     LSTM(50, activation='relu', input_shape=(10, 1)),
    ...     Dense(1)
    ... ])
    >>> X = np.random.rand(100, 10, 1)
    >>> y = np.random.rand(100)
    >>> scores = cross_validate_lstm(model, X, y, tscv=5, metric="mse", epochs=10)
    >>> print(scores)

    See Also
    --------
    sklearn.model_selection.TimeSeriesSplit : Time series cross-validator.
    sklearn.preprocessing.MinMaxScaler : 
        Transforms features by scaling each feature to a given range.
    sklearn.preprocessing.StandardScaler : 
        Standardize features by removing the mean and scaling to unit variance.

    References
    ----------
    .. [1] Bergmeir, C., & Benítez, J. M. (2012). "On the use of cross-validation for 
           time series predictor evaluation." Information Sciences, 191, 192-213.
    .. [2] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    """

    validate_keras_model(model, raise_exception=True) 
    X, y = check_X_y ( X, y )
    
    if isinstance(tscv, int):
        tscv = TimeSeriesSplit(n_splits=tscv)
    if not isinstance(tscv, TimeSeriesSplit):
        raise ValueError("tscv must be a TimeSeriesSplit instance or an integer.")

    if scaler and not hasattr(scaler, 'fit_transform'):
        raise ValueError("Scaler must be a MinMaxScaler, StandardScaler, "
                         "or have a 'fit_transform' method.")

    metric_fn = metric if callable(metric) else {
        "mse": mean_squared_error,
        "accuracy": accuracy_score
    }.get(metric)

    if metric_fn is None:
        raise ValueError(f"Metric {metric} is not supported or not callable.")

    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test, y_train, y_test = ( 
            X[train_index], X[test_index], y[train_index], y[test_index]
            )
        model.fit(X_train, y_train, epochs=epochs, verbose=verbose, **metric_kwargs)
        predictions = model.predict(X_test)

        if scaler:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        score = metric_fn(y_test, predictions, **metric_kwargs)
        scores.append(score)

    return scores


