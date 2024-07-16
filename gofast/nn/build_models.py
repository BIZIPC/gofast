# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Module for building various neural network models.
Includes functions for creating LSTM, MLP, Attention-based models, Autoencoders, 
CNNs, and other types of models.
"""

from ..api.types import _Loss, _Regularizer, _Optimizer, _Metric, _Sequential
from ..api.types import _Model, List, Optional, Union, Tuple
from ..tools.funcutils import ensure_pkg
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    # Lazy-load the required Keras dependencies
    Model = KERAS_DEPS.Model
    Sequential = KERAS_DEPS.Sequential
    Dense = KERAS_DEPS.Dense
    Dropout = KERAS_DEPS.Dropout
    BatchNormalization = KERAS_DEPS.BatchNormalization
    LSTM = KERAS_DEPS.LSTM
    Input = KERAS_DEPS.Input
    Conv2D = KERAS_DEPS.Conv2D
    MaxPooling2D = KERAS_DEPS.MaxPooling2D
    Flatten = KERAS_DEPS.Flatten
    Attention = KERAS_DEPS.Attention
    Concatenate = KERAS_DEPS.Concatenate
    Adam = KERAS_DEPS.Adam
    SGD = KERAS_DEPS.SGD
    RMSprop = KERAS_DEPS.RMSprop

__all__ = [
    "build_lstm_model",
    "build_mlp_model",
    "create_attention_model",
    "create_autoencoder_model",
    "create_cnn_model",
    "create_lstm_model"
]

DEP_MSG=dependency_message('build_lstm_model')

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def build_lstm_model(
    n_lag: Optional[int] = None,
    input_shape: Optional[Tuple[int, int]] = None,
    learning_rate: float = 0.01,
    activation: str = 'relu',
    loss: str = 'mse',
    units: int = 50,
    output_units: int = 1,
    optimizer: Union[str, _Optimizer] = 'adam',
    metrics: Optional[list] = None
) -> _Sequential: 
    """
    Constructs and compiles an LSTM model with customizable configurations, 
    allowing for flexible input shapes.

    Parameters
    ----------
    n_lag : int, optional
        The number of lag observations to use as input features. 
        Ignored if `input_shape` is provided.
        This parameter defines the number of past time steps used as inputs 
        to predict future values.

    input_shape : Tuple[int, int], optional
        Direct specification of the input shape (time_steps, features). Use 
        this instead of `n_lag` for non-uniform input shapes.
        This is useful for specifying the input dimensions directly, allowing 
        for more complex input configurations.

    learning_rate : float, optional
        The learning rate for the optimizer. Defaults to 0.01.
        This controls the step size during the gradient descent optimization.

    activation : str, optional
        Activation function for the LSTM layers. Defaults to 'relu'.
        Common choices include 'relu', 'tanh', and 'sigmoid'.

    loss : str, optional
        Loss function for model training. Defaults to 'mean_squared_error'.
        This determines how the model's predictions are penalized against the 
        true values.

    units : int, optional
        Number of units in the LSTM layer. Defaults to 50.
        This defines the dimensionality of the LSTM's output space.

    output_units : int, optional
        Number of units in the output layer. Suitable for regression tasks. 
        Defaults to 1.
        This is the number of predictions the model will output.

    optimizer : Union[str, Adam], optional
        Optimizer to use. Defaults to 'adam'. Can be an optimizer instance 
        for custom configurations.
        The optimizer updates the model parameters to minimize the loss function.

    metrics : list, optional
        Metrics to be evaluated by the model during training and testing.
        Examples include 'accuracy', 'mae', and 'mse'.

    Returns
    -------
    Sequential
        The compiled Keras Sequential LSTM model.
        The model is ready to be trained and evaluated on time series data.
        
    Notes
    -----
    The LSTM model processes the input sequences through several steps, each 
    involving matrix multiplications and non-linear transformations. The hidden 
    state updates as:

    .. math::

        h_t = \text{LSTM}(X_t, h_{t-1})

    where :math:`h_t` is the hidden state at time step `t`, and :math:`X_t` is 
    the input at time step `t`.

    The output layer generates predictions based on the final hidden state:

    .. math::

        \hat{y} = W \cdot h_T + b

    where :math:`\hat{y}` is the predicted output, :math:`W` and :math:`b` are 
    the weight matrix and bias vector, respectively, and :math:`h_T` is the 
    hidden state at the final time step.


    This function is versatile for constructing LSTM models tailored to specific 
    time series prediction tasks. By allowing customization of input shapes, 
    activation functions, and other hyperparameters, it provides flexibility in 
    model design.

    Examples
    --------
    >>> from gofast.models.deep_search import build_lstm_model
    >>> model = build_lstm_model(n_lag=10, learning_rate=0.01, activation='relu')
    >>> print(model.summary())
    
    >>> model = build_lstm_model(n_lag=10)
    >>> print(model.summary())
    >>> # With direct input shape
    >>> model = build_lstm_model(input_shape=(10, 2))
    >>> print(model.summary())

    See Also
    --------
    tf.keras.models.Sequential : Sequential model API.
    tf.keras.layers.LSTM : Long Short-Term Memory layer.
    tf.keras.optimizers.Adam : Optimizer that implements the Adam algorithm.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    .. [2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." 
           MIT Press.
    """
    # Determine input shape based on n_lag or direct input_shape
    if str(loss).lower() =='mse': 
        loss="mean_squared_error"
        
    if not input_shape:
        if n_lag is None:
            raise ValueError("Either n_lag or input_shape must be provided.")
        input_shape = (n_lag, 1)
    
    model = Sequential([
        LSTM(units=units, activation=activation, input_shape=input_shape),
        Dense(output_units)
    ])
    
    # Configure optimizer
    optimizer_instance = Adam(learning_rate=learning_rate
                              ) if optimizer == 'adam' else optimizer
    model.compile(optimizer=optimizer_instance, loss=loss, metrics=metrics or [])
    
    return model

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def build_mlp_model(
    input_dim: int, 
    output_classes: int = 6,
    hidden_units: List[int] = [14, 10], 
    dropout_rate: float = 0.5, 
    learning_rate: float = 0.001,
    activation_functions: Optional[List[str]] = None,
    optimizer: Union[str, _Optimizer] = 'adam',
    loss_function: Union[str, _Loss] = 'categorical_crossentropy',
    regularizer: Optional[_Regularizer] = None,
    include_batch_norm: bool = False,
    custom_metrics: List[Union[str, _Metric]] = ['accuracy'],
    initializer: str = 'glorot_uniform'
    ) -> _Sequential:
    """
    Constructs and compiles a Multilayer Perceptron (MLP) model, suitable for 
    classification tasks with multiple classes. This function sets up a neural 
    network architecture based on the provided specifications of layers, activation 
    functions, and other hyperparameters.

    Parameters
    ----------
    input_dim : int
        The number of input features. This defines the dimensionality of the 
        input layer of the MLP.
    output_classes : int, optional
        The number of classes for the output layer, which determines the 
        dimensionality of the output layer. Default is 6.
    hidden_units : List[int], optional
        Specifies the number of neurons in each hidden layer as a list. 
        Each element represents a layer, and its value the number of neurons. 
        Defaults to [14, 10].
    dropout_rate : float, optional
        The fraction of the input units to drop as a means of preventing 
        overfitting. Default is 0.5.
    learning_rate : float, optional
        The step size at each iteration while moving toward a minimum of a loss 
        function. Default is 0.001.
    activation_functions : Optional[List[str]], optional
        Specifies the activation functions for each hidden layer as a list. If None,
        'relu' is used for all layers by default.
    optimizer : Union[str, _Optimizer], optional
        The method or algorithm to optimize the parameter of the model. Can be 
        either a string identifier for a built-in optimizer or an Optimizer 
        instance. Default is 'adam'.
    loss_function : Union[str, _Loss], optional
        Specifies the loss function to be used with the model. Can be either a string
        identifier or a Loss instance. Default is 'categorical_crossentropy'.
    regularizer : Optional[_Regularizer], optional
        Regularizer function applied to the kernel weights of the layers. If None,
        no regularization is applied. Default is None.
    include_batch_norm : bool, optional
        Flag to decide whether to include batch normalization layers after each 
        hidden layer. Default is False.
    custom_metrics : List[Union[str, _Metric]], optional
        A list of metrics to be evaluated by the model during training and testing.
        Each metric can be a string identifier or a Metric instance. Default includes
        only 'accuracy'.
    initializer : str, optional
        Initializer for the kernel weights matrix in each layer. Defaults to
        'glorot_uniform', commonly known as Xavier uniform initializer.

    Returns
    -------
    model : Sequential
        The constructed and compiled Keras Sequential model ready for training.

    Examples
    --------
    >> from gofast.models.deep_search import build_mlp_model
    >>> from tensorflow.keras.optimizers import Adam
    >>> model = build_mlp_model(input_dim=20, output_classes=3, 
    ...                         hidden_units=[50, 30], dropout_rate=0.3,
    ...                         learning_rate=0.01, activation_functions=['relu', 'tanh'],
    ...                         optimizer=Adam(learning_rate=0.01), include_batch_norm=True,
    ...                         custom_metrics=['accuracy', 'precision'])
    >>> model.summary()

    Notes
    -----
    The MLP is commonly used for tabular datasets and can efficiently handle
    classification problems. However, it is less suited for data with spatial
    and temporal relationships, where convolutional or recurrent networks
    may perform better.

    See Also
    --------
    Sequential : The TensorFlow/Keras model class that is being used to construct the MLP.
    Adam : One of the common optimizers used with neural networks.
    categorical_crossentropy : A loss function used for multi-class classification.

    References
    ----------
    .. [1] Goodfellow, Ian, et al. "Deep Learning." MIT press, 2016.
    .. [2] Chollet, François. "Deep Learning with Python." Manning Publications Co., 2017.
    """

    if activation_functions is None:
        activation_functions = ['relu'] * len(hidden_units)

    # Initialize the Sequential model
    model = Sequential()
    
    # Add the input layer and first hidden layer
    model.add(Dense(hidden_units[0], input_shape=(input_dim,), 
                    activation=activation_functions[0], kernel_initializer=initializer,
                    kernel_regularizer=regularizer))
    if include_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Add additional hidden layers
    for units, activation in zip(hidden_units[1:], activation_functions[1:]):
        model.add(Dense(units, activation=activation, kernel_initializer=initializer,
                        kernel_regularizer=regularizer))
        if include_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Add the output layer with softmax activation for classification
    model.add(Dense(output_classes, activation='softmax'))

    # Configure the optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        opt = optimizer

    # Compile the model
    model.compile(optimizer=opt, loss=loss_function, metrics=custom_metrics)

    return model

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def create_attention_model(
    input_dim: int,
    seq_length: int,
    lstm_units: List[int],
    attention_units: int,
    dense_units: List[int],
    dropout_rate: float = 0.2,
    output_units: int = 1,
    output_activation: str = "sigmoid"
) -> '_Model':
    """
    Creates a sequence model with an attention mechanism dynamically based on the
    specified architecture parameters.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features per time step.
        This defines the number of features in the input data at each time step.

    seq_length : int
        Length of the input sequences.
        This defines the number of time steps in each input sequence.

    lstm_units : List[int]
        A list of units for each LSTM layer in the encoder part of the model.
        Each entry in the list corresponds to the number of units in a respective 
        LSTM layer.

    attention_units : int
        The number of units in the Attention layer.
        This defines the dimensionality of the attention mechanism output.

    dense_units : List[int]
        A list of units for each Dense layer following the attention mechanism.
        Each entry corresponds to the number of units in a respective dense layer.

    dropout_rate : float, default=0.2
        Dropout rate applied after each LSTM and Dense layer.
        Dropout helps prevent overfitting by randomly setting a fraction of input 
        units to 0 during training.

    output_units : int, default=1
        The number of units in the output layer.
        This defines the dimensionality of the output, such as the number of classes 
        in classification.

    output_activation : str, default='sigmoid'
        Activation function for the output layer.
        Common choices include 'sigmoid' for binary classification and 'softmax' 
        for multi-class classification.

    Returns
    -------
    tf.keras.Model
        The constructed attention-based sequence model as a Keras Model instance.
        The model consists of LSTM layers followed by an attention mechanism, dense 
        layers, and an output layer.
        
    Notes
    -----

    The attention mechanism can be defined as:

    .. math::

        \text{Attention}(h_t, H) = \text{softmax}(W_a \cdot \tanh(W_h \cdot H + W_s \cdot h_t))

    where :math:`h_t` is the hidden state at time step `t`, :math:`H` is the sequence of 
    hidden states, and :math:`W_a`, :math:`W_h`, and :math:`W_s` are weight matrices.

    The attention output is then combined with the LSTM output:

    .. math::

        c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i

    where :math:`\alpha_{t,i}` are the attention weights and :math:`h_i` are the hidden 
    states.


    This function constructs an attention-based model suitable for handling
    sequence data, enhancing the model's ability to focus on relevant parts
    of the input for making predictions. It's particularly useful for tasks
    such as sequence classification, time series forecasting, or any scenario
    where the importance of different parts of the input sequence may vary.

    Examples
    --------
    >>> from gofast.models.deep_search import create_attention_model
    >>> input_dim = 128  # Feature dimensionality
    >>> seq_length = 100  # Length of the sequence
    >>> lstm_units = [64, 64]  # Two LSTM layers
    >>> attention_units = 32  # Attention layer units
    >>> dense_units = [64, 32]  # Two Dense layers after attention
    >>> model = create_attention_model(input_dim, seq_length, lstm_units,
    ...                                attention_units, dense_units,
    ...                                output_units=10, output_activation='softmax')
    >>> model.summary()

    See Also
    --------
    tf.keras.layers.LSTM : Long Short-Term Memory layer - Hochreiter 1997.
    tf.keras.layers.Attention : Layer to compute the attention mechanism.
    tf.keras.layers.Dense : Regular densely-connected NN layer.

    References
    ----------
    .. [1] Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation 
           by Jointly Learning to Align and Translate." arXiv preprint arXiv:1409.0473.
    """

    inputs = Input(shape=(seq_length, input_dim))

    # Encoder (LSTM layers)
    x = inputs
    for units in lstm_units:
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)

    # Attention layer
    query = Dense(attention_units)(x)
    value = Dense(attention_units)(x)
    attention = Attention()([query, value])
    attention = Concatenate()([x, attention])

    # Dense layers after attention
    x = attention
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(output_units, activation=output_activation)(x)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def create_autoencoder_model(
    input_dim: int,
    encoder_layers: List[Tuple[int, Optional[str], float]],
    decoder_layers: List[Tuple[int, Optional[str], float]],
    code_activation: Optional[str] = None,
    output_activation: str = "sigmoid"
) -> '_Model':
    """
    Creates an Autoencoder model dynamically based on the specified architecture
    parameters for the encoder and decoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
        This is the size of the input layer for the autoencoder.

    encoder_layers : List[Tuple[int, Optional[str], float]]
        A list of tuples defining the encoder layers. Each tuple should contain:
            - The number of units (int)
            - An optional activation function (default to 'relu' if None)
            - Dropout rate after the layer (float)

    decoder_layers : List[Tuple[int, Optional[str], float]]
        A list of tuples defining the decoder layers. Each tuple mirrors the structure
        of `encoder_layers`:
            - The number of units (int)
            - An optional activation function (default to 'relu' if None)
            - Dropout rate after the layer (float)

    code_activation : Optional[str], default=None
        Activation function for the bottleneck (code) layer. If `None`, no activation
        is applied (linear activation).
        This is the central layer of the autoencoder where dimensionality reduction 
        occurs.

    output_activation : str, default='sigmoid'
        Activation function for the output layer of the decoder.
        This function is applied to the final output of the autoencoder.

    Returns
    -------
    tf.keras.Model
        The constructed Autoencoder model as a Keras Model instance.
        The model consists of an encoder and a decoder, connected through the 
        bottleneck layer.
        
    Notes
    -----

    The autoencoder consists of two main parts: the encoder and the decoder.

    - Encoder:
    .. math::

        h_i = f(W_i \cdot h_{i-1} + b_i)

    where :math:`h_i` is the activation of layer `i`, :math:`W_i` and :math:`b_i` 
    are the weights and biases of layer `i`, and :math:`f` is the activation function.

    - Decoder:
    .. math::

        \hat{x} = g(W_j \cdot h_j + b_j)

    where :math:`\hat{x}` is the reconstructed input, :math:`W_j` and :math:`b_j` 
    are the weights and biases of layer `j`, and :math:`g` is the activation function 
    for the output layer.


    This function constructs an Autoencoder model that is suitable for dimensionality
    reduction, feature learning, or unsupervised pretraining of a neural network. The
    flexibility in specifying layer configurations allows for easy experimentation
    with different architectures. Autoencoders are useful for learning efficient 
    codings of the input data in an unsupervised manner.

    Examples
    --------
    >>> from gofast.models.deep_search import create_autoencoder_model
    >>> input_dim = 784  # For MNIST images flattened to a vector
    >>> encoder_layers = [(128, 'relu', 0.2), (64, 'relu', 0.2)]
    >>> decoder_layers = [(128, 'relu', 0.2), (784, 'sigmoid', 0.0)]
    >>> autoencoder = create_autoencoder_model(input_dim, encoder_layers,
    ...                                        decoder_layers, code_activation='relu',
    ...                                        output_activation='sigmoid')
    >>> autoencoder.summary()

    See Also
    --------
    tf.keras.layers.Dense : Regular densely-connected NN layer.
    tf.keras.layers.Dropout : Applies Dropout to the input.

    References
    ----------
    .. [1] Hinton, G. E., & Salakhutdinov, R. R. (2006). 
           "Reducing the Dimensionality of Data with Neural Networks." 
           Science, 313(5786), 504-507.
    """

    # Input layer
    input_layer = Input(shape=(input_dim,))
    # Encoder
    x = input_layer
    for units, activation, dropout in encoder_layers:
        x = Dense(units, activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # Bottleneck (code layer)
    code_layer = Dense(decoder_layers[0][0], activation=code_activation)(x)

    # Decoder
    x = code_layer
    for units, activation, dropout in decoder_layers:
        x = Dense(units, activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # Output layer
    output_layer = Dense(input_dim, activation=output_activation)(x)

    # Model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    return autoencoder

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def create_cnn_model(
    input_shape: Tuple[int, int, int],
    conv_layers: List[Tuple[int, Tuple[int, int], Optional[str], float]],
    dense_units: List[int],
    dense_activation: str = "relu",
    output_units: int = 1,
    output_activation: str = "sigmoid"
) -> '_Model':
    """
    Creates a CNN model dynamically based on the specified architecture 
    parameters.
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        The shape of the input data, including the channels.
        This typically corresponds to (height, width, channels) for images.

    conv_layers : List[Tuple[int, Tuple[int, int], Optional[str], float]]
        A list of tuples for configuring the convolutional layers. Each tuple 
        should contain:
            - The number of filters (int)
            - The kernel size (Tuple[int, int])
            - Optional activation function (default to 'relu' if None)
            - Dropout rate after the layer (float)

    dense_units : List[int]
        A list containing the number of units in each dense layer before the 
        output.
        Each entry corresponds to the number of units in a respective dense 
        layer.

    dense_activation : str, default='relu'
        Activation function to use in the dense layers.
        Common choices include 'relu', 'tanh', and 'sigmoid'.

    output_units : int, default=1
        The number of units in the output layer.
        This defines the dimensionality of the output, such as the number of 
        classes in classification.

    output_activation : str, default='sigmoid'
        Activation function for the output layer.
        Common choices include 'sigmoid' for binary classification and 
        'softmax' for multi-class classification.

    Returns
    -------
    tf.keras.Model
        The constructed CNN model.
        The model is built sequentially with the specified number of 
        convolutional layers, each followed by an activation and dropout layer 
        (if specified), then dense layers, and ends with an output layer.

    Notes
    -----
    The convolutional layer applies the following operation:

    .. math::

        \text{Conv}(X) = \text{Activation}(\text{Conv2D}(X, W) + b)

    where :math:`X` is the input, :math:`W` and :math:`b` are the weights and 
    biases, and the activation function is applied element-wise.

    Dropout is applied as:

    .. math::

        \text{Dropout}(x) = \begin{cases} 
        0 & \text{with probability } p \\
        \frac{x}{1-p} & \text{with probability } 1-p 
        \end{cases}

    where :math:`p` is the dropout rate.

    This function constructs a CNN model suitable for image classification or 
    other tasks requiring spatial hierarchy extraction, allowing customization 
    of the network architecture, activation functions, and regularization 
    techniques.

    Examples
    --------
    >>> from gofast.models.deep_search import create_cnn_model
    >>> input_shape = (28, 28, 1)  # Example for MNIST
    >>> conv_layers = [
    ...     (32, (3, 3), 'relu', 0.2),
    ...     (64, (3, 3), 'relu', 0.2)
    ... ]
    >>> dense_units = [128]
    >>> model = create_cnn_model(input_shape, conv_layers, dense_units,
    ...                           output_units=10, output_activation='softmax')
    >>> model.summary()

    See Also
    --------
    tf.keras.layers.Conv2D : Convolution layer (e.g. spatial convolution over images).
    tf.keras.layers.Dense : Regular densely-connected NN layer.
    tf.keras.layers.Dropout : Applies Dropout to the input.

    References
    ----------
    .. [1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). 
           Gradient-based learning applied to document recognition. 
           Proceedings of the IEEE, 86(11), 2278-2324.
    """
    model = Sequential()
    for i, (filters, kernel_size, activation, dropout) in enumerate(conv_layers):
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation or 'relu',
                             input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation or 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Flatten())
    for units in dense_units:
        model.add(Dense(units, activation=dense_activation))
    model.add(Dense(output_units, activation=output_activation))
    return model

@ensure_pkg(
    KERAS_BACKEND or "keras",
    extra=DEP_MSG
)
def create_lstm_model(
    input_shape: Tuple[int, int], 
    n_units_list: List[int], 
    n_future: int, 
    activation: str = "relu", 
    dropout_rate: float = 0.2, 
    dense_activation: Optional[str] = None
) -> '_Model':
    """
    Creates an LSTM model dynamically based on the specified architecture 
    parameters.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        The shape of the input data, excluding the batch size.
        It is typically in the form of (timesteps, features).

    n_units_list : List[int]
        A list containing the number of units in each LSTM layer.
        Each entry in the list corresponds to the number of units in a 
        respective LSTM layer.

    n_future : int
        The number of units in the output dense layer, often corresponding
        to the prediction horizon.
        This defines how many future time steps the model will predict.

    activation : str, default='relu'
        Activation function to use in the LSTM layers.
        Common choices include 'relu', 'tanh', and 'sigmoid'.

    dropout_rate : float, default=0.2
        Dropout rate for the dropout layers following each LSTM layer.
        Dropout helps prevent overfitting by randomly setting a fraction of 
        input units to 0 during training.

    dense_activation : Optional[str], default=None
        Activation function for the output dense layer. If `None`, a linear
        activation is applied.
        This can be any activation function like 'linear', 'relu', 'sigmoid', 
        etc.

    Returns
    -------
    tf.keras.Model
        The constructed LSTM model.
        The model is built sequentially with the specified number of LSTM 
        layers, each followed by a dropout layer, and ends with a dense layer 
        for output.

    Notes
    -----
    The LSTM model is defined by the following equations:

    .. math::

        \text{LSTM}_t = f(W \cdot [h_{t-1}, x_t] + b)

    where :math:`h_{t-1}` is the hidden state from the previous time step, 
    :math:`x_t` is the input at the current time step, :math:`W` and :math:`b` 
    are the weights and biases, and :math:`f` is the activation function.

    Dropout is applied as:

    .. math::

        \text{Dropout}(x) = \begin{cases} 
        0 & \text{with probability } p \\
        \frac{x}{1-p} & \text{with probability } 1-p 
        \end{cases}

    where :math:`p` is the dropout rate.

    
    This function constructs an LSTM model suitable for time series forecasting 
    or sequence prediction tasks, allowing customization of the network 
    architecture, activation functions, and regularization techniques. The model 
    includes LSTM layers followed by dropout layers to prevent overfitting, and 
    a final dense layer for making predictions.

    Examples
    --------
    >>> from gofast.models.deep_search import create_lstm_model
    >>> input_shape = (10, 1)  # 10 time steps, 1 feature
    >>> n_units_list = [50, 25]  # Two LSTM layers with 50 and 25 units
    >>> n_future = 1  # Predicting a single future value
    >>> model = create_lstm_model(input_shape, n_units_list, n_future,
    ...                           activation='tanh', dropout_rate=0.1,
    ...                           dense_activation='linear')
    >>> model.summary()

    See Also
    --------
    tf.keras.layers.LSTM : Long Short-Term Memory layer - Hochreiter 1997.
    tf.keras.layers.Dropout : Applies Dropout to the input.
    tf.keras.layers.Dense : Regular densely-connected NN layer.

    References
    ----------
    .. [1] Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." 
           Neural Computation, 9(8), 1735-1780.
    """

    model = Sequential()
    for i, n_units in enumerate(n_units_list):
        is_return_sequences = i < len(n_units_list) - 1
        model.add(LSTM(units=n_units, activation=activation, 
                       return_sequences=is_return_sequences, 
                       input_shape=input_shape if i == 0 else None))
        model.add(Dropout(dropout_rate))
    model.add(Dense(units=n_future, activation=dense_activation))
    return model

