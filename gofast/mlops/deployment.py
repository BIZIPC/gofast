# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Assist with deploying models efficiently to various environments, 
such as cloud platforms, edge devices, or on-premise servers.
"""

import os
import random 
import time
from numbers import Real 
from typing import Any, Optional, Tuple, Dict

from sklearn.utils._param_validation import Interval, StrOptions

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from .._gofastlog import gofastlog 
from ..api.property import BaseLearner 
from ..compat.sklearn import validate_params 
from ..tools.depsutils import ensure_pkgs
from ..tools.validator import parameter_validator 

logger=gofastlog.get_gofast_logger(__name__)


__all__=[
    "ModelExporter", "APIDeployment", "CloudDeployment", "ABTesting"
    ]


EXTRA_MSG= ( 
    "The {pkg} is required for this functionality. Please install it to proceed."
    )

@ensure_pkgs(
    "torch",
    extra= EXTRA_MSG.format(pkg="torch"),
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class ModelExporter(BaseLearner):
    """
    Manages model export and serialization to various formats such as
    ONNX, TensorFlow, and PyTorch. Supports compression techniques like
    quantization and pruning, as well as version control.

    Parameters
    ----------
    model : object
        The machine learning model to be exported. It can be a PyTorch
        or TensorFlow model.
    model_name : str
        A name for the model, used in filenames and logging.
    versioning : bool, optional
        Whether to enable version control for the exported models.
        Defaults to ``True``.

    Attributes
    ----------
    model : object
        The machine learning model to be exported.
    model_name : str
        The name of the model.
    version : int or None
        The current version of the model for version control. If
        versioning is disabled, this is ``None``.

    Notes
    -----
    The ``ModelExporter`` class provides functionalities for exporting
    machine learning models to different formats, applying compression
    techniques, and managing versions. It supports PyTorch and
    TensorFlow models.

    Examples
    --------
    >>> from gofast.mlops.deployment import ModelExporter
    >>> model = ...  #  PyTorch or TensorFlow model
    >>> exporter = ModelExporter(model, 'my_model')
    >>> exporter.export_to_onnx('path/to/model.onnx')

    See Also
    --------
    torch.onnx.export : Exports PyTorch models to ONNX format.
    tf.saved_model.save : Exports TensorFlow models as SavedModel.

    References
    ----------
    .. [1] PyTorch Documentation: https://pytorch.org/docs/stable/
    .. [2] TensorFlow Documentation: https://www.tensorflow.org/api_docs
    .. [3] ONNX Documentation: https://github.com/onnx/onnx

    """

    @validate_params({
        'model': [object],
        'model_name': [str],
        'versioning': [bool]
    })
    def __init__(self, model: Any, model_name: str, versioning: bool = True):
        """
        Initializes the ``ModelExporter`` with a model and its name.

        Parameters
        ----------
        model : object
            The model to be exported.
        model_name : str
            A name for the model.
        versioning : bool, optional
            Whether to enable version control for the exported models.
            Defaults to ``True``.

        Examples
        --------
        >>> exporter = ModelExporter(model, 'my_model')

        """
        self.model = model
        self.model_name = model_name
        self.version = 1 if versioning else None  

    @validate_params({
        'export_path': [str],
        'input_shape': [tuple, None],
        'opset_version': [int]
    })
    def export_to_onnx(
        self,
        export_path: str,
        input_shape: Optional[Tuple] = None,
        opset_version: int = 12
    ):
        """
        Exports the model to ONNX format with additional flexibility.

        Parameters
        ----------
        export_path : str
            Path where the ONNX file will be saved.
        input_shape : tuple, optional
            Input shape to generate dummy input. Defaults to
            ``(1, 3, 224, 224)`` if not provided.
        opset_version : int, optional
            ONNX opset version to use for export. Defaults to ``12``.

        Notes
        -----
        The model is exported to ONNX format using the specified opset
        version. A dummy input of the given shape is used for tracing
        the model.

        Examples
        --------
        >>> exporter.export_to_onnx('model.onnx', input_shape=(1, 3, 224, 224))

        """
        import torch

        logger.info(f"Exporting '{self.model_name}' to ONNX format.")
        if input_shape is None:
            input_shape = (1, 3, 224, 224)  # Default input shape

        if not isinstance(self.model, torch.nn.Module):
            logger.error("Model must be a PyTorch model for ONNX export.")
            raise TypeError("Model must be a PyTorch model for ONNX export.")

        dummy_input = torch.randn(*input_shape)
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                export_path,
                opset_version=opset_version
            )
            logger.info(
                f"Model exported to ONNX at '{export_path}' with opset version {opset_version}"
            )
        except Exception as e:
            logger.error(f"Failed to export '{self.model_name}' to ONNX: {e}")
            raise


    @ensure_pkgs(
        "tensorflow",
        extra="The 'tensorflow' package is required for this functionality.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({
        'export_path': [str],
        'model_type': [StrOptions({'SavedModel', 'HDF5'})]
    })
    def export_to_tensorflow(
        self,
        export_path: str,
        model_type: str = 'SavedModel'
    ):
        """
        Exports the model to TensorFlow format.

        Parameters
        ----------
        export_path : str
            Path where the TensorFlow model will be saved.
        model_type : {'SavedModel', 'HDF5'}, optional
            TensorFlow format to export. Defaults to ``'SavedModel'``.

        Notes
        -----
        Exports the model to TensorFlow format. Supports exporting as
        a SavedModel or HDF5 file.

        Examples
        --------
        >>> exporter.export_to_tensorflow('model_dir', model_type='SavedModel')

        """
        import tensorflow as tf

        logger.info(
            f"Exporting '{self.model_name}' to TensorFlow format ({model_type})."
        )

        if not isinstance(self.model, tf.keras.Model):
            logger.error("Model must be a TensorFlow Keras model for TensorFlow export.")
            raise TypeError("Model must be a TensorFlow Keras model for TensorFlow export.")

        try:
            if model_type == 'SavedModel':
                tf.saved_model.save(self.model, export_path)
                logger.info(f"Model exported as SavedModel at '{export_path}'")
            elif model_type == 'HDF5':
                self.model.save(f"{export_path}.h5")
                logger.info(f"Model exported as HDF5 at '{export_path}.h5'")
        except Exception as e:
            logger.error(
                f"Failed to export '{self.model_name}' to TensorFlow format: {e}"
            )
            raise

    @ensure_pkgs(
        "onnx",
        extra="The 'onnx' package is required if 'include_onnx' is set to 'True'", 
        partial_check= True,
        condition= lambda *args, **kwargs: kwargs.get("include_onnx")==True,
        )
    @validate_params({
        'export_path': [str],
        'include_onnx': [bool],
        'input_shape': [tuple, None]
    })
    def export_to_torch(
        self,
        export_path: str,
        include_onnx: bool = False,
        input_shape: Optional[Tuple] = None
    ):
        """
        Exports the model to PyTorch format (.pt), with an option to
        also export to ONNX.

        Parameters
        ----------
        export_path : str
            Path where the PyTorch model will be saved.
        include_onnx : bool, optional
            Whether to also export the model to ONNX format. Defaults
            to ``False``.
        input_shape : tuple, optional
            Input shape for ONNX export. Required if ``include_onnx``
            is ``True``. Defaults to ``None``.

        Notes
        -----
        Saves the PyTorch model's state dictionary to the specified
        path. If ``include_onnx`` is ``True``, the model is also
        exported to ONNX format using the provided ``input_shape``.

        Examples
        --------
        >>> exporter.export_to_torch('model.pt', include_onnx=True, input_shape=(1, 3, 224, 224))

        """
        import torch

        logger.info(f"Exporting '{self.model_name}' to PyTorch format.")

        if not isinstance(self.model, torch.nn.Module):
            logger.error("Model must be a PyTorch model for PyTorch export.")
            raise TypeError("Model must be a PyTorch model for PyTorch export.")

        try:
            torch.save(self.model.state_dict(), export_path)
            logger.info(f"Model exported to PyTorch at '{export_path}'")

            if include_onnx:
                if input_shape is None:
                    logger.error("Input shape must be provided for ONNX export.")
                    raise ValueError("Input shape must be provided for ONNX export.")

                logger.info("Also exporting to ONNX format.")
                onnx_export_path = f"{os.path.splitext(export_path)[0]}.onnx"
                self.export_to_onnx(onnx_export_path, input_shape)
        except Exception as e:
            logger.error(f"Failed to export '{self.model_name}' to PyTorch: {e}")
            raise

    @validate_params({
        'method': [StrOptions({'quantization', 'pruning'})],
    })
    def compress_model(self, method: str = "quantization", **kwargs):
        """
        Compresses the model using different methods such as
        quantization or pruning.

        Parameters
        ----------
        method : {'quantization', 'pruning'}, optional
            Compression method to use. Defaults to ``'quantization'``.
        **kwargs : dict
            Additional arguments for specific compression methods.

        Notes
        -----
        The model can be compressed using quantization or pruning
        techniques. Additional parameters specific to each method can
        be passed via ``**kwargs``.

        Examples
        --------
        >>> exporter.compress_model(method='quantization', quantize_weights=True)

        """
        logger.info(f"Compressing model '{self.model_name}' using {method}.")
        try:
            if method == "quantization":
                self._apply_quantization(**kwargs)
            elif method == "pruning":
                self._apply_pruning(**kwargs)
        except Exception as e:
            logger.error(f"Failed to compress model '{self.model_name}': {e}")
            raise

    @ensure_pkgs(
        "tensorflow",
        extra="The 'tensorflow' library is required if framework='tensorflow'", 
        partial_check= True,
        condition= lambda *args, **kwargs: kwargs.get("framework")=="tensorflow"
        )
    def _apply_quantization(
        self,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
        framework: str = "pytorch",
        quantization_method: str = "dynamic",
        custom_layers: Optional[Dict] = None,
        calibration_data: Optional[Any] = None
    ):
        """
        Applies quantization to the model for more efficient deployment.

        Parameters
        ----------
        quantize_weights : bool, optional
            Whether to quantize model weights. Defaults to ``True``.
        quantize_activations : bool, optional
            Whether to quantize activations. Defaults to ``True``.
        framework : str, optional
            Framework to apply quantization ('pytorch', 'tensorflow').
            Defaults to ``'pytorch'``.
        quantization_method : str, optional
            The quantization method to apply ('dynamic', 'static',
            'post_training'). Defaults to ``'dynamic'`` for PyTorch and
            ``'post_training'`` for TensorFlow.
        custom_layers : dict, optional
            A dictionary of layers to apply custom quantization to
            (PyTorch only). Example: ``{torch.nn.Linear: torch.qint8}``.
        calibration_data : optional
            Data used for calibration in static quantization.

        Notes
        -----
        Quantization reduces model size and increases inference speed by
        reducing the precision of weights and activations.

        Examples
        --------
        >>> exporter._apply_quantization(quantize_weights=True, framework='pytorch')

        """
        logger.info(
            f"Applying {quantization_method} quantization for {framework}."
        )

        if framework == "pytorch":
            import torch
            if not isinstance(self.model, torch.nn.Module):
                logger.error("Model must be a PyTorch model for PyTorch quantization.")
                raise TypeError("Model must be a PyTorch model for PyTorch quantization.")

            if quantization_method == "dynamic":
                if quantize_weights:
                    logger.info("Applying dynamic quantization for weights.")
                    self.model = torch.quantization.quantize_dynamic(
                        self.model,
                        custom_layers if custom_layers else {torch.nn.Linear},
                        dtype=torch.qint8
                    )
            elif quantization_method == "static":
                if quantize_weights or quantize_activations:
                    logger.info("Applying static quantization for weights and activations.")
                    self.model.eval()
                    self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    torch.quantization.prepare(self.model, inplace=True)
                    if calibration_data is None:
                        calibration_data = torch.randn(1, 3, 224, 224)
                    self.model(calibration_data)
                    torch.quantization.convert(self.model, inplace=True)
            else:
                raise ValueError(f"Unsupported quantization method for PyTorch: {quantization_method}")

        elif framework == "tensorflow":
            import tensorflow as tf
            if not isinstance(self.model, tf.keras.Model):
                logger.error("Model must be a TensorFlow Keras model for TensorFlow quantization.")
                raise TypeError("Model must be a TensorFlow Keras model for TensorFlow quantization.")

            if quantization_method == "post_training":
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                if quantize_weights:
                    logger.info("Applying post-training quantization for weights.")
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                if quantize_activations:
                    logger.info("Quantizing activations during post-training quantization.")
                    converter.target_spec.supported_types = [tf.float16]
                tflite_model = converter.convert()
                tflite_path = f"{self.model_name}_quantized.tflite"
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)
                logger.info(f"Model exported as TFLite at '{tflite_path}'.")
            else:
                raise ValueError(f"Unsupported quantization method for TensorFlow: {quantization_method}")

        else:
            raise ValueError(f"Unsupported framework for quantization: {framework}")

        logger.info(f"Quantization ({quantization_method}) complete for {framework}.")

    def _apply_pruning(
        self,
        prune_percentage: float = 0.2,
        prune_method: str = "unstructured",
        custom_layers: Optional[Dict] = None
    ):
        """
        Applies pruning to the model to reduce size, either using
        unstructured or structured pruning.

        Parameters
        ----------
        prune_percentage : float, optional
            The percentage of weights to prune. Defaults to ``0.2``.
        prune_method : {'unstructured', 'structured'}, optional
            Pruning method to use. Defaults to ``'unstructured'``.
        custom_layers : dict, optional
            Dictionary of layers/modules to apply custom pruning.
            Example: ``{torch.nn.Linear: 0.5}``.

        Notes
        -----
        Pruning reduces model size by removing less important weights.

        Examples
        --------
        >>> exporter._apply_pruning(prune_percentage=0.3, prune_method='structured')

        """
        import torch

        logger.info(
            f"Applying {prune_method} pruning to {prune_percentage * 100}% of model weights."
        )

        if not isinstance(self.model, torch.nn.Module):
            logger.error("Model must be a PyTorch model for pruning.")
            raise TypeError("Model must be a PyTorch model for pruning.")

        if prune_method == "unstructured":
            self._apply_unstructured_pruning(prune_percentage, custom_layers)
        elif prune_method == "structured":
            self._apply_structured_pruning(prune_percentage, custom_layers)
        else:
            raise ValueError(f"Unsupported pruning method: {prune_method}")

        logger.info("Pruning complete.")

    def _apply_unstructured_pruning(
        self,
        prune_percentage: float,
        custom_layers: Optional[Dict] = None
    ):
        """
        Applies unstructured pruning, where weights are pruned globally
        without considering structured connections.

        Parameters
        ----------
        prune_percentage : float
            Percentage of weights to prune.
        custom_layers : dict, optional
            Layers and percentage of pruning specific to each layer.

        """
        import torch.nn.utils.prune as prune

        logger.info(f"Applying unstructured pruning to {prune_percentage * 100}% of weights.")

        if custom_layers:
            for layer_type, percentage in custom_layers.items():
                logger.info(f"Pruning {percentage * 100}% of {layer_type} layers.")
                prune.global_unstructured(
                    [
                        (module, 'weight')
                        for module in self.model.modules()
                        if isinstance(module, layer_type)
                    ],
                    pruning_method=prune.L1Unstructured,
                    amount=percentage
                )
        else:
            prune.global_unstructured(
                [
                    (module, 'weight')
                    for module in self.model.modules()
                    if hasattr(module, 'weight')
                ],
                pruning_method=prune.L1Unstructured,
                amount=prune_percentage
            )

        logger.info("Unstructured pruning complete.")

    def _apply_structured_pruning(
        self,
        prune_percentage: float,
        custom_layers: Optional[Dict] = None
    ):
        """
        Applies structured pruning, where entire filters or channels are
        pruned.

        Parameters
        ----------
        prune_percentage : float
            Percentage of weights to prune.
        custom_layers : dict, optional
            Layers and percentage of pruning specific to each layer.

        """
        import torch.nn.utils.prune as prune

        logger.info(f"Applying structured pruning to {prune_percentage * 100}% of filters/channels.")

        if custom_layers:
            for layer_type, percentage in custom_layers.items():
                logger.info(f"Pruning {percentage * 100}% of {layer_type} filters/channels.")
                for module in self.model.modules():
                    if isinstance(module, layer_type):
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=percentage,
                            n=2,
                            dim=0
                        )
        else:
            for module in self.model.modules():
                if hasattr(module, 'weight'):
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=prune_percentage,
                        n=2,
                        dim=0
                    )

        logger.info("Structured pruning complete.")

    @validate_params({
        'export_path': [str],
        'input_shape': [tuple, None]
    })
    def version_control(
        self,
        export_path: str,
        input_shape: Optional[Tuple] = None
    ):
        """
        Saves the model with version control, tracking export versions.

        Parameters
        ----------
        export_path : str
            Base path where the model will be saved.
        input_shape : tuple, optional
            Input shape for ONNX export. Defaults to
            ``(1, 3, 224, 224)``.

        Notes
        -----
        If versioning is enabled, the model is saved with a version
        suffix. The version number is incremented after each save.

        Examples
        --------
        >>> exporter.version_control('model_v', input_shape=(1, 3, 224, 224))

        """
        import torch 
        import tensorflow as tf 
        if self.version is None:
            logger.warning("Version control is disabled.")
            return

        logger.info(f"Saving model version {self.version}.")
        if input_shape is None:
            input_shape = (1, 3, 224, 224)

        export_with_version = f"{export_path}_v{self.version}"
        try:
            if isinstance(self.model, torch.nn.Module):
                self.export_to_torch(export_with_version + '.pt')
            elif isinstance(self.model, tf.keras.Model):
                self.export_to_tensorflow(export_with_version)
            else:
                logger.error("Unsupported model type for version control.")
                raise TypeError("Unsupported model type for version control.")

            self.version += 1
            logger.info(f"Model version {self.version} saved at '{export_with_version}'.")
        except Exception as e:
            logger.error(f"Failed to save model version {self.version}: {e}")
            raise


@ensure_pkgs(
    "fastapi, flask, uvicorn, pydantic",
    extra="The 'fastapi', 'flask', 'uvicorn', and 'pydantic' packages are required "
          "for this functionality. Please install them to proceed.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA
)
class APIDeployment(BaseLearner):
    """
    Manages deployment of models as APIs using FastAPI or Flask. Supports
    versioning, scaling, monitoring, rate limiting, and graceful shutdown.

    Parameters
    ----------
    model : object
        The machine learning model to serve via the API.
    model_name : str
        Name of the model being served.
    api_type : {'FastAPI', 'Flask'}, optional
        API framework to use. Defaults to ``'FastAPI'``.
    max_requests : int, optional
        Maximum number of requests allowed before rate limiting kicks in.
        Defaults to ``1000``.

    Attributes
    ----------
    model : object
        The machine learning model to serve.
    model_name : str
        Name of the model being served.
    api_type : str
        API framework being used.
    max_requests : int
        Maximum number of requests allowed.
    request_count : int
        Counter for the number of requests received.
    start_time : float
        Timestamp when the API server started.
    app : FastAPI or Flask
        The API application instance.

    Notes
    -----
    The ``APIDeployment`` class allows you to deploy a machine learning model
    as an API using either FastAPI or Flask. It includes features such as
    rate limiting, health checks, version control, and graceful shutdown.

    Examples
    --------
    >>> from gofast.mlops.deployment import APIDeployment
    >>> model = ...  # Your machine learning model
    >>> api = APIDeployment(model, 'my_model', api_type='FastAPI')
    >>> api.create_api()
    >>> api.serve_api(host='0.0.0.0', port=8000)

    See Also
    --------
    FastAPI : A modern, fast web framework for building APIs with Python.
    Flask : A lightweight WSGI web application framework.

    References
    ----------
    .. [1] FastAPI Documentation: https://fastapi.tiangolo.com/
    .. [2] Flask Documentation: https://flask.palletsprojects.com/
    .. [3] Uvicorn Documentation: https://www.uvicorn.org/

    """

    @validate_params({
        'model': [object],
        'model_name': [str],
        'api_type': [StrOptions({'FastAPI', 'Flask'})],
        'max_requests': [int]
    })
    def __init__(
        self,
        model: Any,
        model_name: str,
        api_type: str = "FastAPI",
        max_requests: int = 1000
    ):
        """
        Initializes the ``APIDeployment`` class.

        Parameters
        ----------
        model : object
            The machine learning model to serve via the API.
        model_name : str
            Name of the model being served.
        api_type : {'FastAPI', 'Flask'}, optional
            API framework to use. Defaults to ``'FastAPI'``.
        max_requests : int, optional
            Maximum number of requests allowed before rate limiting kicks
            in. Defaults to ``1000``.

        Raises
        ------
        ValueError
            If an unsupported API type is specified.

        Examples
        --------
        >>> api = APIDeployment(model, 'my_model', api_type='FastAPI')

        """
        self.model = model
        self.model_name = model_name
        self.api_type = api_type
        self.max_requests = max_requests
        self.request_count = 0
        self.start_time = time.time()

        if api_type == "FastAPI":
            from fastapi import FastAPI
            self.app = FastAPI()
        elif api_type == "Flask":
            from flask import Flask
            self.app = Flask(__name__)
        else:
            raise ValueError("Unsupported API type. Choose 'FastAPI' or 'Flask'.")

    def create_api(self):
        """
        Creates API endpoints to serve predictions from the model. Includes
        rate limiting and request logging.

        Notes
        -----
        This method defines the `/predict` endpoint for making predictions
        and includes rate limiting to restrict excessive requests.

        Examples
        --------
        >>> api.create_api()

        """
        if self.api_type == "FastAPI":
            from fastapi import Request, HTTPException
            from pydantic import BaseModel

            class PredictRequest(BaseModel):
                input_data: Dict[str, Any]

            @self.app.post("/predict")
            async def predict(request: Request, data: PredictRequest):
                await self._rate_limit()
                logger.info(f"Received prediction request: {data}")
                try:
                    result = self.model.predict(data.input_data)
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    raise HTTPException(status_code=500, detail="Prediction failed")
                logger.info(f"Prediction result: {result}")
                return {"result": result}

        elif self.api_type == "Flask":
            from flask import request as flask_request, jsonify # , Response

            @self.app.route("/predict", methods=["POST"])
            def predict_flask():
                error_response = self._rate_limit_flask()
                if error_response:
                    return error_response
                data = flask_request.get_json()
                logger.info(f"Received prediction request: {data}")
                try:
                    result = self.model.predict(data["input_data"])
                except Exception as e:
                    logger.error(f"Error during prediction: {e}")
                    return jsonify({"error": "Prediction failed"}), 500
                logger.info(f"Prediction result: {result}")
                return jsonify({"result": result})

    @validate_params({
        'host': [str],
        'port': [int],
        'debug': [bool]
    })
    def serve_api(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = False
    ):
        """
        Launches the API server using FastAPI or Flask.

        Parameters
        ----------
        host : str, optional
            Host to serve the API on. Defaults to ``'0.0.0.0'``.
        port : int, optional
            Port to serve the API on. Defaults to ``8000``.
        debug : bool, optional
            Whether to run the server in debug mode (useful for Flask).
            Defaults to ``False``.

        Notes
        -----
        This method starts the API server using the specified host and port.

        Examples
        --------
        >>> api.serve_api(host='127.0.0.1', port=8080)

        """
        logger.info(
            f"Serving '{self.model_name}' model API on {host}:{port} using {self.api_type}."
        )
        if self.api_type == "FastAPI":
            import uvicorn
            uvicorn.run(self.app, host=host, port=port)
        elif self.api_type == "Flask":
            self.app.run(host=host, port=port, debug=debug)

    def health_check(self):
        """
        Provides a health endpoint to monitor the status of the deployed API.

        Notes
        -----
        This method defines the `/health` endpoint, which returns the
        current status and uptime of the API.

        Examples
        --------
        >>> api.health_check()

        """
        if self.api_type == "FastAPI":
            from fastapi import Request

            @self.app.get("/health")
            async def health(request: Request):
                uptime = time.time() - self.start_time
                logger.info(f"Health check requested, API uptime: {uptime:.2f} seconds.")
                return {"status": "healthy", "uptime": f"{uptime:.2f} seconds"}

        elif self.api_type == "Flask":
            from flask import jsonify

            @self.app.route("/health", methods=["GET"])
            def health_flask():
                uptime = time.time() - self.start_time
                logger.info(f"Health check requested, API uptime: {uptime:.2f} seconds.")
                return jsonify({"status": "healthy", "uptime": f"{uptime:.2f} seconds"})

    @validate_params({
        'model_version': [int]
    })
    def version_control(self, model_version: int):
        """
        Integrates version control, allowing multiple model versions to be
        served.

        Parameters
        ----------
        model_version : int
            The version of the model being served.

        Notes
        -----
        This method defines the `/model_version` endpoint, which returns
        the version of the model currently being served.

        Examples
        --------
        >>> api.version_control(model_version=1)

        """
        if self.api_type == "FastAPI":
            @self.app.get("/model_version")
            async def model_version_endpoint():
                logger.info(f"Serving model version: {model_version}")
                return {"version": model_version}

        elif self.api_type == "Flask":
            from flask import jsonify

            @self.app.route("/model_version", methods=["GET"])
            def model_version_flask():
                logger.info(f"Serving model version: {model_version}")
                return jsonify({"version": model_version})

    async def _rate_limit(self):
        """
        Implements rate limiting for FastAPI to restrict the number of
        requests.

        Raises
        ------
        HTTPException
            If the rate limit is exceeded.

        """
        from fastapi import HTTPException

        self.request_count += 1
        if self.request_count > self.max_requests:
            logger.warning("Rate limit exceeded. Please try again later.")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        logger.info(f"Request count: {self.request_count}/{self.max_requests}")

    def _rate_limit_flask(self):
        """
        Implements rate limiting for Flask to restrict the number of
        requests.

        Returns
        -------
        response : Flask Response or None
            Returns a Flask response if rate limit is exceeded, otherwise
            returns ``None``.

        """
        from flask import jsonify, make_response

        self.request_count += 1
        if self.request_count > self.max_requests:
            logger.warning("Rate limit exceeded. Please try again later.")
            response = make_response(
                jsonify({"error": "Rate limit exceeded. Please try again later."}),
                429
            )
            return response
        logger.info(f"Request count: {self.request_count}/{self.max_requests}")
        return None

    def shutdown(self):
        """
        Gracefully shuts down the API server, ensuring all requests are
        processed before stopping.

        Notes
        -----
        This method sets up signal handlers or shutdown events to gracefully
        stop the API server.

        Examples
        --------
        >>> api.shutdown()

        """
        logger.info(f"Setting up graceful shutdown for {self.api_type} API server.")

        if self.api_type == "FastAPI":
            @self.app.on_event("shutdown")
            async def fastapi_shutdown():
                """
                FastAPI-specific shutdown event handler.
                """
                logger.info("FastAPI server is shutting down...")
                await self._cleanup_resources()
                logger.info("FastAPI server has successfully shut down.")

        elif self.api_type == "Flask":
            import signal

            def signal_handler(sig, frame):
                """
                Flask-specific signal handler for graceful shutdown.
                """
                logger.info("Flask server is shutting down...")
                self._cleanup_resources_sync()
                logger.info("Flask server has successfully shut down.")
                exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    async def _cleanup_resources(self):
        """
        Asynchronous cleanup of resources (e.g., closing databases) for
        FastAPI.

        Notes
        -----
        Perform any necessary asynchronous cleanup tasks here.

        """
        import asyncio

        logger.info("Performing asynchronous resource cleanup for FastAPI...")
        await asyncio.sleep(1)  # Simulate asynchronous resource cleanup.
        logger.info("Asynchronous resource cleanup complete.")

    def _cleanup_resources_sync(self):
        """
        Synchronous cleanup of resources (e.g., closing databases) for
        Flask.

        Notes
        -----
        Perform any necessary synchronous cleanup tasks here.

        """
        logger.info("Performing synchronous resource cleanup for Flask...")
        time.sleep(1)  # Simulate synchronous resource cleanup.
        logger.info("Synchronous resource cleanup complete.")


class CloudDeployment(BaseLearner):
    """
    Manages cloud deployment for models across AWS SageMaker, GCP AI Platform,
    and Azure Machine Learning. Supports multi-cloud deployment and continuous
    deployment.

    Parameters
    ----------
    model : object
        The machine learning model to deploy.
    platform : {'aws', 'gcp', 'azure'}
        Cloud platform for deployment.
    model_name : str
        Name of the model.

    Attributes
    ----------
    model : object
        The machine learning model to deploy.
    platform : str
        Cloud platform for deployment.
    model_name : str
        Name of the model.

    Notes
    -----
    The ``CloudDeployment`` class provides methods to deploy machine learning
    models to different cloud platforms, including AWS SageMaker, Google Cloud
    AI Platform, and Azure Machine Learning. It supports multi-cloud deployment
    and continuous deployment strategies.

    Examples
    --------
    >>> from gofast.mlops.deployment import CloudDeployment
    >>> model = ...  # Your machine learning model
    >>> deployer = CloudDeployment(model, platform='aws', model_name='my_model')
    >>> config = {
    ...     'model_data': 's3://my-bucket/model.tar.gz',
    ...     'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole'
    ... }
    >>> deployer.deploy_to_aws(config)

    See Also
    --------
    boto3.client : AWS SDK for Python to interact with AWS services.
    google.cloud.aiplatform : Google Cloud AI Platform SDK.
    azureml.core : Azure Machine Learning SDK.

    References
    ----------
    .. [1] AWS SageMaker Documentation: https://docs.aws.amazon.com/sagemaker/
    .. [2] Google Cloud AI Platform Documentation: https://cloud.google.com/ai-platform
    .. [3] Azure Machine Learning Documentation: https://docs.microsoft.com/en-us/azure/machine-learning/

    """

    @validate_params({
        'model': [object],
        'platform': [StrOptions({'aws', 'gcp', 'azure'})],
        'model_name': [str],
    })
    def __init__(self, model: Any, platform: str, model_name: str):
        """
        Initializes the ``CloudDeployment`` class.

        Parameters
        ----------
        model : object
            The model to deploy.
        platform : {'aws', 'gcp', 'azure'}
            Cloud platform for deployment.
        model_name : str
            Name of the model.

        Raises
        ------
        ValueError
            If an unsupported platform is specified.

        Examples
        --------
        >>> deployer = CloudDeployment(model, platform='gcp', model_name='my_model')

        """
        self.model = model
        self.model_name = model_name

        self.platform = parameter_validator(
            "platform", target_strs= ["aws", "gcp", "azure"], 
            error_msg= ( 
                f"Unsupported platform: {platform}. Supported platforms"
                " are 'aws', 'gcp', and 'azure'."))(platform)
 
    @ensure_pkgs(
        "boto3",
        extra=EXTRA_MSG.format(pkg="boto3"),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    @validate_params({
        'config': [dict],
    })
    def deploy_to_aws(self, config: Dict[str, Any]):
        """
        Deploys the model to AWS SageMaker.

        Parameters
        ----------
        config : dict
            Configuration details for AWS deployment, including:
            - ``model_data`` (str): S3 URI where the model artifacts are stored.
            - ``instance_type`` (str, optional): Type of EC2 instance for deployment.
              Defaults to ``'ml.m5.large'``.
            - ``role_arn`` (str): AWS IAM role ARN with SageMaker permissions.

        Notes
        -----
        This method uses the AWS SDK for Python (:mod:`boto3`) to create a model
        in SageMaker, configure an endpoint, and deploy the model to the endpoint.

        Examples
        --------
        >>> config = {
        ...     'model_data': 's3://my-bucket/model.tar.gz',
        ...     'instance_type': 'ml.m5.large',
        ...     'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole'
        ... }
        >>> deployer.deploy_to_aws(config)

        Raises
        ------
        Exception
            If deployment to AWS SageMaker fails.

        See Also
        --------
        boto3.client : AWS SDK for Python.

        """
        logger.info(f"Deploying '{self.model_name}' to AWS SageMaker.")

        # Ensure boto3 is imported
        import boto3

        # Initialize SageMaker client
        sagemaker_client = boto3.client('sagemaker')

        try:
            model_data = config['model_data']
            instance_type = config.get('instance_type', 'ml.m5.large')
            role_arn = config['role_arn']

            response = sagemaker_client.create_model(
                ModelName=self.model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.6.0-cpu-py3',
                    'ModelDataUrl': model_data
                },
                ExecutionRoleArn=role_arn
            )
            logger.info(f"Model '{self.model_name}' created on SageMaker: {response}")

            # Deploy endpoint configuration
            endpoint_config_name = f"{self.model_name}-endpoint-config"
            sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': self.model_name,
                        'InstanceType': instance_type,
                        'InitialInstanceCount': 1
                    }
                ]
            )
            logger.info(f"Endpoint configuration '{endpoint_config_name}' created for '{self.model_name}'.")

            # Deploy endpoint
            endpoint_name = f"{self.model_name}-endpoint"
            sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Model '{self.model_name}' deployed at SageMaker endpoint '{endpoint_name}'.")

        except Exception as e:
            logger.error(f"Failed to deploy '{self.model_name}' to AWS SageMaker: {e}")
            raise

    @ensure_pkgs(
        "google",
        extra=EXTRA_MSG.format(pkg='google-cloud-aiplatform'), 
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA, 
        dist_name="google-cloud-aiplatform", 
        infer_dist_name=True
    )
    @validate_params({
        'config': [dict],
    })
    def deploy_to_gcp(self, config: Dict[str, Any]):
        """
        Deploys the model to Google Cloud AI Platform.

        Parameters
        ----------
        config : dict
            Configuration details for GCP deployment, including:
            - ``project`` (str): GCP project ID.
            - ``location`` (str, optional): GCP region. Defaults to ``'us-central1'``.
            - ``model_path`` (str): GCS URI where the model artifacts are stored.
            - ``machine_type`` (str, optional): Type of machine for deployment.
              Defaults to ``'n1-standard-4'``.
            - ``min_replica_count`` (int, optional): Minimum number of replicas.
              Defaults to ``1``.
            - ``max_replica_count`` (int, optional): Maximum number of replicas.
              Defaults to ``3``.

        Notes
        -----
        This method uses the Google Cloud AI Platform SDK to upload the model
        and deploy it as an endpoint.

        Examples
        --------
        >>> config = {
        ...     'project': 'my-gcp-project',
        ...     'location': 'us-central1',
        ...     'model_path': 'gs://my-bucket/model/',
        ...     'machine_type': 'n1-standard-4',
        ...     'min_replica_count': 1,
        ...     'max_replica_count': 3
        ... }
        >>> deployer.deploy_to_gcp(config)

        Raises
        ------
        Exception
            If deployment to GCP AI Platform fails.

        See Also
        --------
        google.cloud.aiplatform : Google Cloud AI Platform SDK.

        """
        logger.info(f"Deploying '{self.model_name}' to GCP AI Platform.")

        # Ensure google.cloud.aiplatform is imported
        from google.cloud import aiplatform as gcp_ai

        try:
            project = config['project']
            location = config.get('location', 'us-central1')
            model_path = config['model_path']

            gcp_ai.init(project=project, location=location)

            # Upload model to GCP
            model = gcp_ai.Model.upload(
                display_name=self.model_name,
                artifact_uri=model_path
            )
            logger.info(f"Model '{self.model_name}' uploaded to GCP AI Platform.")

            # Deploy the model as an endpoint
            endpoint = model.deploy(
                machine_type=config.get('machine_type', 'n1-standard-4'),
                min_replica_count=config.get('min_replica_count', 1),
                max_replica_count=config.get('max_replica_count', 3)
            )
            logger.info(f"Model '{self.model_name}' deployed at GCP endpoint '{endpoint.resource_name}'.")

        except Exception as e:
            logger.error(f"Failed to deploy '{self.model_name}' to GCP AI Platform: {e}")
            raise

    @ensure_pkgs(
        "azureml",
        extra=EXTRA_MSG.format(pkg='azureml-core'), 
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA, 
        dist_name="azureml-core", 
        infer_dist_name=True
    )
    @validate_params({
        'config': [dict],
    })
    def deploy_to_azure(self, config: Dict[str, Any]):
        """
        Deploys the model to Azure Machine Learning.

        Parameters
        ----------
        config : dict
            Configuration details for Azure ML deployment, including:
            - ``workspace_config`` (str): Path to Azure ML workspace config file.
            - ``model_path`` (str): Local path to the model file or directory.
            - ``aks_cluster_name`` (str): Name of the AKS cluster for deployment.
            - ``inference_config`` (:class:`azureml.core.model.InferenceConfig`):
                Inference configuration.
            - ``deployment_config`` 
            (:class:`azureml.core.webservice.AksWebserviceDeploymentConfiguration`): 
                Deployment config.

        Notes
        -----
        This method uses the Azure Machine Learning SDK to register the model
        and deploy it to an AKS cluster.

        Examples
        --------
        >>> from azureml.core.model import InferenceConfig
        >>> from azureml.core.webservice import AksWebservice, AksWebserviceDeploymentConfiguration
        >>> inference_config = InferenceConfig(entry_script='score.py', environment=my_env)
        >>> deployment_config = AksWebserviceDeploymentConfiguration(cpu_cores=1, memory_gb=1)
        >>> config = {
        ...     'workspace_config': 'config.json',
        ...     'model_path': 'model/',
        ...     'aks_cluster_name': 'my-aks-cluster',
        ...     'inference_config': inference_config,
        ...     'deployment_config': deployment_config
        ... }
        >>> deployer.deploy_to_azure(config)

        Raises
        ------
        Exception
            If deployment to Azure Machine Learning fails.

        See Also
        --------
        azureml.core : Azure Machine Learning SDK.

        """
        logger.info(f"Deploying '{self.model_name}' to Azure Machine Learning.")

        # Ensure azureml.core is imported
        from azureml.core import Workspace, Model
        from azureml.core.compute import AksCompute
        from azureml.core.model import InferenceConfig # noqa
        from azureml.core.webservice import AksWebservice # noqa

        try:
            workspace_config = config['workspace_config']
            model_path = config['model_path']
            aks_cluster_name = config['aks_cluster_name']
            inference_config = config['inference_config']
            deployment_config = config['deployment_config']

            # Load workspace configuration
            ws = Workspace.from_config(path=workspace_config)

            # Register the model
            model = Model.register(
                workspace=ws,
                model_path=model_path,
                model_name=self.model_name
            )
            logger.info(f"Model '{self.model_name}' registered in Azure ML Workspace.")

            # Get the AKS cluster
            deployment_target = AksCompute(workspace=ws, name=aks_cluster_name)

            # Deploy the model
            service = Model.deploy(
                workspace=ws,
                name=self.model_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=deployment_config,
                deployment_target=deployment_target,
                overwrite=True
            )
            service.wait_for_deployment(show_output=True)
            logger.info(f"Model '{self.model_name}' deployed to Azure at {service.scoring_uri}.")

        except Exception as e:
            logger.error(f"Failed to deploy '{self.model_name}' to Azure ML: {e}")
            raise

    @validate_params({
        'pipeline_config': [dict],
    })
    def continuous_deployment(self, pipeline_config: Dict[str, Any]):
        """
        Supports continuous deployment of models, automatically updating the
        model in the cloud.

        Parameters
        ----------
        pipeline_config : dict
            Configuration for CI/CD pipeline, including:
            - ``platform`` (str): CI/CD platform ('jenkins', 'github', etc.).
            - Other platform-specific configurations.

        Notes
        -----
        This method sets up continuous deployment pipelines using CI/CD tools
        such as Jenkins or GitHub Actions. The implementation details would
        depend on the specific CI/CD platform.

        Examples
        --------
        >>> pipeline_config = {
        ...     'platform': 'github',
        ...     'repo': 'https://github.com/user/repo',
        ...     # Additional configurations
        ... }
        >>> deployer.continuous_deployment(pipeline_config)

        Raises
        ------
        Exception
            If setting up continuous deployment fails.

        """
        logger.info(f"Setting up continuous deployment for '{self.model_name}'.")

        try:
            platform = pipeline_config['platform']

            if platform == 'jenkins':
                logger.info("Setting up Jenkins CI/CD pipeline.")
                # Logic to configure and trigger Jenkins pipeline
                pass  # Placeholder for actual implementation
            elif platform == 'github':
                logger.info("Setting up GitHub Actions CI/CD pipeline.")
                # Logic to set up GitHub Actions pipeline
                pass  # Placeholder for actual implementation
            else:
                logger.error(f"Unsupported CI/CD platform: {platform}")
                raise ValueError(f"Unsupported CI/CD platform: {platform}")

        except Exception as e:
            logger.error(f"Failed to set up continuous deployment for '{self.model_name}': {e}")
            raise

class ABTesting(BaseLearner):
    """
    Manages A/B testing for models in production, enabling traffic routing
    between different versions. Supports dynamic traffic split adjustments,
    performance evaluation, and rollback mechanisms.

    Parameters
    ----------
    model_v1 : object
        The first model to be evaluated in A/B testing.
    model_v2 : object
        The second model to be evaluated in A/B testing.
    split_ratio : float, optional
        Initial traffic split between `model_v1` and `model_v2`. Represents
        the probability of routing a request to `model_v1`. Defaults to
        ``0.5``.
    min_split_ratio : float, optional
        Minimum allowable traffic ratio for any model. Ensures that no model
        receives less than this fraction of traffic. Defaults to ``0.1``.
    max_split_ratio : float, optional
        Maximum allowable traffic ratio for any model. Ensures that no model
        receives more than this fraction of traffic. Defaults to ``0.9``.
    performance_threshold : float, optional
        Minimum performance difference required to adjust traffic split.
        If the absolute difference in performance between the models is
        greater than this threshold, the traffic split is adjusted.
        Defaults to ``0.05``.
    traffic_increment : float, optional
        Increment by which traffic is adjusted based on performance.
        Adjusts the `split_ratio` by this amount towards the better
        performing model. Defaults to ``0.1``.
    graceful_degradation : bool, optional
        If ``True``, prevents abrupt changes to traffic when both models
        perform poorly by resetting the `split_ratio` to ``0.5``.
        Defaults to ``True``.

    Attributes
    ----------
    model_v1 : object
        The first model in the A/B test.
    model_v2 : object
        The second model in the A/B test.
    split_ratio : float
        Current traffic split ratio for `model_v1`.
    min_split_ratio : float
        Minimum allowable traffic ratio.
    max_split_ratio : float
        Maximum allowable traffic ratio.
    performance_threshold : float
        Threshold for performance difference to adjust traffic.
    traffic_increment : float
        Amount by which to adjust traffic split.
    graceful_degradation : bool
        Flag to enable graceful degradation.

    Notes
    -----
    The A/B testing process involves routing a fraction of incoming
    traffic to two different models and adjusting the traffic split
    based on their performance. The traffic split ratio for `model_v1`
    is represented as :math:`p`, and the probability of routing a
    request to `model_v1` is :math:`P(\text{route to model\_v1}) = p`.

    The traffic split is adjusted based on performance metrics. If the
    performance difference exceeds the `performance_threshold`, the
    `split_ratio` is adjusted towards the better-performing model by
    `traffic_increment`.

    Examples
    --------
    >>> from gofast.mlops.deployment import ABTesting
    >>> model_v1 = ...  # Your first model
    >>> model_v2 = ...  # Your second model
    >>> ab_test = ABTesting(model_v1, model_v2, split_ratio=0.6)
    >>> # Route a request
    >>> response = ab_test.route_traffic(request_data)
    >>> # Evaluate performance
    >>> performance_metrics = {'model_v1': 0.85, 'model_v2': 0.78}
    >>> ab_test.evaluate_performance(performance_metrics)
    >>> # Adjusted split ratio
    >>> print(ab_test.split_ratio)
    0.7

    See Also
    --------
    ModelDeployment : Class for deploying models in production.

    References
    ----------
    .. [1] Kohavi, R., Longbotham, R., Sommerfield, D., & Henne, R. M. (2009).
       "Controlled experiments on the web: survey and practical guide."
       *Data Mining and Knowledge Discovery*, 18(1), 140-181.

    """

    @validate_params({
        'model_v1': [object],
        'model_v2': [object],
        'split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'min_split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'max_split_ratio': [Interval(Real, 0.0, 1.0, closed='both')],
        'performance_threshold': [Interval(Real, 0.0, 1.0, closed='both')],
        'traffic_increment': [Interval(Real, 0.0, 1.0, closed='both')],
        'graceful_degradation': [bool]
    })
    def __init__(
        self,
        model_v1: Any,
        model_v2: Any,
        split_ratio: float = 0.5,
        min_split_ratio: float = 0.1,
        max_split_ratio: float = 0.9,
        performance_threshold: float = 0.05,
        traffic_increment: float = 0.1,
        graceful_degradation: bool = True
    ):
        """
        Initializes the ``ABTesting`` class.

        Parameters
        ----------
        model_v1 : object
            First model to be evaluated in A/B testing.
        model_v2 : object
            Second model to be evaluated in A/B testing.
        split_ratio : float, optional
            Initial traffic split between `model_v1` and `model_v2`.
            Defaults to ``0.5``.
        min_split_ratio : float, optional
            Minimum allowable traffic ratio for any model. Defaults to
            ``0.1``.
        max_split_ratio : float, optional
            Maximum allowable traffic ratio for any model. Defaults to
            ``0.9``.
        performance_threshold : float, optional
            Minimum performance difference required to adjust traffic.
            Defaults to ``0.05``.
        traffic_increment : float, optional
            Increment by which traffic is adjusted based on performance.
            Defaults to ``0.1``.
        graceful_degradation : bool, optional
            If ``True``, prevents abrupt changes to traffic when both
            models perform poorly. Defaults to ``True``.

        Raises
        ------
        ValueError
            If `min_split_ratio` is greater than `max_split_ratio` or
            if `split_ratio` is not between `min_split_ratio` and
            `max_split_ratio`.

        Examples
        --------
        >>> ab_test = ABTesting(model_v1, model_v2, split_ratio=0.6)

        """
        if min_split_ratio > max_split_ratio:
            raise ValueError("min_split_ratio cannot be greater than max_split_ratio.")
        if not (min_split_ratio <= split_ratio <= max_split_ratio):
            raise ValueError("split_ratio must be between min_split_ratio and max_split_ratio.")
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.split_ratio = split_ratio
        self.min_split_ratio = min_split_ratio
        self.max_split_ratio = max_split_ratio
        self.performance_threshold = performance_threshold
        self.traffic_increment = traffic_increment
        self.graceful_degradation = graceful_degradation

    @validate_params({
        'request': [dict]
    })
    def route_traffic(self, request: Dict) -> Any:
        """
        Routes traffic between `model_v1` and `model_v2` based on the
        current `split_ratio`.

        Parameters
        ----------
        request : dict
            Incoming request data that the model will process.

        Returns
        -------
        response : Any
            Model prediction response.

        Notes
        -----
        The request is routed to `model_v1` with probability equal to
        `split_ratio` and to `model_v2` with probability
        :math:`1 - \text{split_ratio}`.

        Examples
        --------
        >>> response = ab_test.route_traffic(request_data)

        """
        if random.random() < self.split_ratio:
            logger.info("Routing to model version 1.")
            return self.model_v1.predict(request)
        else:
            logger.info("Routing to model version 2.")
            return self.model_v2.predict(request)

    @validate_params({
        'performance_metrics': [dict]
    })
    def evaluate_performance(self, performance_metrics: Dict[str, float]):
        """
        Adjusts the traffic split between `model_v1` and `model_v2`
        based on performance metrics.

        Parameters
        ----------
        performance_metrics : dict
            Dictionary containing the performance metrics for each model.
            Example: ``{"model_v1": 0.85, "model_v2": 0.78}``.

        Notes
        -----
        If the absolute difference in performance exceeds
        `performance_threshold`, the `split_ratio` is adjusted towards
        the better-performing model by `traffic_increment`. The new
        `split_ratio` is constrained between `min_split_ratio` and
        `max_split_ratio`.

        The performance adjustment can be mathematically represented as:

        .. math::

            \text{split\_ratio} = \min\left(
                \max\left(
                    \text{split\_ratio} \pm \text{traffic\_increment},
                    \text{min\_split\_ratio}
                \right),
                \text{max\_split\_ratio}
            )

        Where the sign of :math:`\pm` depends on which model is performing
        better.

        Examples
        --------
        >>> performance_metrics = {'model_v1': 0.90, 'model_v2': 0.80}
        >>> ab_test.evaluate_performance(performance_metrics)

        """
        model_v1_performance = performance_metrics.get("model_v1")
        model_v2_performance = performance_metrics.get("model_v2")

        if model_v1_performance is None or model_v2_performance is None:
            raise ValueError("Performance metrics must include 'model_v1' and 'model_v2'.")

        # Compare performance based on the defined threshold
        performance_diff = model_v1_performance - model_v2_performance

        if abs(performance_diff) > self.performance_threshold:
            if model_v1_performance > model_v2_performance:
                logger.info("Model_v1 performs better. Adjusting split ratio towards model_v1.")
                self.split_ratio = min(
                    self.split_ratio + self.traffic_increment,
                    self.max_split_ratio
                )
            else:
                logger.info("Model_v2 performs better. Adjusting split ratio towards model_v2.")
                self.split_ratio = max(
                    self.split_ratio - self.traffic_increment,
                    self.min_split_ratio
                )
        else:
            logger.info(f"Performance difference within threshold ({self.performance_threshold}). No traffic adjustment.")

        logger.info(f"Adjusted traffic split ratio: {self.split_ratio}")

        # Apply graceful degradation if both models are underperforming
        if self.graceful_degradation:
            self._apply_graceful_degradation(model_v1_performance, model_v2_performance)

    def _apply_graceful_degradation(self, performance_v1: float, performance_v2: float):
        """
        Applies graceful degradation, preventing abrupt traffic shifts when
        both models perform poorly.

        Parameters
        ----------
        performance_v1 : float
            Performance score of `model_v1`.
        performance_v2 : float
            Performance score of `model_v2`.

        Notes
        -----
        If the average performance of both models is below a certain
        threshold (e.g., ``0.5``), the `split_ratio` is reset to ``0.5``
        to stabilize the system.

        """
        average_performance = (performance_v1 + performance_v2) / 2
        if average_performance < 0.5:  # Threshold for poor performance
            logger.warning("Both models are underperforming. Applying graceful degradation.")
            self.split_ratio = 0.5  # Reset to equal traffic split
            logger.info("Reset traffic split ratio to 0.5 due to underperformance.")

    def rollback(self):
        """
        Rolls back to the better-performing model based on the current
        `split_ratio`.

        Notes
        -----
        If the `split_ratio` is greater than ``0.8``, all traffic is routed
        to `model_v1`. If the `split_ratio` is less than ``0.2``, all
        traffic is routed to `model_v2`. This function can be used to
        quickly switch to the dominant model in case of significant
        performance issues.

        Examples
        --------
        >>> ab_test.rollback()
        >>> # All traffic is now routed to the dominant model.

        """
        if self.split_ratio > 0.8:
            logger.info("Rolling back to model_v1 as the dominant model.")
            self.split_ratio = 1.0  # Route all traffic to model_v1
        elif self.split_ratio < 0.2:
            logger.info("Rolling back to model_v2 as the dominant model.")
            self.split_ratio = 0.0  # Route all traffic to model_v2
        else:
            logger.info("No significant performance difference for rollback.")


if __name__=='__main__': 
    my_model= ...
    # AWS Deployment
    aws_config = {
        "model_data": "s3://my-bucket/model.tar.gz",
        "instance_type": "ml.m5.large",
        "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole"
    }
    cloud_manager = CloudDeployment(model=my_model, platform="aws", model_name="my_model")
    cloud_manager.deploy_to_aws(config=aws_config)
    
    # GCP Deployment
    gcp_config = {
        "project": "my-gcp-project",
        "model_path": "gs://my-bucket/models/model.tar.gz",
        "machine_type": "n1-standard-4",
        "min_replica_count": 1,
        "max_replica_count": 3
    }
    cloud_manager = CloudDeployment(model=my_model, platform="gcp", model_name="my_model")
    cloud_manager.deploy_to_gcp(config=gcp_config)
    
    # Azure Deployment
    inference_config = ...# Predefined InferenceConfig object
    deployment_config= ... # Predefined AksWebserviceDeploymentConfiguration object
    config = {
    "model_path": "azureml-models/model.pkl",
    "aks_cluster_name": "my-aks-cluster",
    "inference_config": inference_config,  
    "deployment_config": deployment_config 
}

    azure_config = {
        "model_path": "azureml-models/model.pkl",
        "aks_cluster_name": "my-aks-cluster",
        "inference_config": inference_config,
        "deployment_config": deployment_config
    }
    cloud_manager = CloudDeployment(model=my_model, platform="azure", model_name="my_model")
    cloud_manager.deploy_to_azure(config=azure_config)
    
    # Continuous Deployment Setup
    pipeline_config = {
        "platform": "github",
        "github_repo": "https://github.com/myrepo/model-deploy",
        "github_actions_workflow": ".github/workflows/deploy.yaml"
    }
    cloud_manager.continuous_deployment(pipeline_config)

    # Initialize the A/B testing manager with model_v1, model_v2, and a starting traffic split of 50/50
    my_model_v1= ...
    my_model_v2=...
    ab_tester = ABTesting(model_v1=my_model_v1, model_v2=my_model_v2, split_ratio=0.5,
                          min_split_ratio=0.1, max_split_ratio=0.9, performance_threshold=0.05, 
                          traffic_increment=0.1)
    
    # Simulate routing traffic for a new request
    response = ab_tester.route_traffic(request={"input_data": "example"})
    
    # Evaluate performance metrics and adjust traffic split accordingly
    performance_metrics = {"model_v1": 0.82, "model_v2": 0.78}
    ab_tester.evaluate_performance(performance_metrics)
    
    # Apply rollback if one model significantly outperforms the other
    ab_tester.rollback()
