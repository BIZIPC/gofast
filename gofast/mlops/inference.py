# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Streamline the inference process, ensuring that models perform efficiently 
and reliably in real-time or batch inference scenarios.
"""
import os
from numbers import Integral, Real
import random
import time
import multiprocessing
import threading
import json
import pickle
import concurrent.futures

from typing import Any, List, Dict, Callable, Optional

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from .._gofastlog import gofastlog 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..decorators import RunReturn, smartFitRun
from ..tools.depsutils import ensure_pkg
from ..tools.validator import check_is_runned 

from ._base import BaseInference

logger=gofastlog.get_gofast_logger(__name__)

__all__= [
    "BatchInference", "StreamingInference", "MultiModelServing",
    "InferenceParallelizer", "InferenceCacheManager"
    ]

@smartFitRun 
class BatchInference(BaseInference):
    """
    Optimizes batch inference for large datasets by managing batch sizes,
    parallelism, memory usage, and GPU acceleration.

    Parameters
    ----------
    model : object
        The machine learning model to perform batch inference on. The model
        should implement a `predict` method or be callable.
    batch_size : int, default=32
        Number of inputs to process in a single batch.
    max_workers : int, default=4
        Number of parallel workers for batch inference.
    timeout : float or None, default=None
        Timeout for each inference task in seconds. If `None`, no timeout
        is applied.
    optimize_memory : bool, default=True
        Whether to enable memory optimizations during inference.
    gpu_enabled : bool, default=False
        Enables GPU acceleration if `True`.
    enable_padding : bool, default=False
        Enables padding of batches to match `batch_size`.
    log_level : {'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'}, default='INFO'
        Logging verbosity level.

    Notes
    -----
    The `BatchInference` class facilitates efficient batch inference by
    leveraging parallelism and GPU acceleration. It splits the data into
    batches and processes them using a thread pool executor.

    If `gpu_enabled` is set to `True`, the model and data are moved to the
    GPU for inference. The class checks for the availability of CUDA-enabled
    GPUs and falls back to CPU inference if none are available.

    Mathematically, for a dataset with :math:`N` samples and a `batch_size`
    of :math:`B`, the number of batches :math:`K` is calculated as:

    .. math::

        K = \left\lceil \dfrac{N}{B} \right\rceil

    where :math:`\lceil \cdot \rceil` denotes the ceiling function.

    Examples
    --------
    >>> from gofast.mlops.inference import BatchInference
    >>> model = MyModel()  # A model should implement a predict method
    >>> data = [{'input': x} for x in range(1000)]
    >>> batch_inference = BatchInference(
    ...     model, batch_size=64, gpu_enabled=True
    ... )
    >>> results = batch_inference.run(data)
    >>> print(results)

    See Also
    --------
    concurrent.futures.ThreadPoolExecutor : For parallel execution.
    torch.cuda.is_available : To check GPU availability.

    References
    ----------
    .. [1] Dean, J., & Ghemawat, S. (2008). "MapReduce: Simplified Data
       Processing on Large Clusters." *Communications of the ACM*, 51(1),
       107-113.

    """

    def __init__(
        self,
        model: Any,
        batch_size: int = 32,
        max_workers: int = 4,
        timeout: Optional[float] = None,
        optimize_memory: bool = True,
        gpu_enabled: bool = False,
        enable_padding: bool = False,
        log_level: str = 'INFO'
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
            timeout=timeout,
            optimize_memory=optimize_memory,
            gpu_enabled=gpu_enabled,
            enable_padding=enable_padding,
            log_level=log_level
        )
        
    @RunReturn( attribute_name='results_') 
    def run(
        self,
        data: List[Dict],
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None,
        handle_errors: bool = True
    ) -> List[Any]:
        """
        Runs batch inference on the provided data with optional error handling.

        Parameters
        ----------
        data : list of dict
            List of input data for inference.
        preprocess_fn : callable or None, default=None
            Optional preprocessing function to apply to each batch.
        postprocess_fn : callable or None, default=None
            Optional postprocessing function to apply to model output.
        handle_errors : bool, default=True
            If `True`, handles errors and skips failed batches.

        Returns
        -------
        results_ : list
            The model's predictions for each batch.

        Notes
        -----
        The method splits the data into batches and uses a thread pool
        executor to process them in parallel. If `enable_padding` is set
        to `True`, the last batch is padded to match `batch_size`.

        Examples
        --------
        >>> results_ = batch_inference.run_batch_inference(
        ...     data,
        ...     preprocess_fn=preprocess,
        ...     postprocess_fn=postprocess
        ... )
        """
        logger.info(
            f"Running batch inference with batch size: {self.batch_size}, "
            f"max workers: {self.max_workers}, GPU enabled: {self.gpu_enabled}."
        )
        batches = [
            data[i:i + self.batch_size]
            for i in range(0, len(data), self.batch_size)
        ]

        if self.enable_padding:
            logger.info(
                "Padding enabled. Padding the last batch to match batch size."
            )
            batches = self._pad_batches(batches)

        self.results_ = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_batch = {
                executor.submit(
                    self._infer_batch, batch, preprocess_fn, postprocess_fn
                ): batch for batch in batches
            }
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_result = future.result(timeout=self.timeout)
                    self.results_.extend(batch_result)
                except Exception as e:
                    if handle_errors:
                        logger.error(
                            f"Error during batch inference: {e}. "
                            f"Skipping batch."
                        )
                    else:
                        raise e

    def _infer_batch(
        self,
        batch: List[Dict],
        preprocess_fn: Optional[Callable],
        postprocess_fn: Optional[Callable]
    ) -> List[Any]:
        """
        Performs inference on a single batch with optional pre- and
        post-processing.

        Parameters
        ----------
        batch : list of dict
            Input batch data for inference.
        preprocess_fn : callable or None
            Preprocessing function to apply to the batch.
        postprocess_fn : callable or None
            Postprocessing function to apply to the model output.

        Returns
        -------
        predictions : list
            The model's predictions for the batch.

        Raises
        ------
        ValueError
            If the model does not implement a `predict` method or is not
            callable.

        """
        logger.info(f"Processing batch of size {len(batch)}.")

        # Apply preprocessing if provided
        if preprocess_fn:
            batch = [preprocess_fn(item) for item in batch]

        # Inference step
        if self.gpu_enabled:
            logger.info("Using GPU for inference.")
            predictions = self._gpu_inference(batch)
        else:
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(batch)
            elif callable(self.model):
                predictions = self.model(batch)
            else:
                raise ValueError(
                    "Model must implement a `predict` method or be callable."
                )

        # Apply postprocessing if provided
        if postprocess_fn:
            predictions = [postprocess_fn(pred) for pred in predictions]

        return predictions

    @ensure_pkg(
        "torch",
        extra="The 'torch' package is required for GPU inference.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _gpu_inference(self, batch: List[Dict]) -> List[Any]:
        """
        Performs GPU-accelerated inference using PyTorch with CUDA.

        Parameters
        ----------
        batch : list of dict
            Input batch data, typically as a list of dictionaries.

        Returns
        -------
        predictions : list
            Model predictions using GPU acceleration.

        Notes
        -----
        The method checks if CUDA is available and transfers the model
        and data to the GPU. If CUDA is not available, it falls back to
        CPU inference.

        Raises
        ------
        Exception
            If an error occurs during GPU inference.

        """
        logger.info("Performing GPU inference using PyTorch with CUDA.")
        
        import torch
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error(
                "CUDA is not available. Falling back to CPU inference."
            )
            if hasattr(self.model, 'predict'):
                return self.model.predict(batch)
            elif callable(self.model):
                return self.model(batch)
            else:
                raise ValueError(
                    "Model must implement a `predict` method or be callable."
                )

        # Convert batch to tensor and transfer to GPU
        logger.info("Transferring data to GPU.")
        gpu_batch = self._batch_to_tensor(batch).cuda()

        # Ensure the model is on GPU
        if not next(self.model.parameters()).is_cuda:
            logger.info("Transferring model to GPU.")
            self.model = self.model.cuda()

        # Set the model to evaluation mode
        self.model.eval()

        # Perform inference on the GPU
        with torch.no_grad():
            try:
                logger.info("Running inference on GPU.")
                predictions = self.model(gpu_batch)
            except Exception as e:
                logger.error(f"Error during GPU inference: {e}")
                raise e

        # Transfer results back to CPU and convert to list
        logger.info("Transferring predictions back to CPU.")
        predictions = predictions.cpu().detach().numpy()

        return predictions.tolist()
    

    def _batch_to_tensor(self, batch: List[Dict]) : 
        """
        Converts a batch of input dictionaries into a PyTorch tensor for
        GPU processing.

        Parameters
        ----------
        batch : list of dict
            Input batch data as a list of dictionaries.

        Returns
        -------
        tensor : torch.Tensor
            PyTorch tensor representation of the batch.

        Notes
        -----
        Assumes that each item in the batch is a dictionary with numerical
        values.

        Raises
        ------
        Exception
            If an error occurs during the conversion.

        """
        logger.info("Converting batch to PyTorch tensor.")
        
        import torch

        try:
            # Convert the list of dictionaries to a list of lists
            batch_values = [
                [float(value) for value in item.values()] for item in batch
            ]
            return torch.tensor(batch_values, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error while converting batch to tensor: {e}")
            raise e

    def _pad_batches(self, batches: List[List[Dict]]) -> List[List[Dict]]:
        """
        Pads the last batch to match the batch size if `enable_padding` is
        set to True.

        Parameters
        ----------
        batches : list of list of dict
            List of batches.

        Returns
        -------
        padded_batches : list of list of dict
            Padded batches.

        Notes
        -----
        The last batch is padded by repeating the last item until the batch
        size matches `batch_size`.

        """
        if len(batches[-1]) < self.batch_size:
            padding_size = self.batch_size - len(batches[-1])
            padded_batch = batches[-1] + [batches[-1][-1]] * padding_size
            batches[-1] = padded_batch
            logger.info(
                f"Last batch padded with {padding_size} additional items."
            )
        return batches
    
    
@smartFitRun 
class StreamingInference(BaseInference):
    """
    Manages real-time streaming inference with support for Kafka integration.

    Consumes input data from a Kafka topic, processes it using a machine
    learning model, and sends results back to another Kafka topic.

    Parameters
    ----------
    model : object
        The machine learning model to perform real-time inference. The model
        should implement a ``predict`` method or be callable.
    kafka_topic : str
        Kafka topic to consume data from.
    kafka_servers : list of str
        List of Kafka servers in the format ``host:port``.
    group_id : str
        Group ID for Kafka consumer_s.
    result_topic : str or None, default=None
        Kafka topic to send inference results to. If ``None``, defaults to
        ``kafka_topic + "_results"``.
    batch_size : int, default=1
        Number of messages to process in each batch.
    consumer__timeout_ms : int, default=1000
        Kafka consumer_ timeout in milliseconds.
    max_retries : int, default=3
        Maximum number of retries for sending results to Kafka.

    Notes
    -----
    The ``StreamingInference`` class facilitates real-time inference by
    consuming data from a Kafka topic, processing it using a provided model,
    and producing the results to another Kafka topic.

    The class handles batching of messages, error handling, and supports
    preprocessing and postprocessing functions.

    Examples
    --------
    >>> from gofast.mlops.inference import StreamingInference
    >>> def preprocess(data):
    ...     # Preprocessing logic
    ...     return data
    >>> def postprocess(prediction):
    ...     # Postprocessing logic
    ...     return prediction
    >>> model = MyModel()  # the model should implement a predict method
    >>> streaming_inference = StreamingInference(
    ...     model=model,
    ...     kafka_topic='input_topic',
    ...     kafka_servers=['localhost:9092'],
    ...     group_id='my_group'
    ... )
    >>> streaming_inference.run(
    ...     preprocess_fn=preprocess,
    ...     postprocess_fn=postprocess
    ... )
    >>> # To stop the streaming
    >>> streaming_inference.stop_streaming()
    >>> streaming_inference.shutdown()

    See Also
    --------
    kafka.KafkaConsumer : Kafka consumer_ class.
    kafka.KafkaProducer : Kafka producer class.

    References
    ----------
    .. [1] Apache Kafka Documentation: https://kafka.apache.org/documentation/

    """

    
    @validate_params({
        'kafka_topic': [str],
        'kafka_servers': [list],
        'group_id': [str],
        'result_topic': [str, None],
        'batch_size': [Interval(Integral, 1, None, closed='left')],
        'consumer_timeout_ms': [Interval(Integral, 1, None, closed='left')],
        'max_retries': [Interval(Integral, 1, None, closed='left')],
    })
    def __init__(
        self,
        model: Any,
        kafka_topic: str,
        kafka_servers: List[str],
        group_id: str,
        result_topic: Optional[str] = None,
        batch_size: int = 1,
        consumer_timeout_ms: int = 1000,
        max_retries: int = 3
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            max_workers=1,
        )
        
        self.model = model
        self.kafka_topic = kafka_topic
        self.kafka_servers = kafka_servers
        self.group_id = group_id
        self.result_topic = result_topic or f"{kafka_topic}_results"
        self.batch_size = batch_size
        self.consumer_timeout_ms = consumer_timeout_ms
        self.max_retries = max_retries


    @RunReturn 
    @ensure_pkg(
        "kafka",
        extra="The 'kafka' package is required for Kafka integration.",
        auto_install=INSTALL_DEPENDENCIES,  
        use_conda=USE_CONDA  
    )
    def run(
        self,
        preprocess_fn: Optional[Callable] = None,
        postprocess_fn: Optional[Callable] = None
    ):
        """
        Starts consuming data from Kafka for real-time inference.

        Parameters
        ----------
        preprocess_fn : callable or None, default=None
            Optional preprocessing function to apply to each message.
        postprocess_fn : callable or None, default=None
            Optional postprocessing function to apply to model output.

        Notes
        -----
        This method starts a separate thread to consume messages from the Kafka
        topic and process them in real-time.

        """
        from kafka import KafkaConsumer, KafkaProducer
        
        # Kafka consumer_ and producer
        self.consumer_ = KafkaConsumer(
            self.kafka_topic,
            bootstrap_servers=self.kafka_servers,
            group_id=self.group_id,
            consumer_timeout_ms=self.consumer__timeout_ms,
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        self.producer_ = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # For thread control
        self._running = threading.Event()
        self._running.set()
        self._consumer_thread = None
        
        logger.info(
            f"Starting streaming inference on Kafka topic: {self.kafka_topic}."
        )
        # Start the consumer_ thread for consuming and processing messages
        self._consumer_thread = threading.Thread(
            target=self._consume_and_infer,
            args=(preprocess_fn, postprocess_fn)
        )
        self._consumer_thread.start()
        
    def _consume_and_infer(
        self,
        preprocess_fn: Optional[Callable],
        postprocess_fn: Optional[Callable]
    ):
        """
        Consumes data from Kafka and runs inference in real-time, sending the
        results back to Kafka.

        Parameters
        ----------
        preprocess_fn : callable or None
            Preprocessing function for each message.
        postprocess_fn : callable or None
            Postprocessing function for model output.

        """
        batch = []

        # Continue processing while the thread is running
        while self._running.is_set():
            try:
                for message in self.consumer_:
                    if not self._running.is_set():
                        break

                    data = message.value
                    logger.info(f"Received message from Kafka: {data}")

                    # Preprocessing step
                    if preprocess_fn:
                        data = preprocess_fn(data)

                    batch.append(data)

                    # Once we have enough messages, process the batch
                    if len(batch) >= self.batch_size:
                        # Inference step
                        if hasattr(self.model, 'predict'):
                            predictions = self.model.predict(batch)
                        elif callable(self.model):
                            predictions = self.model(batch)
                        else:
                            raise ValueError(
                                "Model must implement a `predict` method or be callable."
                            )

                        # Postprocessing step
                        if postprocess_fn:
                            predictions = [
                                postprocess_fn(pred) for pred in predictions
                            ]

                        # Send the results to Kafka
                        self._send_to_kafka(predictions)
                        batch.clear()

            except Exception as e:
                logger.error(f"Error during consumption or inference: {e}")

        logger.info("Stopped consuming from Kafka.")

    def _send_to_kafka(self, results: List[Any]):
        """
        Sends inference results to Kafka producer.

        Parameters
        ----------
        results : list
            List of results to send to Kafka.

        """
        for result in results:
            retries = 0
            sent = False
            while not sent and retries < self.max_retries:
                try:
                    self.producer_.send(self.result_topic, value=result)
                    self.producer_.flush()
                    logger.info(f"Sent result to Kafka: {result}")
                    sent = True
                except Exception as e:
                    retries += 1
                    logger.error(
                        f"Error sending result to Kafka, retrying "
                        f"{retries}/{self.max_retries}: {e}"
                    )
                    if retries >= self.max_retries:
                        logger.error("Max retries reached. Dropping message.")
                        break

    def stop_streaming(self):
        """
        Stops the streaming inference process by signaling the consumer_ thread
        to stop.

        Notes
        -----
        This method sets an event to signal the consumer thread to stop and
        waits for the thread to join.

        """
        logger.info("Stopping the streaming inference process.")
        self._running.clear()
        if self._consumer__thread:
            self._consumer__thread.join()

    def shutdown(self):
        """
        Gracefully shuts down the Kafka consumer and producer, ensuring all
        resources are cleaned up.

        Notes
        -----
        This method stops the streaming process and closes the Kafka consumer_
        and producer.

        """
        logger.info("Shutting down Kafka consumer and producer.")
        self.stop_streaming()
        self.consumer_.close()
        self.producer_.close()
        
    def __gofast_is_runned__(self) -> bool:
        """
        Determines if the StreamingInference instance has been "runned" by
        checking the state of the `_running` event.

        Returns
        -------
        bool
            True if the streaming process has started (i.e., `_running` is
            set), otherwise False.
        """
        return self._running.is_set()
    
    
@smartFitRun 
class MultiModelServing(BaseClass):
    """
    Manages inference for multiple models, supporting dynamic routing,
    traffic balancing, model performance tracking, and failure handling.

    Parameters
    ----------
    models : dict of str to object
        Dictionary mapping model names to model objects. Each model must
        implement a ``predict`` method or be callable.
    traffic_split : dict of str to float, optional
        Traffic split ratios for each model. The keys are model names,
        and the values are the proportions of traffic to send to each
        model. The values should sum to 1.0. If ``None``, traffic is split
        equally among models.
    performance_metrics : dict of str to dict of str to float, optional
        Dictionary to monitor performance of each model. For example,
        ``{'model_v1': {'latency': 0.2, 'accuracy': 0.9}}``. This can be
        used to select models based on performance metrics like latency and
        accuracy.
    fallback_models : dict of str to list of str, optional
        Dictionary specifying fallback models for each primary model. For
        example, ``{'model_v1': ['model_v2', 'model_v3']}``.
    latency_threshold : float or None, optional
        Maximum acceptable latency (in seconds) before selecting another
        model based on performance. If ``None``, latency is not considered
        in model selection.
    error_handling : bool, default=True
        Enables error handling and retries when a model fails.
    retry_attempts : int, default=3
        Number of times to retry with backup models in case of failure.
    traffic_algorithm : callable or None, optional
        Custom function for traffic routing. Should accept ``models`` and
        ``traffic_split`` as arguments. If ``None``, uses weighted random
        selection based on ``traffic_split``.

    Notes
    -----
    The :class:`MultiModelServing` class facilitates the serving of
    multiple models in a production environment. It supports dynamic
    routing of inference requests based on traffic splits or real-time
    performance metrics such as latency and accuracy. It also provides
    mechanisms for failure handling and retries using fallback models.

    The selection of models can be based on traffic splitting ratios or
    performance metrics. If a latency threshold is provided via
    ``latency_threshold``, the model with the best performance below the
    threshold is selected.

    Traffic splitting can be done using a custom algorithm provided via
    ``traffic_algorithm`` or default to weighted random selection based on
    the ``traffic_split`` ratios.

    Examples
    --------
    >>> from gofast.mlops.inference import MultiModelServing
    >>> model_v1 = MyModelV1()
    >>> model_v2 = MyModelV2()
    >>> models = {'model_v1': model_v1, 'model_v2': model_v2}
    >>> traffic_split = {'model_v1': 0.7, 'model_v2': 0.3}
    >>> fallback_models = {'model_v1': ['model_v2']}
    >>> multi_model_serving = MultiModelServing(
    ...     models=models,
    ...     traffic_split=traffic_split,
    ...     fallback_models=fallback_models,
    ...     latency_threshold=0.5
    ... )
    >>> data = {'input': [1, 2, 3]}
    >>> prediction = multi_model_serving.predict(data)

    See Also
    --------
    BatchInference : For batch processing of inference requests.
    StreamingInference : For real-time streaming inference.

    References
    ----------
    .. [1] Seldon Core Documentation: https://docs.seldon.io/projects/seldon-core/en/latest/
    .. [2] "Architecting for Scale with Multi-Model Serving", Cloud Best Practices.

    """

    @validate_params({
        'models': [dict],
        'traffic_split': [dict, None],
        'performance_metrics': [dict, None],
        'fallback_models': [dict, None],
        'latency_threshold': [Interval(Real, 0, None, closed='left'), None],
        'error_handling': [bool],
        'retry_attempts': [Interval(Integral, 1, None, closed='left')],
        'traffic_algorithm': [callable, None],
    })
    def __init__(
        self,
        models: Dict[str, Any],
        traffic_split: Optional[Dict[str, float]] = None,
        performance_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        fallback_models: Optional[Dict[str, List[str]]] = None,
        latency_threshold: Optional[float] = None,
        error_handling: bool = True,
        retry_attempts: int = 3,
        traffic_algorithm: Optional[Callable] = None,
    ):
        
        self.models = models
        self.traffic_split = traffic_split or {
            model: 1 / len(models) for model in models
        }
        self.performance_metrics = performance_metrics or {
            model: {'latency': None, 'accuracy': None} for model in models
        }
        self.fallback_models = fallback_models or {model: [] for model in models}
        self.latency_threshold = latency_threshold
        self.error_handling = error_handling
        self.retry_attempts = retry_attempts
        self.traffic_algorithm = traffic_algorithm

    @RunReturn( attribute_name ="results_") 
    def run(self, data: Dict, model_name: Optional[str] = None) -> Any:
        """
        Routes the inference request to the appropriate model based on
        traffic split or specified model, with error handling, retry
        attempts, and performance-based model selection.

        Parameters
        ----------
        data : dict
            Input data for inference.
        model_name : str or None, optional
            Specific model to route the request to. If ``None``, a model is
            selected based on traffic split or performance metrics.

        Returns
        -------
        result : Any
            Model prediction result.

        Raises
        ------
        ValueError
            If the specified model name is not found.
        RuntimeError
            If all fallback models fail during retries.

        Notes
        -----
        If ``model_name`` is provided, the request is routed to the
        specified model. Otherwise, the model is selected based on
        performance metrics (if ``latency_threshold`` is set) or traffic
        splitting ratios.

        Examples
        --------
        >>> prediction = multi_model_serving.predict(data)

        """
        try:
            if model_name:
                model = self.models.get(model_name)
                if not model:
                    raise ValueError(f"Model '{model_name}' not found.")
                logger.info(f"Routing request to specified model '{model_name}'.")
            else:
                if self.latency_threshold is not None:
                    model = self._select_model_by_performance()
                else:
                    model = self._select_model_by_traffic()

            self.results_= self._run_inference(model, data)

        except Exception as e:
            logger.error(
                f"Error during inference with model "
                f"'{model_name or 'auto-selected'}': {e}"
            )
            if self.error_handling:
                self.results_= self._retry_inference(data, model_name or 'auto-selected')
            else:
                raise

    def _retry_inference(self, data: Dict, failed_model: str) -> Any:
        """
        Retries inference with fallback models if the initial model fails.

        Parameters
        ----------
        data : dict
            Input data for inference.
        failed_model : str
            The name of the model that failed.

        Returns
        -------
        result : Any
            Model prediction result from the fallback model.

        Raises
        ------
        RuntimeError
            If all fallback models fail during retries.

        Notes
        -----
        The method attempts to retry inference using fallback models
        specified in ``fallback_models`` for the failed model. It respects
        the ``retry_attempts`` limit.

        """
        fallback_models = self.fallback_models.get(failed_model, [])
        attempts = 0

        for backup_model_name in fallback_models:
            if attempts >= self.retry_attempts:
                logger.error(
                    f"Exceeded max retries ({self.retry_attempts}). "
                    f"No more fallback models available."
                )
                break

            try:
                backup_model = self.models.get(backup_model_name)
                if backup_model:
                    logger.info(
                        f"Retrying inference with fallback model '{backup_model_name}'."
                    )
                    result = self._run_inference(backup_model, data)
                    return result
                else:
                    logger.warning(f"Fallback model '{backup_model_name}' not found.")
            except Exception as e:
                logger.error(
                    f"Error during inference with fallback model "
                    f"'{backup_model_name}': {e}"
                )
            attempts += 1

        raise RuntimeError("All fallback models failed.")

    def _run_inference(self, model: Any, data: Dict) -> Any:
        """
        Executes inference on the selected model and records performance
        metrics.

        Parameters
        ----------
        model : object
            The selected model to perform inference. Must implement a
            ``predict`` method or be callable.
        data : dict
            Input data for inference.

        Returns
        -------
        result : Any
            The model's prediction result.

        Raises
        ------
        Exception
            If the model's inference fails.

        Notes
        -----
        The method measures the latency of the inference and updates the
        ``performance_metrics`` dictionary.

        """
        start_time = time.time()
        model_name = next(
            key for key, value in self.models.items() if value == model
        )

        try:
            if hasattr(model, 'predict'):
                result = model.predict(data)
            elif callable(model):
                result = model(data)
            else:
                raise ValueError(
                    f"Model '{model_name}' must implement a 'predict' "
                    f"method or be callable."
                )
        except Exception as e:
            logger.error(f"Error during inference with model '{model_name}': {e}")
            raise
        finally:
            end_time = time.time()
            latency = end_time - start_time
            self.performance_metrics[model_name]['latency'] = latency
            logger.info(
                f"Model '{model_name}' inference latency: {latency:.4f} seconds"
            )

        return result

    def _select_model_by_performance(self) -> Any:
        """
        Selects a model based on its real-time performance metrics.

        Returns
        -------
        model : object
            The selected model for inference.

        Notes
        -----
        The method selects the model with the lowest latency that is below
        the ``latency_threshold``. If no model meets the threshold, the
        model with the lowest latency is selected, and a warning is logged.

        Mathematically, let :math:`M = \{ m_1, m_2, \dots, m_n \}` be the
        set of models with corresponding latencies
        :math:`L = \{ l_1, l_2, \dots, l_n \}`. The selected model
        :math:`m_k` is:

        .. math::

            m_k = \arg\min_{i} \{ l_i \mid l_i \leq T \}

        where :math:`T` is the latency threshold. If no :math:`l_i \leq T`,
        then select :math:`m_k = \arg\min_{i} l_i`.

        """
        available_models = [
            (model_name, metrics['latency'])
            for model_name, metrics in self.performance_metrics.items()
            if metrics['latency'] is not None
        ]

        if not available_models:
            logger.warning(
                "No performance metrics available. Selecting a model "
                "based on traffic split."
            )
            return self._select_model_by_traffic()

        # Sort models by latency
        available_models.sort(key=lambda x: x[1])

        for model_name, latency in available_models:
            if latency <= self.latency_threshold:
                logger.info(
                    f"Selected model '{model_name}' based on latency: "
                    f"{latency:.4f} seconds"
                )
                return self.models[model_name]

        # If no model meets the latency threshold, select the one with lowest latency
        best_model_name, best_latency = available_models[0]
        logger.warning(
            f"No model meets the latency threshold of {self.latency_threshold} "
            f"seconds. Selecting model '{best_model_name}' with lowest latency "
            f"of {best_latency:.4f} seconds."
        )
        return self.models[best_model_name]

    def _select_model_by_traffic(self) -> Any:
        """
        Selects a model based on traffic split ratios or a custom traffic
        algorithm.

        Returns
        -------
        model : object
            The selected model for inference.

        Notes
        -----
        If a custom traffic algorithm is provided via ``traffic_algorithm``,
        it is used to select the model. Otherwise, a weighted random
        selection is performed based on the ``traffic_split`` ratios.

        Mathematically, given the set of models :math:`M = \{ m_1, m_2,
        \dots, m_n \}` and corresponding traffic split ratios :math:`w_i`
        where :math:`\sum_{i=1}^n w_i = 1`, the probability of selecting
        model :math:`m_i` is :math:`P(m_i) = w_i`.

        """
        if self.traffic_algorithm:
            selected_model_name = self.traffic_algorithm(
                self.models, self.traffic_split
            )
            logger.info(
                f"Selected model '{selected_model_name}' using custom "
                f"traffic algorithm."
            )
            return self.models[selected_model_name]

        model_names = list(self.traffic_split.keys())
        model_weights = list(self.traffic_split.values())

        selected_model_name = random.choices(
            model_names, weights=model_weights, k=1
        )[0]
        logger.info(
            f"Routing request to model '{selected_model_name}' based on "
            f"traffic split."
        )
        return self.models[selected_model_name]


@smartFitRun 
class InferenceParallelizer(BaseInference):
    """
    Parallelizes model inference across multiple threads or processes for
    high-throughput inference, with options for batch processing, error
    handling, timeouts, and resource affinity.

    Parameters
    ----------
    model : object
        The machine learning model for parallelized inference. The model
        should implement a ``predict`` method or be callable.
    parallel_type : {'threads', 'processes'}, default='threads'
        Type of parallelism to use. Choose between ``'threads'`` or
        ``'processes'``.
    num_workers : int, default=4
        Number of threads or processes to use for parallel inference.
    timeout : float or None, default=None
        Timeout in seconds for each inference task. If ``None``, no timeout
        is applied.
    batch_size : int, default=1
        Number of inputs to process in each batch.
    handle_errors : bool, default=True
        Whether to handle errors gracefully during inference.
    cpu_affinity : list of int or None, default=None
        List of CPU cores to pin the threads or processes to. If ``None``,
        no CPU affinity is set.
    gpu_enabled : bool, default=False
        Whether to enable GPU acceleration using PyTorch.

    Notes
    -----
    The :class:`InferenceParallelizer` class facilitates high-throughput
    model inference by parallelizing inference tasks across multiple threads
    or processes. It supports batch processing, error handling, timeouts,
    and resource affinity settings.

    The class can use either threading or multiprocessing for parallelism.
    Threading is suitable for I/O-bound tasks, while multiprocessing is
    better for CPU-bound tasks due to the Global Interpreter Lock (GIL) in
    Python.

    If ``gpu_enabled`` is set to ``True``, the class uses PyTorch for
    GPU-accelerated inference. The model must support GPU execution and be
    compatible with PyTorch.

    Examples
    --------
    >>> from gofast.mlops.inference import InferenceParallelizer
    >>> model = MyModel()  # Your model should implement a predict method
    >>> data = [{'input1': x, 'input2': x * 2} for x in range(1000)]
    >>> parallelizer = InferenceParallelizer(
    ...     model=model,
    ...     parallel_type='processes',
    ...     max_workers=4,
    ...     batch_size=10,
    ...     gpu_enabled=False
    ... )
    >>> results = parallelizer.run_parallel_inference(data)
    >>> print(results)

    See Also
    --------
    concurrent.futures.ThreadPoolExecutor : For threading-based parallelism.
    multiprocessing.Pool : For process-based parallelism.

    References
    ----------
    .. [1] "Python Parallel Processing", Python Documentation.
       https://docs.python.org/3/library/concurrent.futures.html
    .. [2] Micha Gorelick and Ian Ozsvald, *High Performance Python*, O'Reilly Media, 2014.

    """

    @validate_params({
        'parallel_type': [StrOptions({'threads', 'processes'})],
        'handle_errors': [bool],
        'cpu_affinity': [list, None],
    })
    def __init__(
        self,
        model: Any,
        parallel_type: str = 'threads',
        max_workers: int = 4,
        timeout: Optional[float] = None,
        batch_size: int = 1,
        handle_errors: bool = True,
        cpu_affinity: Optional[List[int]] = None,
        gpu_enabled: bool = False
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
            timeout=timeout,
            gpu_enabled=gpu_enabled,
        )
        
        self.parallel_type = parallel_type
        self.handle_errors = handle_errors
        self.cpu_affinity = cpu_affinity

    
    @RunReturn(attribute_name="results_")
    @validate_params ({"data": [list]})
    def run(self, data: List[Dict]) -> List[Any]:
        """
        Runs inference in parallel on the provided data with batch processing
        and error handling.

        Parameters
        ----------
        data : list of dict
            List of input data for inference.

        Returns
        -------
        results : list
            The model's predictions.

        Notes
        -----
        The method splits the input data into batches of size ``batch_size``
        and distributes them across multiple threads or processes for parallel
        inference.

        If ``handle_errors`` is set to ``True``, errors during inference are
        caught and logged, and the corresponding batch results are skipped.

        Examples
        --------
        >>> results = parallelizer.run(data)

        """
        
        if self.cpu_affinity:
            logger.info(f"Setting CPU affinity to cores: {self.cpu_affinity}")
            try:
                os.sched_setaffinity(0, self.cpu_affinity)
            except AttributeError:
                logger.warning("CPU affinity setting is not supported on this platform.")
            except Exception as e:
                logger.error(f"Error setting CPU affinity: {e}")
                
        # Split data into batches
        batches = [data[i:i + self.batch_size]
                   for i in range(0, len(data), self.batch_size)]

        if self.parallel_type == "threads":
            self.results_= self._run_threaded_inference(batches)
        elif self.parallel_type == "processes":
            self.results_= self._run_multiprocess_inference(batches)
        else:
            raise ValueError(
                "Unsupported parallelization type. Choose 'threads' or 'processes'."
            )

    def _run_threaded_inference(self, batches: List[List[Dict]]) -> List[Any]:
        """
        Runs inference using multi-threading, with optional error handling
        and timeout.

        Parameters
        ----------
        batches : list of list of dict
            Input data in batches.

        Returns
        -------
        results : list
            Predictions from the model.

        Notes
        -----
        This method uses a :class:`concurrent.futures.ThreadPoolExecutor` to
        execute inference tasks in separate threads.

        """
        results = []

        def thread_task(batch):
            try:
                if self.gpu_enabled:
                    return self._gpu_inference(batch)
                if hasattr(self.model, 'predict'):
                    return self.model.predict(batch)
                elif callable(self.model):
                    return self.model(batch)
                else:
                    raise ValueError(
                        "Model must implement a 'predict' method or be callable."
                    )
            except Exception as e:
                if self.handle_errors:
                    logger.error(f"Error during threaded inference: {e}")
                    return None
                else:
                    raise

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(thread_task, batch): batch for batch in batches
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout)
                    if result is not None:
                        results.extend(result)
                except concurrent.futures.TimeoutError:
                    logger.error("Threaded inference timed out.")
                except Exception as e:
                    logger.error(f"Exception in threaded inference: {e}")

        return results

    def _run_multiprocess_inference(self, batches: List[List[Dict]]) -> List[Any]:
        """
        Runs inference using multi-processing, with optional error handling
        and timeout.

        Parameters
        ----------
        batches : list of list of dict
            Input data in batches.

        Returns
        -------
        results : list
            Predictions from the model.

        Notes
        -----
        This method uses a :class:`multiprocessing.Pool` to execute inference
        tasks in separate processes.

        """
        results = []

        def process_task(batch):
            try:
                if self.gpu_enabled:
                    return self._gpu_inference(batch)
                if hasattr(self.model, 'predict'):
                    return self.model.predict(batch)
                elif callable(self.model):
                    return self.model(batch)
                else:
                    raise ValueError(
                        "Model must implement a 'predict' method or be callable."
                    )
            except Exception as e:
                if self.handle_errors:
                    logger.error(f"Error during multiprocessing inference: {e}")
                    return None
                else:
                    raise

        with multiprocessing.Pool(processes=self.max_workers) as pool:
            future_results = [
                pool.apply_async(process_task, (batch,)) for batch in batches
            ]
            for future in future_results:
                try:
                    result = future.get(timeout=self.timeout)
                    if result is not None:
                        results.extend(result)
                except multiprocessing.TimeoutError:
                    logger.error("Multiprocessing inference timed out.")
                except Exception as e:
                    logger.error(f"Exception in multiprocessing inference: {e}")

        return results

    @ensure_pkg(
        "torch",
        extra="The 'torch' package is required for GPU inference.",
        auto_install=INSTALL_DEPENDENCIES,  
        use_conda=USE_CONDA  
    )
    def _gpu_inference(self, batch: List[Dict]) -> List[Any]:
        """
        Handles GPU-accelerated inference for a batch of data using PyTorch.

        Parameters
        ----------
        batch : list of dict
            Input batch data for inference. Each dictionary is converted to a
            tensor.

        Returns
        -------
        predictions : list
            Model predictions using GPU acceleration.

        Notes
        -----
        The method checks if CUDA is available and moves the model and data
        to the GPU for inference. If CUDA is not available, it falls back to
        CPU inference.

        Raises
        ------
        RuntimeError
            If the model does not support CUDA or an error occurs during
            inference.

        """
        logger.info("Performing GPU inference.")

        import torch

        # Ensure CUDA is available
        if not torch.cuda.is_available():
            logger.error(
                "CUDA is not available on this machine. Falling back to CPU inference."
            )
            if hasattr(self.model, 'predict'):
                return self.model.predict(batch)
            elif callable(self.model):
                return self.model(batch)
            else:
                raise ValueError(
                    "Model must implement a 'predict' method or be callable."
                )

        if not hasattr(self.model, "cuda"):
            raise RuntimeError(
                "GPU inference requested, but the model does not support CUDA."
            )

        try:
            # Convert the input batch to a PyTorch tensor
            gpu_batch = self._batch_to_tensor(batch).cuda()  # Transfer batch to GPU

            # Ensure the model is on the GPU
            if not next(self.model.parameters()).is_cuda:
                logger.info("Transferring model to GPU.")
                self.model = self.model.cuda()

            # Set model to evaluation mode
            self.model.eval()

            # Perform inference on the GPU with no gradient tracking
            with torch.no_grad():
                logger.info("Running inference on GPU.")
                predictions = self.model(gpu_batch)

            # Transfer results back to CPU and convert to list
            logger.info("Transferring predictions back to CPU.")
            predictions = predictions.cpu().detach().numpy()

            return predictions.tolist()

        except Exception as e:
            logger.error(f"Error during GPU inference: {e}")
            raise

    @ensure_pkg(
        "torch",
        extra="The 'torch' package is required for converting batches to tensors.",
        auto_install=INSTALL_DEPENDENCIES,  
        use_conda=USE_CONDA  
    )
    def _batch_to_tensor(self, batch: List[Dict]):
        """
        Converts a batch of input dictionaries into a PyTorch tensor for GPU
        processing.

        Parameters
        ----------
        batch : list of dict
            Input batch data as a list of dictionaries.

        Returns
        -------
        tensor : torch.Tensor
            PyTorch tensor representation of the batch.

        Notes
        -----
        Assumes that each item in the batch is a dictionary with numerical
        values.

        Raises
        ------
        Exception
            If an error occurs during the conversion.

        """
        logger.info("Converting batch data to PyTorch tensor.")

        try:
            # Assuming the batch is a list of dictionaries with numerical values
            # Example: [{'input1': 1.0, 'input2': 2.0}, {'input1': 3.0, 'input2': 4.0}]
            batch_values = [
                [float(value) for value in item.values()] for item in batch
            ]
            import torch
            return torch.tensor(batch_values, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error while converting batch to tensor: {e}")
            raise

@smartFitRun 
class InferenceCacheManager(BaseInference):
    """
    Adds caching support for model inference to reduce redundant predictions.

    Supports custom cache eviction policies, time-based expiration, and
    persistent caching.

    Parameters
    ----------
    model : object
        The model to perform inference with. The model should implement a
        ``predict`` method or be callable.
    cache_size : int, default=1000
        Size of the cache, i.e., the maximum number of entries to store.
    eviction_policy : {'LRU', 'LFU', 'TTL'}, default='LRU'
        Cache eviction policy to use:
            - 'LRU': Least Recently Used
            - 'LFU': Least Frequently Used
            - 'TTL': Time-To-Live based eviction
    ttl : int or None, default=None
        Time-to-live for cache entries in seconds. Only applicable when
        ``eviction_policy`` is 'TTL'.
    custom_hash_fn : callable or None, default=None
        Custom hash function to generate cache keys from input data.
    persistent_cache_path : str or None, default=None
        Path to save/load persistent cache. If ``None``, the cache is not saved
        or loaded from disk.

    Notes
    -----
    The `InferenceCacheManager` class provides a caching layer for model
    inference to avoid redundant computations. It supports different cache
    eviction policies and can persist the cache to disk for reuse across
    sessions.

    Examples
    --------
    >>> from gofast.mlops.inference import InferenceCacheManager
    >>> model = MyModel()  # Your model should implement a predict method
    >>> cache_manager = InferenceCacheManager(
    ...     model=model,
    ...     cache_size=500,
    ...     eviction_policy='LRU'
    ... )
    >>> data = {'input1': 1.0, 'input2': 2.0}
    >>> result = cache_manager.predict(data)
    >>> print(result)

    See Also
    --------
    cachetools.LRUCache : Least Recently Used cache implementation.
    cachetools.LFUCache : Least Frequently Used cache implementation.
    cachetools.TTLCache : Time-To-Live based cache implementation.

    References
    ----------
    .. [1] "cachetools – Extensible memoizing collections and decorators",
       Python Package Index.
       https://pypi.org/project/cachetools/

    """

    @validate_params({
        'cache_size': [Interval(Integral, 1, None, closed='left')],
        'eviction_policy': [StrOptions({'LRU', 'LFU', 'TTL'})],
        'ttl': [Interval(Integral, 1, None, closed='left'), None],
        'custom_hash_fn': [callable, None],
        'persistent_cache_path': [str, None],
    })
    def __init__(
        self,
        model: Any,
        cache_size: int = 1000,
        eviction_policy: str = 'LRU',
        ttl: Optional[int] = None,
        custom_hash_fn: Optional[Callable] = None,
        persistent_cache_path: Optional[str] = None,
    ):
        super().__init__( model=model )
 
        self.model = model
        self.ttl= ttl
        self.cache_size=cache_size
        self.custom_hash_fn = custom_hash_fn
        self.persistent_cache_path = persistent_cache_path
        self.eviction_policy= eviction_policy
        

        
    @ensure_pkg(
        "cachetools",
        extra="The 'cachetools' package is required for caching functionality.",
        auto_install=INSTALL_DEPENDENCIES,  
        use_conda=USE_CONDA  
    )
    def _init_lru_cache(self, cache_size: int):
        """
        Initializes an LRUCache.

        Parameters
        ----------
        cache_size : int
            Maximum number of entries to store in the cache.

        Returns
        -------
        cache : cachetools.LRUCache
            Initialized LRU cache.

        """
        from cachetools import LRUCache
        return LRUCache(maxsize=cache_size)

    @ensure_pkg(
        "cachetools",
        extra="The 'cachetools' package is required for caching functionality.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _init_lfu_cache(self, cache_size: int):
        """
        Initializes an LFUCache.

        Parameters
        ----------
        cache_size : int
            Maximum number of entries to store in the cache.

        Returns
        -------
        cache : cachetools.LFUCache
            Initialized LFU cache.

        """
        from cachetools import LFUCache
        return LFUCache(maxsize=cache_size)

    @ensure_pkg(
        "cachetools",
        extra="The 'cachetools' package is required for caching functionality.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _init_ttl_cache(self, cache_size: int, ttl: int):
        """
        Initializes a TTLCache.

        Parameters
        ----------
        cache_size : int
            Maximum number of entries to store in the cache.
        ttl : int
            Time-to-live for cache entries in seconds.

        Returns
        -------
        cache : cachetools.TTLCache
            Initialized TTL cache.

        """
        from cachetools import TTLCache
        return TTLCache(maxsize=cache_size, ttl=ttl)

    @RunReturn 
    def run(self, data: Dict) -> Any:
        """
        Performs inference with caching. If the input data has been seen
        before, retrieves the cached result.

        Parameters
        ----------
        data : dict
            Input data for inference.

        Returns
        -------
        result : Any
            Model prediction result (cached or computed).

        Notes
        -----
        The method generates a hash of the input data to use as a cache key.
        If the result is cached, it is returned directly. Otherwise, the
        model's ``predict`` method is called, and the result is cached for
        future use.

        Examples
        --------
        >>> result = cache_manager.run(data)

        """
        # Initialize the cache based on the chosen eviction policy
        if self.eviction_policy == 'LRU':
            self.cache_ = self._init_lru_cache(self.cache_size)
        elif self.eviction_policy == 'LFU':
            self.cache_ = self._init_lfu_cache(self.cache_size)
        elif self.eviction_policy == 'TTL':
            if self.ttl is None:
                raise ValueError(
                    "TTL (Time-to-live) must be specified for TTL cache."
                )
            self.cache_ = self._init_ttl_cache(self.cache_size, self.ttl)
        else:
            raise ValueError(f"Unsupported eviction policy: {self.eviction_policy}")

        # Load persistent cache if path is provided
        if self.persistent_cache_path:
            self._load_persistent_cache()
            
        data_hash = self._hash_data(data)
        if data_hash in self.cache_:
            logger.info("Returning cached result.")
            self.result_= self.cache_[data_hash]
            return 

        if hasattr(self.model, 'predict'):
            self.result_ = self.model.predict(data)
        elif callable(self.model):
            self.result_ = self.model(data)
        else:
            raise ValueError(
                "Model must implement a 'predict' method or be callable."
            )

        self.cache_[data_hash] = self.result_

        # Save cache if persistence is enabled
        if self.persistent_cache_path:
            self._save_persistent_cache()


    def _hash_data(self, data: Dict) -> int:
        """
        Generates a hash for the input data for caching purposes.

        Uses a custom hash function if provided; otherwise, it uses a default
        hash based on the data's items.

        Parameters
        ----------
        data : dict
            Input data.

        Returns
        -------
        data_hash : int
            Hash value representing the input data.

        Notes
        -----
        The default hash function converts the data items to a frozenset and
        computes its hash. A custom hash function can be provided via the
        ``custom_hash_fn`` parameter during initialization.

        """
        if self.custom_hash_fn:
            return self.custom_hash_fn(data)
        # Convert data to a frozenset of sorted items to ensure consistent ordering
        data_items = frozenset(sorted(data.items()))
        return hash(data_items)

    def _save_persistent_cache(self):
        """
        Saves the cache to disk to allow persistent caching across sessions.

        Notes
        -----
        The cache is saved using the ``pickle`` module. If an error occurs
        during saving, it is logged, and the cache is not saved.

        """
        try:
            with open(self.persistent_cache_path, 'wb') as cache_file:
                pickle.dump(self.cache_, cache_file)
            logger.info(f"Cache saved to '{self.persistent_cache_path}'.")
        except Exception as e:
            logger.error(f"Failed to save cache to disk: {e}")

    def _load_persistent_cache(self):
        """
        Loads the cache from disk if persistent caching is enabled.

        Notes
        -----
        The cache is loaded using the ``pickle`` module. If an error occurs
        during loading, it is logged, and an empty cache is used.

        """
        try:
            with open(self.persistent_cache_path, 'rb') as cache_file:
                self.cache_ = pickle.load(cache_file)
            logger.info(f"Cache loaded from '{self.persistent_cache_path}'.")
        except Exception as e:
            logger.error(f"Failed to load cache from disk: {e}")
            self.cache_ = {}

    def clear_cache(self):
        """
        Clears the cache manually.

        Notes
        -----
        This method clears all entries from the cache. If persistent caching
        is enabled, it also deletes the cache file from disk.

        Examples
        --------
        >>> cache_manager.clear_cache()

        """
        check_is_runned(self, attributes=['cache_'])
        self.cache_.clear()
        logger.info("Cache cleared.")
        if self.persistent_cache_path and os.path.exists(self.persistent_cache_path):
            try:
                os.remove(self.persistent_cache_path)
                logger.info(
                    f"Persistent cache file '{self.persistent_cache_path}' deleted."
                )
            except Exception as e:
                logger.error(f"Failed to delete persistent cache file: {e}")

    def cache_info(self) -> Dict[str, Any]:
        """
        Provides cache statistics such as hits, misses, size, and max size.

        Returns
        -------
        info : dict
            Dictionary containing cache statistics.

        Notes
        -----
        The available statistics depend on the cache type. For example, ``hits``
        and ``misses`` are available for some cache types.

        Examples
        --------
        >>> info = cache_manager.cache_info()
        >>> print(info)

        """
        check_is_runned(self, attributes=['cache_'])
        info = {
            'cache_size': len(self.cache_),
            'max_cache_size': self.cache_.maxsize,
            'hits': getattr(self.cache_, 'hits', None),
            'misses': getattr(self.cache_, 'misses', None),
        }
        logger.info(f"Cache info: {info}")
        return info


    
    
