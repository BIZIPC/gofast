# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Data-related callbacks for training, including BaseData, DataAugmentation,
DataOps, and DataLogging. These classes handle data preprocessing, augmentation,
operations, and logging to support model training workflows.
"""

import time 
import json 
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from .._gofastlog import gofastlog
from ._base import Callback 

logger = gofastlog.get_gofast_logger(__name__)

__all__= ["BaseData", "DataAugmentation", "DataOps", "DataLogging"]


class BaseData(Callback):
    def __init__(
        self,
        model=None,
        data_transformations: Optional[List[Callable]] = None,
        batch_transformations: Optional[List[Callable]] = None,
        epoch_operations: Optional[List[Callable]] = None,
        batch_operations: Optional[List[Callable]] = None,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.model = model
        self.data_transformations = data_transformations or []
        self.batch_transformations = batch_transformations or []
        self.epoch_operations = epoch_operations or []
        self.batch_operations = batch_operations or []
        self.data_statistics: Dict[int, Dict[str, Any]] = {}

    def on_epoch_start(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        self.data_statistics.setdefault(epoch, {})
        epoch_start_time = time.time()
        self.data_statistics[epoch]['epoch_start'] = epoch_start_time

        if self.verbose:
            logger.info(
                f"Epoch {epoch + 1} started at "
                f"{epoch_start_time:.4f} seconds."
            )

        self._log_epoch_start_statistics(epoch)
        self._apply_data_transformation(epoch)
        self._check_data_integrity(epoch)
        self._handle_epoch_specific_operations(epoch)

    def on_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        logs = logs or {}
        batch_start_time = time.time()
        if 'batch_times' not in self.data_statistics:
            self.data_statistics['batch_times'] = {}
        self.data_statistics['batch_times'][batch] = {
            'start_time': batch_start_time
        }

        if self.batch_transformations:
            data = self.model.current_batch_data
            data = self._apply_batch_transformations(data)
            self.model.current_batch_data = data
            if self.verbose:
                logger.info(f"Batch {batch + 1}: Applied batch transformations.")

        self._handle_batch_specific_operations(batch)

    def on_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        logs = logs or {}
        batch_end_time = time.time()
        if 'batch_times' in self.data_statistics and batch in self.data_statistics['batch_times']:
            start_time = self.data_statistics['batch_times'][batch]['start_time']
            duration = batch_end_time - start_time
            self.data_statistics['batch_times'][batch]['end_time'] = batch_end_time
            self.data_statistics['batch_times'][batch]['duration'] = duration
            if self.verbose:
                logger.info(f"Batch {batch + 1}: Processed in {duration:.4f} seconds.")

        if logs:
            if 'batch_metrics' not in self.data_statistics:
                self.data_statistics['batch_metrics'] = {}
            self.data_statistics['batch_metrics'][batch] = logs

        self._handle_batch_specific_operations(batch)

    def _handle_epoch_specific_operations(self, epoch: int):
        for operation in self.epoch_operations:
            operation(self.model, epoch)
            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: Executed epoch operation "
                    f"{operation.__name__}."
                )

    def _handle_batch_specific_operations(self, batch: int):
        for operation in self.batch_operations:
            operation(self.model, batch)
            if self.verbose:
                logger.info(
                    f"Batch {batch + 1}: Executed batch operation "
                    f"{operation.__name__}."
                )

    def _transform_data(self, data):
        transformed_data = data
        for transform in self.data_transformations:
            transformed_data = transform(transformed_data)
            if self.verbose:
                logger.info(
                    f"Applied data transformation: {transform.__name__}"
                )
        return transformed_data

    def _apply_data_transformation(self, epoch: int):
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            transformed_data = self._transform_data(data)
            self.model.train_data = transformed_data
            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: Data transformations applied."
                )

    def _apply_batch_transformations(self, data):
        transformed_data = data
        for transform in self.batch_transformations:
            transformed_data = transform(transformed_data)
            if self.verbose:
                logger.info(
                    f"Applied batch transformation: {transform.__name__}"
                )
        return transformed_data

    def _log_epoch_start_statistics(self, epoch: int):
        if hasattr(self.model, 'train_data'):
            train_data_size = len(self.model.train_data)
            self.data_statistics[epoch]['train_data_size'] = train_data_size
            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: Training data size: "
                    f"{train_data_size} samples."
                )
            self._log_data_distribution(epoch)

    def _log_data_distribution(self, epoch: int):
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            labels = getattr(data, 'labels', None)
            if labels is not None:
                unique_labels, counts = np.unique(labels, return_counts=True)
                class_counts = dict(zip(unique_labels, counts))
                self.data_statistics[epoch]['data_distribution'] = class_counts
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch + 1}: Data distribution: {class_counts}"
                    )
            else:
                logger.warning(
                    f"No 'labels' attribute found in training data for epoch {epoch + 1}."
                )

    def _check_data_integrity(self, epoch: int):
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            if len(data) == 0:
                raise ValueError(
                    f"Epoch {epoch + 1}: Training data is empty."
                )
            if isinstance(data, dict):
                invalid_keys = [
                    key for key, value in data.items()
                    if not isinstance(value, np.ndarray)
                ]
                if invalid_keys:
                    raise TypeError(
                        f"Epoch {epoch + 1}: Data contains invalid types for keys "
                        f"{invalid_keys}. Expected numpy arrays."
                    )

BaseData.__doc__ = """\
Base class for data-related callbacks in the gofast package.

This class serves as a foundational framework for implementing
data-related callbacks that operate during different stages of the
training process, such as data transformations, logging, and
statistics collection. It provides flexibility to customize data
handling by allowing the injection of functions at epoch and batch
levels.

Parameters
----------
model : object, optional
    The model instance associated with this callback. This is
    typically set automatically by the training framework.

data_transformations : list of callable, optional
    A list of functions applied to the training data at the start of
    each epoch. Each function should accept the data as input and
    return the transformed data.

batch_transformations : list of callable, optional
    A list of functions applied to each batch of data at the start of
    each batch. Each function should accept the batch data as input
    and return the transformed batch data.

epoch_operations : list of callable, optional
    A list of functions executed at the start of each epoch. Each
    function should accept the model and the epoch index as inputs.

batch_operations : list of callable, optional
    A list of functions executed at the start and end of each batch.
    Each function should accept the model and the batch index as
    inputs.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means progress messages.

Methods
-------
on_epoch_start(epoch, logs=None)
    Called at the start of an epoch. Applies data transformations and
    executes epoch operations.

on_epoch_end(epoch, logs=None)
    Called at the end of an epoch. Collects statistics and logs data
    distribution.

on_batch_start(batch, logs=None)
    Called at the start of a batch. Applies batch transformations and
    executes batch operations.

on_batch_end(batch, logs=None)
    Called at the end of a batch. Collects statistics and logs
    processing time.

Notes
-----
The `BaseData` class is designed to be inherited by other
data-related callbacks in the gofast package, such as `DataAugmentation`,
`DataOps`, and `DataLogging`. It provides common functionality and a
flexible interface for customizing data processing during training.

Data transformations can include mathematical operations such as
normalization, where data is scaled to have zero mean and unit
variance:

.. math::

    x' = \\frac{x - \\mu}{\\sigma}

where :math:`\\mu` is the mean and :math:`\\sigma` is the standard
deviation of the dataset.

Examples
--------
>>> from gofast.callbacks.data import BaseData
>>> def normalize_data(data):
...     mean = data.mean()
...     std = data.std()
...     return (data - mean) / std
>>> base_data_callback = BaseData(
...     data_transformations=[normalize_data],
...     verbose=1
... )

See Also
--------
DataAugmentation : Callback for applying data augmentation techniques.
DataOps : Callback for executing custom data operations.
DataLogging : Callback for logging data statistics.

References
----------
.. [1] Keras Callbacks API. https://keras.io/api/callbacks/
.. [2] Goodfellow et al., "Deep Learning", MIT Press, 2016.
"""

class DataOps(BaseData):
    def __init__(
        self,
        model=None,
        data_operations: Optional[List[Callable]] = None,
        data_transformations: Optional[List[Callable]] = None,
        epoch_operations: Optional[List[Callable]] = None,
        batch_operations: Optional[List[Callable]] = None,
        batch_transformations: Optional[List[Callable]] = None,
        verbose: int = 0
    ):
        combined_data_transformations = (
            data_transformations or []) + (data_operations or [])
        super().__init__(
            model=model,
            data_transformations=combined_data_transformations,
            batch_transformations=batch_transformations,
            epoch_operations=epoch_operations,
            batch_operations=batch_operations,
            verbose=verbose
        )

    def on_epoch_start(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        if self.verbose:
            logger.info(f"Epoch {epoch + 1}: Starting Data Operations...")
        self.data_statistics[epoch] = {'epoch_start': time.time()}
        super().on_epoch_start(epoch, logs)

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        if epoch in self.data_statistics:
            self.data_statistics[epoch]['epoch_end'] = time.time()
        self._log_data_operations(epoch)
        super().on_epoch_end(epoch, logs)

    def _log_data_operations(self, epoch: int):
        if epoch in self.data_statistics:
            operations_applied = [
                func.__name__ for func in self.data_transformations
            ]
            self.data_statistics[epoch]['data_operations'] = operations_applied
            if self.verbose:
                logger.info(
                    f"Epoch {epoch + 1}: Data operations applied: "
                    f"{', '.join(operations_applied)}"
                )

DataOps.__doc__ = """\
Callback for executing custom data operations during training.

The `DataOps` class extends `BaseData` to perform custom data
operations at the epoch level. It allows the user to specify a list
of data operations that are applied to the training data at the
beginning of each epoch.

Parameters
----------
model : object, optional
    The model instance associated with this callback.

data_operations : list of callable, optional
    A list of functions that perform operations on the data. Each
    function should accept the data as input and return the modified
    data.

data_transformations : list of callable, optional
    Additional data transformations to be applied alongside data
    operations.

epoch_operations : list of callable, optional
    Additional operations to be executed at the start of each epoch.

batch_operations : list of callable, optional
    Operations to be executed at the start and end of each batch.

batch_transformations : list of callable, optional
    Transformations to be applied to each batch of data.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means progress messages.

Methods
-------
on_epoch_start(epoch, logs=None)
    Called at the start of an epoch. Applies data operations and
    data transformations.

on_epoch_end(epoch, logs=None)
    Called at the end of an epoch. Logs data operations performed.

Notes
-----
`DataOps` provides flexibility to implement any custom data
manipulations needed during training, such as data cleaning,
balancing, or feature engineering.

Examples
--------
>>> from gofast.callbacks.data import DataOps
>>> def remove_outliers(data):
...     # Implement outlier removal
...     return cleaned_data
>>> data_ops = DataOps(
...     data_operations=[remove_outliers],
...     verbose=1
... )

See Also
--------
BaseData : Base class for data-related callbacks.
DataAugmentation : Callback for applying data augmentation.

References
----------
.. [1] Bishop, C. M. (2006). Pattern Recognition and Machine
       Learning. Springer.
"""

class DataAugmentation(BaseData):
    def __init__(
        self,
        model=None,
        augmentation_functions: Optional[List[Callable]] = None,
        batch_transformations: Optional[List[Callable]] = None,
        verbose: int = 0
    ):
        combined_batch_transformations = (
            batch_transformations or []) + (augmentation_functions or [])
        super().__init__(
            model=model,
            batch_transformations=combined_batch_transformations,
            verbose=verbose
        )
        if not self.batch_transformations:
            logger.warning("No augmentation functions provided.")

    def on_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        super().on_batch_start(batch, logs)
        if self.verbose:
            logger.info(f"Batch {batch + 1}: Data augmentation applied.")


DataAugmentation.__doc__ = """\
Callback for applying data augmentation during training.

The `DataAugmentation` class extends `BaseData` to perform data
augmentation on training data at the batch level. It allows the user
to specify a list of augmentation functions that are applied to each
batch before it is processed by the model.

Parameters
----------
model : object, optional
    The model instance associated with this callback.

augmentation_functions : list of callable, optional
    A list of functions that perform data augmentation. Each function
    should accept the batch data as input and return the augmented
    batch data.

batch_transformations : list of callable, optional
    Additional batch transformations to be applied alongside
    augmentation functions.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means progress messages.

Methods
-------
on_batch_start(batch, logs=None)
    Called at the start of a batch. Applies augmentation functions and
    batch transformations to the batch data.

Notes
-----
Data augmentation techniques help improve model generalization by
creating variations of the training data. Common augmentation methods
include rotations, flips, scaling, and noise addition.

Examples
--------
>>> from gofast.callbacks.data import DataAugmentation
>>> def random_flip(batch_data):
...     # Implement random flipping of images
...     return flipped_data
>>> data_augmentation = DataAugmentation(
...     augmentation_functions=[random_flip],
...     verbose=1
... )

See Also
--------
BaseData : Base class for data-related callbacks.
DataOps : Callback for executing custom data operations.

References
----------
.. [1] Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on
       Image Data Augmentation for Deep Learning.
       Journal of Big Data, 6(1), 60.
.. [2] Goodfellow et al., "Deep Learning", MIT Press, 2016.
"""

class DataLogging(BaseData):
    def __init__(
        self,
        model=None,
        log_file: str = "data_log.json",
        statistics_functions: Optional[List[Callable]] = None,
        log_on_epoch_end: bool = True,
        log_on_batch_end: bool = False,
        verbose: int = 0
    ):
        super().__init__(model=model, verbose=verbose)
        self.log_file = log_file
        self.statistics_functions = statistics_functions or [self._default_statistics]
        self.log_on_epoch_end = log_on_epoch_end
        self.log_on_batch_end = log_on_batch_end

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        if self.log_on_epoch_end:
            self._log_data(index=epoch, logs=logs, scope='epoch')
        super().on_epoch_end(epoch, logs)

    def on_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        if self.log_on_batch_end:
            self._log_data(index=batch, logs=logs, scope='batch')
        super().on_batch_end(batch, logs)

    def _log_data(self, index: int, logs: Optional[Dict[str, Any]], scope: str):
        if hasattr(self.model, 'train_data'):
            data = self.model.train_data
            data_info = {
                scope: index + 1,
                'data_statistics': self._get_data_statistics(data),
                'logs': logs or {}
            }
            try:
                with open(self.log_file, 'a') as f:
                    json.dump(data_info, f)
                    f.write("\n")
                if self.verbose:
                    logger.info(
                        f"{scope.capitalize()} {index + 1}:"
                        f" Data logged to {self.log_file}."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to log data at {scope} {index + 1}: {e}"
                )
        else:
            logger.warning(
                f"{scope.capitalize()} {index + 1}:"
                f" 'train_data' attribute not found in model."
            )

    def _get_data_statistics(self, data):
        statistics = {}
        for func in self.statistics_functions:
            try:
                stats = func(data)
                statistics.update(stats)
            except Exception as e:
                logger.error(
                    f"Error computing data statistics with {func.__name__}: {e}")
        return statistics

    def _default_statistics(self, data):
        try:
            features = data.get('features')
            if features is not None and isinstance(features, np.ndarray):
                feature_means = {
                    f"feature_{i}_mean": np.mean(features[:, i])
                    for i in range(features.shape[1])
                }
                return feature_means
            else:
                logger.warning(
                    "No 'features' key in data or data type is incorrect."
                )
                return {}
        except Exception as e:
            logger.error(f"Error computing default data statistics: {e}")
            return {}


DataLogging.__doc__ = """\
Callback for logging data statistics during training.

The `DataLogging` class extends `BaseData` to log statistics and
information about the data at specified intervals during training.
It allows the user to specify custom statistics functions and
controls whether logging occurs at the end of epochs and/or batches.

Parameters
----------
model : object, optional
    The model instance associated with this callback.

log_file : str, optional
    File path where the data logs will be stored. Default is
    `"data_log.json"`.

statistics_functions : list of callable, optional
    A list of functions that compute statistics from the data. Each
    function should accept the data as input and return a dictionary
    of statistics.

log_on_epoch_end : bool, optional
    Whether to log data at the end of each epoch. Default is `True`.

log_on_batch_end : bool, optional
    Whether to log data at the end of each batch. Default is `False`.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means progress messages.

Methods
-------
on_epoch_end(epoch, logs=None)
    Called at the end of an epoch. Logs data statistics if
    `log_on_epoch_end` is `True`.

on_batch_end(batch, logs=None)
    Called at the end of a batch. Logs data statistics if
    `log_on_batch_end` is `True`.

Notes
-----
Logging data statistics can help monitor the training process and
detect issues such as data drift or anomalies.

Examples
--------
>>> from gofast.callbacks.data import DataLogging
>>> def compute_feature_stats(data):
...     # Compute custom statistics
...     return stats_dict
>>> data_logging = DataLogging(
...     statistics_functions=[compute_feature_stats],
...     log_on_epoch_end=True,
...     verbose=1
... )

See Also
--------
BaseData : Base class for data-related callbacks.
DataOps : Callback for executing custom data operations.

References
----------
.. [1] Murphy, K. P. (2012). Machine Learning: A Probabilistic
       Perspective. MIT Press.
"""

