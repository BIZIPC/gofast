# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Callbacks for model training, including ModelCheckpoint, EarlyStopping, and
LearningRateScheduler. These classes provide mechanisms to save the model
during training, stop training early based on certain criteria, and adjust
the learning rate dynamically.
"""

from typing import Optional, Dict, Any, Callable, Union
from .._gofastlog import gofastlog 
from ._base import Callback 

logger = gofastlog.get_gofast_logger(__name__)

__all__ = ["ModelCheckpoint", "EarlyStopping", "LearningRateScheduler"]


class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        verbose: int = 0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        restore_best_weights: bool = False
    ):
        super().__init__(verbose=verbose)
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best: Union[float, int] = None
        self.monitor_op: Callable[[float, float], bool] = None
        self.best_weights = None

        self._init_monitor_op()

    def _init_monitor_op(self):
        if self.mode not in ['auto', 'min', 'max']:
            logger.warning(
                f"EarlyStopping mode '{self.mode}' is unknown, "
                "fallback to 'auto' mode."
            )
            self.mode = 'auto'

        if self.mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best = float('inf')
        elif self.mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best = float('-inf')
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = lambda current, best: current > best + self.min_delta
                self.best = float('-inf')
            else:
                self.monitor_op = lambda current, best: current < best - self.min_delta
                self.best = float('inf')

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the start of training.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the start of training.
        """
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of an epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Contains the logs from the end of the epoch, including metrics.
        """
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(
                f"EarlyStopping requires '{self.monitor}' available in logs. "
                "Skipping early stopping."
            )
            return

        if self.best is None:
            self.best = current
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            return

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.verbose > 0:
                logger.info(
                    f"Epoch {epoch + 1}: {self.monitor} improved to {current:.5f}"
                )
        else:
            self.wait += 1
            if self.verbose > 0:
                logger.info(
                    f"Epoch {epoch + 1}: {self.monitor} did not improve "
                    f"from {self.best:.5f}. Wait count: {self.wait}/{self.patience}"
                )
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    logger.info(
                        f"Epoch {epoch + 1}: early stopping triggered."
                    )
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        logger.info(
                            "Restoring model weights from the best epoch."
                        )
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the end of training.
        """
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(
                f"Training stopped at epoch {self.stopped_epoch + 1}."
            )
            
EarlyStopping.__doc__ = """\
EarlyStopping callback to stop training when a monitored metric has
stopped improving.

The `EarlyStopping` callback monitors a specified metric during
training and stops the training process if no improvement is observed
after a certain number of epochs, defined by the `patience` parameter.
Optionally, it can restore the model weights from the epoch with the
best value of the monitored metric.

Parameters
----------
monitor : str, optional
    Quantity to be monitored. Typically a validation metric such as
    `'val_loss'` or `'val_accuracy'`. Default is `'val_loss'`.

min_delta : float, optional
    Minimum change in the monitored quantity to qualify as an
    improvement. This is a threshold to ignore small fluctuations.
    Default is `0.0`.

patience : int, optional
    Number of epochs with no improvement after which training will be
    stopped. For example, if `patience=2`, training will stop after
    two consecutive epochs with no improvement. Default is `10`.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means messages are logged.
    Default is `0`.

mode : str, optional
    One of `'auto'`, `'min'`, or `'max'`. In `'min'` mode, training
    will stop when the monitored quantity stops decreasing; in `'max'`
    mode, it will stop when the monitored quantity stops increasing.
    In `'auto'` mode, the direction is automatically inferred from
    the name of the monitored quantity. Default is `'auto'`.

baseline : float, optional
    Baseline value for the monitored quantity. Training will stop if
    the model does not show improvement over the baseline. Default is
    `None`.

restore_best_weights : bool, optional
    Whether to restore model weights from the epoch with the best
    value of the monitored quantity. If `False`, the model weights
    obtained at the last step of training are used. Default is
    `False`.

Methods
-------
on_train_begin(logs=None)
    Called at the start of training. Resets the wait counter and the
    best metric value.

on_epoch_end(epoch, logs=None)
    Called at the end of an epoch. Checks if the monitored metric has
    improved, updates the wait counter, and decides whether to stop
    training.

on_train_end(logs=None)
    Called at the end of training. If training was stopped early,
    logs the epoch at which training stopped.

Notes
-----
The `EarlyStopping` callback is a form of regularization to prevent
overfitting [1]_. By monitoring a validation metric, it stops training
when the model starts to overfit.

Mathematically, the improvement is considered significant if:

.. math::

    \\text{current} < \\text{best} - \\text{min\\_delta}

for `'min'` mode, or:

.. math::

    \\text{current} > \\text{best} + \\text{min\\_delta}

for `'max'` mode.

Examples
--------
>>> from gofast.callbacks.model import EarlyStopping
>>> early_stopping = EarlyStopping(monitor='val_loss', patience=5)
>>> model.fit(X_train, y_train, callbacks=[early_stopping])

See Also
--------
LearningRateScheduler : Callback to adjust the learning rate during training.
Callback : Base class for creating callbacks.

References
----------
.. [1] Y. Bengio, "Practical recommendations for gradient-based training
       of deep architectures," in *Neural Networks: Tricks of the Trade*,
       Springer, 2012.
"""

class LearningRateScheduler(Callback):
    def __init__(
        self,
        schedule: Callable[[int, float], float],
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.schedule = schedule

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of an epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Contains the logs from the end of the epoch, including metrics.
        """
        logs = logs or {}
        current_lr = self.model.get_learning_rate()
        try:
            new_lr = self.schedule(epoch, current_lr)
        except Exception as e:
            logger.error(f"Error in schedule function: {e}")
            raise

        if not isinstance(new_lr, (float, int)):
            raise ValueError(
                f"The output of the schedule function should be a number. "
                f"Received: {new_lr}"
            )

        self.model.set_learning_rate(new_lr)

        if self.verbose > 0:
            logger.info(
                f"Epoch {epoch + 1}: learning rate updated to {new_lr:.6f}."
            )
    
LearningRateScheduler.__doc__ = """\
LearningRateScheduler adjusts the learning rate during training.

The `LearningRateScheduler` callback adjusts the learning rate at
each epoch according to a user-defined schedule function. The schedule
function takes the epoch index and the current learning rate as inputs
and returns the new learning rate.

Parameters
----------
schedule : callable
    A function that takes an integer parameter `epoch` (index of the
    current epoch) and a float parameter `current_lr` (current learning
    rate), and returns a new learning rate as a float.

verbose : int, optional
    Verbosity mode. `0` means silent, `1` means messages are logged.
    Default is `0`.

Methods
-------
on_epoch_end(epoch, logs=None)
    Called at the end of an epoch. Updates the learning rate according
    to the schedule function.

Notes
-----
Adjusting the learning rate during training can help in achieving
better convergence and avoiding local minima [1]_. The schedule
function allows for implementing various learning rate policies such
as step decay, exponential decay, or custom schedules.

Examples
--------
>>> from gofast.callbacks.model import LearningRateScheduler
>>> def lr_schedule(epoch, lr):
...     if epoch > 10:
...         return lr * 0.1
...     return lr
>>> lr_scheduler = LearningRateScheduler(schedule=lr_schedule)
>>> model.fit(X_train, y_train, callbacks=[lr_scheduler])

See Also
--------
EarlyStopping : Callback to stop training when a metric has stopped improving.
Callback : Base class for creating callbacks.

References
----------
.. [1] L. Smith, "Cyclical Learning Rates for Training Neural Networks,"
       *IEEE Winter Conference on Applications of Computer Vision (WACV)*,
       2017.
"""

class ModelCheckpoint(Callback):
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = 'auto'
    ):
        super().__init__(verbose=verbose)
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = None

        if mode not in ['auto', 'min', 'max']:
            logger.warning(
                f"ModelCheckpoint mode {mode} is unknown, "
                "fallback to auto mode."
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b
            self.best = float('-inf')
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = lambda a, b: a > b
                self.best = float('-inf')
            else:
                self.monitor_op = lambda a, b: a < b
                self.best = float('inf')

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(
                f"Monitor '{self.monitor}' is not available. "
                "Skipping checkpoint."
            )
        else:
            if self.save_best_only:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch + 1}: {self.monitor} improved "
                            f"from {self.best:.5f} to {current:.5f}, "
                            f"saving model to {self.filepath}"
                        )
                    self.best = current
                    self.model.save(self.filepath)
                else:
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch + 1}: {self.monitor} did not "
                            f"improve from {self.best:.5f}"
                        )
            else:
                if self.verbose > 0:
                    logger.info(
                        f"Epoch {epoch + 1}: saving model to {self.filepath}"
                    )
                self.model.save(self.filepath)
                
ModelCheckpoint.__doc__ = """\
Save the model after every epoch based on a monitored metric.

The `ModelCheckpoint` callback saves the model to the specified
`filepath` after each epoch. If `save_best_only` is set to `True`, it
only saves the model when the monitored metric has improved compared to
the previous best value. This is useful for saving the model with the
best performance during training.

Parameters
----------
filepath : str
    The path where the model file will be saved. This can include named
    formatting options, which will be filled with the values of `epoch`
    and keys in `logs` (passed in `on_epoch_end`).
monitor : str, optional
    The name of the metric to monitor, such as `'val_loss'` or
    `'val_accuracy'`. Default is `'val_loss'`.
verbose : int, optional
    Verbosity mode. `0` means silent, `1` means messages are logged.
    Default is `0`.
save_best_only : bool, optional
    If `True`, only saves the model when the monitored metric improves.
    If `False`, saves the model after every epoch. Default is `False`.
mode : str, optional
    One of `{'auto', 'min', 'max'}`. In `'min'` mode, the monitored
    metric is expected to decrease; in `'max'` mode, it is expected to
    increase. In `'auto'` mode, the direction is automatically inferred
    from the name of the monitored metric. Default is `'auto'`.

Methods
-------
on_epoch_end(epoch, logs=None)
    Called at the end of each epoch. Checks if the monitored metric has
    improved and saves the model if appropriate.

Notes
-----
The `ModelCheckpoint` callback monitors a metric, denoted as :math:`M`,
and saves the model when an improvement is detected. The best value,
:math:`M_{\text{best}}`, is updated according to the `mode`:

.. math::

    M_{\text{best}} = \begin{cases}
        \min(M_{\text{best}}, M) & \text{if mode} = 'min' \\
        \max(M_{\text{best}}, M) & \text{if mode} = 'max' \\
    \end{cases}

An improvement is considered when the current metric value satisfies:

.. math::

    \text{monitor\_op}(M, M_{\text{best}}) = \text{True}

where `monitor_op` is the comparison operator defined by the `mode`.

**Important Notes:**

- Ensure that the directory specified in `filepath` exists; otherwise,
  an error will be raised.
- The callback does not save the optimizer state. To resume training
  with the exact state, additional mechanisms are required.

Examples
--------
>>> from gofast.callbacks.model import ModelCheckpoint
>>> checkpoint = ModelCheckpoint(
...     filepath='models/best_model_epoch_{epoch:02d}.h5',
...     monitor='val_accuracy',
...     save_best_only=True,
...     mode='max',
...     verbose=1
... )
>>> model.fit(X_train, y_train, epochs=50, callbacks=[checkpoint])

In this example, the model is saved to the specified `filepath` only
when the validation accuracy improves.

See Also
--------
EarlyStopping : Stop training when a monitored metric has stopped improving.
ReduceLROnPlateau : Reduce learning rate when a metric has stopped improving.

References
----------
.. [1] F. Chollet, "Keras Documentation: Model Checkpointing",
       https://keras.io/api/callbacks/model_checkpoint/
.. [2] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*.
       MIT Press.
"""
