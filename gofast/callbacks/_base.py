# -*- coding: utf-8 -*-


from typing import Optional, Dict, Any

from .._gofastlog import gofastlog 
from  ..api.property import BaseClass 
logger = gofastlog.get_gofast_logger(__name__)

class Callback (BaseClass):
    """
    Base class for creating callbacks in the gofast package.

    This class defines the basic structure and methods for callbacks. Users can
    inherit from this class and override the methods to define custom behaviors.

    Attributes
    ----------
    model : object
        The model instance associated with this callback.
    params : dict
        A dictionary of parameters (e.g., epochs, batch size).
    verbose : int
        Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch).
    history : dict
        Records the history of metrics and other information.
    """

    def __init__(
        self,
        verbose: int = 0,
        **kwargs
    ):
        self.model = None
        self.params: Dict[str, Any] = {}
        self.verbose = verbose
        self.history: Dict[str, Any] = {}
        self.logger = logger
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_model(self, model):
        """
        Sets the model attribute.

        Parameters
        ----------
        model : object
            The model instance to be set.
        """
        self.model = model

    def set_params(self, params: Dict[str, Any]):
        """
        Sets the parameters attribute.

        Parameters
        ----------
        params : dict
            A dictionary of parameters (e.g., epochs, batch size).
        """
        self.params = params

    def on_train_start(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the start of training.

        Parameters
        ----------
        logs : dict, optional
            Currently no data is passed to this argument for this method,
            but that may change in the future.
        """
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of training.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the end of training.
        """
        pass

    def on_epoch_start(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of an epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Contains the logs from the start of the epoch.
        """
        pass

    def on_epoch_end(
        self,
        epoch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of an epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            Contains the logs from the end of the epoch, including metrics.
        """
        pass

    def on_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of a batch.

        Parameters
        ----------
        batch : int
            The current batch number.
        logs : dict, optional
            Contains the logs from the start of the batch.
        """
        pass

    def on_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of a batch.

        Parameters
        ----------
        batch : int
            The current batch number.
        logs : dict, optional
            Contains the logs from the end of the batch, including metrics.
        """
        pass

    def on_train_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of a training batch.

        Parameters
        ----------
        batch : int
            The current training batch number.
        logs : dict, optional
            Contains the logs from the start of the training batch.
        """
        pass

    def on_train_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of a training batch.

        Parameters
        ----------
        batch : int
            The current training batch number.
        logs : dict, optional
            Contains the logs from the end of the training batch.
        """
        pass

    def on_validate_start(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the start of validation.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the start of validation.
        """
        pass

    def on_validate_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of validation.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the end of validation.
        """
        pass

    def on_validate_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of a validation batch.

        Parameters
        ----------
        batch : int
            The current validation batch number.
        logs : dict, optional
            Contains the logs from the start of the validation batch.
        """
        pass

    def on_validate_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of a validation batch.

        Parameters
        ----------
        batch : int
            The current validation batch number.
        logs : dict, optional
            Contains the logs from the end of the validation batch.
        """
        pass

    def on_test_start(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the start of testing.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the start of testing.
        """
        pass

    def on_test_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of testing.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the end of testing.
        """
        pass

    def on_test_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of a testing batch.

        Parameters
        ----------
        batch : int
            The current testing batch number.
        logs : dict, optional
            Contains the logs from the start of the testing batch.
        """
        pass

    def on_test_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of a testing batch.

        Parameters
        ----------
        batch : int
            The current testing batch number.
        logs : dict, optional
            Contains the logs from the end of the testing batch.
        """
        pass

    def on_predict_start(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the start of prediction.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the start of prediction.
        """
        pass

    def on_predict_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of prediction.

        Parameters
        ----------
        logs : dict, optional
            Contains the logs from the end of prediction.
        """
        pass

    def on_predict_batch_start(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the start of a prediction batch.

        Parameters
        ----------
        batch : int
            The current prediction batch number.
        logs : dict, optional
            Contains the logs from the start of the prediction batch.
        """
        pass

    def on_predict_batch_end(
        self,
        batch: int,
        logs: Optional[Dict[str, Any]] = None
    ):
        """
        Called at the end of a prediction batch.

        Parameters
        ----------
        batch : int
            The current prediction batch number.
        logs : dict, optional
            Contains the logs from the end of the prediction batch.
        """
        pass

    def on_exception(self, exception: Exception):
        """
        Called when an exception occurs during training/testing.

        Parameters
        ----------
        exception : Exception
            The exception that occurred.
        """
        pass

    def __getstate__(self):
        # Return state values to be pickled
        state = self.__dict__.copy()
        # Exclude unpicklable attributes if necessary
        return state

    def __setstate__(self, state):
        # Restore state from the unpickled state
        self.__dict__.update(state)
