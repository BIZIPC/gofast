# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Contains utility functions and classes for machine learning operations 
(MLOps), including configuration management, experiment tracking,
logging setup, hyperparameter management, testing utilities, pipeline
management, and metadata handling.

These utilities are designed to assist in the development, deployment, and
maintenance of machine learning models in production environments.
"""

import os
import json
import logging
import random
import yaml
import pickle
import hashlib
from itertools import product
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager

import numpy as np

from ..api.property import BaseClass
from ..decorators import EnsureFileExists 
from .._gofastlog import gofastlog 
from ..tools.validator import parameter_validator 

logger=gofastlog.get_gofast_logger(__name__)


__all__ = [
     'ConfigManager',
     'CrossValidator',
     'DataVersioning',
     'EarlyStopping',
     'ExperimentTracker',
     'MetadataManager',
     'ParameterGrid',
     'PipelineBuilder',
     'Timer',
     'TrainTestSplitter',
     'calculate_metrics',
     'get_model_metadata',
     'load_model',
     'load_pipeline',
     'log_model_summary',
     'save_model',
     'save_pipeline',
     'set_random_seed',
     'setup_logging',
]


class ConfigManager(BaseClass):
    """
    Manages configuration files in JSON or YAML format.

    This class provides methods to load, save, and update configuration
    settings for machine learning projects. It supports both JSON and YAML
    formats.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    config_format : {'json', 'yaml'}, optional
        Format of the configuration file. Defaults to ``'json'``.

    Attributes
    ----------
    config : dict
        Dictionary containing the configuration settings.

    Examples
    --------
    >>> from gofast.mlops.utils import ConfigManager
    >>> config_manager = ConfigManager('config.yaml', config_format='yaml')
    >>> config = config_manager.load_config()
    >>> config['learning_rate'] = 0.001
    >>> config_manager.save_config()

    """
    
    @EnsureFileExists
    def __init__(self, config_file: str, config_format: str = 'json'):

        self.config_format = config_format
        self.config: Dict[str, Any] = {}
        
        self.config_file =parameter_validator(
            config_format, target_strs=['json', 'yaml', 'yml'],  
            return_target_str=True, error_msg=(
                "config_format must be 'json' or 'yaml'")
            )
        
    def load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the file.

        Returns
        -------
        config : dict
            The loaded configuration settings.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.

        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        with open(self.config_file, 'r') as f:
            if self.config_format == 'json':
                self.config = json.load(f)
            elif self.config_format == 'yaml':
                self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from '{self.config_file}'.")
        return self.config

    def save_config(self):
        """
        Saves the current configuration to the file.

        Raises
        ------
        IOError
            If the configuration file cannot be written.

        """
        with open(self.config_file, 'w') as f:
            if self.config_format == 'json':
                json.dump(self.config, f, indent=4)
            elif self.config_format == 'yaml':
                yaml.safe_dump(self.config, f)
        logger.info(f"Configuration saved to '{self.config_file}'.")

    def update_config(self, updates: Dict[str, Any]):
        """
        Updates the configuration with new settings.

        Parameters
        ----------
        updates : dict
            Dictionary containing the configuration updates.

        Examples
        --------
        >>> config_manager.update_config({'batch_size': 64})

        """
        self.config.update(updates)
        logger.info("Configuration updated.")

class ExperimentTracker(BaseClass):
    """
    Tracks experiments, logging hyperparameters, metrics, and artifacts.

    This class provides methods to log experiment details, save artifacts,
    and maintain a record of different runs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    base_dir : str, optional
        Base directory to store experiment logs and artifacts.
        Defaults to ``'experiments'``.

    Attributes
    ----------
    experiment_dir : str
        Directory where experiment logs and artifacts are stored.

    Examples
    --------
    >>> from gofast.mlops.utils import ExperimentTracker
    >>> tracker = ExperimentTracker('my_experiment')
    >>> tracker.log_params({'learning_rate': 0.001, 'batch_size': 32})
    >>> tracker.log_metrics({'accuracy': 0.95})
    >>> tracker.save_artifact('model.pkl')

    """

    def __init__(self, experiment_name: str, base_dir: str = 'experiments'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.params_file = os.path.join(self.experiment_dir, 'params.json')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        self.artifacts_dir = os.path.join(self.experiment_dir, 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        logger.info(f"Experiment directory created at '{self.experiment_dir}'.")

    def log_params(self, params: Dict[str, Any]):
        """
        Logs hyperparameters for the experiment.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameters.

        """
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info("Parameters logged.")

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Logs metrics for the experiment.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric names and values.

        """
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics logged.")

    @EnsureFileExists(file_param="artifact_path", action="ignore")
    def save_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Saves an artifact file to the experiment directory.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file.
        artifact_name : str, optional
            Name to save the artifact as. If not provided, uses the original file name.

        """
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)
        dest_path = os.path.join(self.artifacts_dir, artifact_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(artifact_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Artifact '{artifact_name}' saved.")

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Sets up logging configuration.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. If not provided, logs are output to the console.
    level : int, optional
        Logging level. Defaults to ``logging.INFO``.

    Examples
    --------
    >>> from gofast.mlops.utils import setup_logging
    >>> setup_logging('training.log')

    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger.info("Logging setup complete.")

def save_model(model: Any, model_path: str):
    """
    Saves a machine learning model to the specified path using pickle.

    Parameters
    ----------
    model : object
        The machine learning model to save.
    model_path : str
        Path where the model will be saved.

    Examples
    --------
    >>> from gofast.mlops.utils import save_model
    >>> save_model(model, 'models/my_model.pkl')

    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to '{model_path}'.")

def load_model(model_path: str) -> Any:
    """
    Loads a machine learning model from the specified path.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    model : object
        The loaded machine learning model.

    Examples
    --------
    >>> from gofast.mlops.utils import load_model
    >>> model = load_model('models/my_model.pkl')

    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from '{model_path}'.")
    return model

@contextmanager
def Timer(name: str):
    """
    Context manager for timing code execution.

    Parameters
    ----------
    name : str
        Name to associate with the timing block.

    Examples
    --------
    >>> from gofast.mlops.utils import Timer
    >>> with Timer('data_loading'):
    ...     # Code block to time
    ...     data = load_data()

    """
    start_time = datetime.now()
    logger.info(f"Starting '{name}'...")
    yield
    elapsed_time = datetime.now() - start_time
    logger.info(f"'{name}' completed in {elapsed_time}.")

def set_random_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The seed value to use. Defaults to ``42``.

    Notes
    -----
    This function sets the seed for the `random`, `numpy`, and `torch` (if available)
    modules to ensure reproducible results.

    Examples
    --------
    >>> from gofast.mlops.utils import set_random_seed
    >>> set_random_seed(123)

    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}.")
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info("Random seed set for PyTorch.")
    except ImportError:
        logger.warning("PyTorch not installed; skipping torch seed setting.")

class EarlyStopping(BaseClass):
    """
    Implements early stopping mechanism to halt training when performance degrades.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped.
        Defaults to ``5``.
    delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement.
        Defaults to ``0.0``.
    monitor : str, optional
        Metric to monitor. Defaults to ``'loss'``.

    Attributes
    ----------
    best_score : float
        Best score achieved so far.
    counter : int
        Number of epochs since the last improvement.
    early_stop : bool
        Flag indicating whether early stopping should trigger.

    Examples
    --------
    >>> from gofast.mlops.utils import EarlyStopping
    >>> early_stopper = EarlyStopping(patience=3)
    >>> for epoch in range(epochs):
    ...     train_loss = train(...)
    ...     val_loss = validate(...)
    ...     early_stopper(val_loss)
    ...     if early_stopper.early_stop:
    ...         break

    """

    def __init__(self, patience: int = 5, delta: float = 0.0, monitor: str = 'loss'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value: float):
        """
        Updates the early stopping mechanism with the latest metric value.

        Parameters
        ----------
        value : float
            The latest value of the monitored metric.

        """
        score = -value if self.monitor == 'loss' else value
        if self.best_score is None:
            self.best_score = score
            logger.info("Initial best score set.")
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"No improvement. Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")
        else:
            self.best_score = score
            self.counter = 0
            logger.info("Improvement detected. Counter reset.")

def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: Optional[List[str]] = None
        ) -> Dict[str, float]:
    """
    Calculates evaluation metrics for model predictions.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels.
    y_pred : ndarray
        Predicted labels or probabilities.
    metrics : list of str, optional
        List of metrics to calculate. Supported metrics are:
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.

    Returns
    -------
    results : dict
        Dictionary of metric names and their calculated values.

    Examples
    --------
    >>> from gofast.mlops.utils import calculate_metrics
    >>> metrics = calculate_metrics(y_true, y_pred, metrics=['accuracy', 'f1'])

    """
    from sklearn.metrics import ( 
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score
        )

    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            results['precision'] = precision_score(y_true, y_pred, average='binary')
        elif metric == 'recall':
            results['recall'] = recall_score(y_true, y_pred, average='binary')
        elif metric == 'f1':
            results['f1'] = f1_score(y_true, y_pred, average='binary')
        elif metric == 'roc_auc':
            if len(np.unique(y_true)) == 2:
                results['roc_auc'] = roc_auc_score(y_true, y_pred)
            else:
                logger.warning("ROC AUC is not defined for multi-class classification.")
        else:
            logger.warning(f"Unsupported metric: {metric}")
    logger.info("Metrics calculated.")
    return results


class DataVersioning(BaseClass):
    """
    Manages data versioning for datasets using checksums and metadata.

    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    metadata_file : str, optional
        Path to the metadata file for storing checksums.
        Defaults to ``'data_metadata.json'``.

    Attributes
    ----------
    data_dir : str
        Directory containing the data files.
    metadata_file : str
        Path to the metadata file.

    Examples
    --------
    >>> from gofast.mlops.utils import DataVersioning
    >>> data_versioning = DataVersioning('data/')
    >>> data_versioning.generate_checksums()
    >>> data_versioning.check_for_changes()

    """

    def __init__(self, data_dir: str, metadata_file: str = 'data_metadata.json'):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.metadata: Dict[str, str] = {}

    def generate_checksums(self):
        """
        Generates checksums for all files in the data directory and saves them
        to the metadata file.

        """
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                checksum = self._calculate_checksum(file_path)
                relative_path = os.path.relpath(file_path, self.data_dir)
                self.metadata[relative_path] = checksum
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        logger.info("Data checksums generated and saved.")

    def check_for_changes(self) -> bool:
        """
        Checks for changes in the data files by comparing current checksums with
        those stored in the metadata file.

        Returns
        -------
        changes_detected : bool
            True if changes are detected, False otherwise.

        """

        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file '{self.metadata_file}' not found.")

        with open(self.metadata_file, 'r') as f:
            stored_metadata = json.load(f)

        current_metadata = {}
        changes_detected = False

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                checksum = self._calculate_checksum(file_path)
                relative_path = os.path.relpath(file_path, self.data_dir)
                current_metadata[relative_path] = checksum
                if relative_path not in stored_metadata or stored_metadata[relative_path] != checksum:
                    logger.warning(f"Change detected in file '{relative_path}'.")
                    changes_detected = True

        if changes_detected:
            logger.info("Data changes detected.")
        else:
            logger.info("No data changes detected.")
        return changes_detected

    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculates the MD5 checksum of a file.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        checksum : str
            MD5 checksum of the file.

        """

        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class ParameterGrid(BaseClass):
    """
    Generates a grid of parameter combinations for hyperparameter tuning.

    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    Examples
    --------
    >>> from gofast.mlops.utils import ParameterGrid
    >>> param_grid = {
    ...     'learning_rate': [0.001, 0.01],
    ...     'batch_size': [16, 32],
    ...     'optimizer': ['adam', 'sgd']
    ... }
    >>> grid = ParameterGrid(param_grid)
    >>> for params in grid:
    ...     print(params)

    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self._grid = self._generate_grid()

    def _generate_grid(self) -> List[Dict[str, Any]]:
        

        items = sorted(self.param_grid.items())
        keys, values = zip(*items)
        experiments = [dict(zip(keys, v)) for v in product(*values)]
        logger.info(f"Generated {len(experiments)} parameter combinations.")
        return experiments

    def __iter__(self):
        return iter(self._grid)

    def __len__(self):
        return len(self._grid)

    def __getitem__(self, idx):
        return self._grid[idx]


class TrainTestSplitter(BaseClass):
    """
    Utility class for splitting data into training and testing sets.

    This class provides methods to split datasets into training and testing
    subsets, with options for stratification and shuffling.

    Parameters
    ----------
    test_size : float, optional
        Proportion of the dataset to include in the test split. Defaults to
        ``0.2``.
    random_state : int, optional
        Seed used by the random number generator. Defaults to ``42``.
    stratify : bool, optional
        Whether to perform stratified splitting based on the target variable.
        Defaults to ``False``.

    Examples
    --------
    >>> from gofast.mlops.utils import TrainTestSplitter
    >>> splitter = TrainTestSplitter(test_size=0.3, stratify=True)
    >>> X_train, X_test, y_train, y_test = splitter.split(X, y)

    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42, 
                 stratify: bool = False):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(self, X: np.ndarray, y: np.ndarray):
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Target vector.

        Returns
        -------
        X_train : ndarray
            Training feature matrix.
        X_test : ndarray
            Testing feature matrix.
        y_train : ndarray
            Training target vector.
        y_test : ndarray
            Testing target vector.

        """
        from sklearn.model_selection import train_test_split

        stratify_param = y if self.stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        logger.info("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test

class CrossValidator(BaseClass):
    """
    Utility class for performing cross-validation.

    This class provides methods to perform k-fold cross-validation on models,
    allowing for evaluation of model performance across different data splits.

    Parameters
    ----------
    model : object
        The machine learning model to evaluate.
    cv : int, optional
        Number of folds in the cross-validation. Defaults to ``5``.

    Examples
    --------
    >>> from gofast.mlops.utils import CrossValidator
    >>> cross_validator = CrossValidator(model, cv=5)
    >>> scores = cross_validator.evaluate(X, y, scoring='accuracy')

    """

    def __init__(self, model: Any, cv: int = 5):
        self.model = model
        self.cv = cv

    def evaluate(self, X: np.ndarray, y: np.ndarray, scoring: str = 'accuracy'
                 ) -> Dict[str, Any]:
        """
        Evaluates the model using cross-validation.

        Parameters
        ----------
        X : ndarray
            Feature matrix.
        y : ndarray
            Target vector.
        scoring : str, optional
            Scoring metric to use. Defaults to ``'accuracy'``.

        Returns
        -------
        results : dict
            Dictionary containing cross-validation scores and statistics.

        """
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(self.model, X, y, cv=self.cv, scoring=scoring)
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores.tolist()
        }
        logger.info("Cross-validation completed.")
        return results

class PipelineBuilder(BaseClass):
    """
    Utility class for building machine learning pipelines.

    This class provides methods to create pipelines that chain together
    preprocessing steps and estimators, simplifying the model development
    process.

    Parameters
    ----------
    steps : list of tuple
        List of (name, transform) tuples (implementing fit/transform) that are
        chained together in the pipeline.

    Examples
    --------
    >>> from gofast.mlops.utils import PipelineBuilder
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipeline_builder = PipelineBuilder([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', LogisticRegression())
    ... ])
    >>> pipeline = pipeline_builder.build()
    >>> pipeline.fit(X_train, y_train)

    """

    def __init__(self, steps: List[tuple]):
        self.steps = steps

    def build(self):
        """
        Builds the machine learning pipeline.

        Returns
        -------
        pipeline : sklearn.pipeline.Pipeline
            The constructed pipeline object.

        """
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline(self.steps)
        logger.info("Pipeline constructed.")
        return pipeline

class MetadataManager(BaseClass):
    """
    Manages model metadata, including parameters, hyperparameters, and metrics.

    This class provides methods to save and load metadata associated with models,
    facilitating model versioning and reproducibility.

    Parameters
    ----------
    metadata_file : str, optional
        Path to the metadata file. Defaults to ``'model_metadata.json'``.

    Attributes
    ----------
    metadata : dict
        Dictionary containing the model metadata.

    Examples
    --------
    >>> from gofast.mlops.utils import MetadataManager
    >>> metadata_manager = MetadataManager('metadata.json')
    >>> metadata_manager.save_metadata({'model_name': 'my_model', 'version': 1})
    >>> metadata = metadata_manager.load_metadata()

    """

    def __init__(self, metadata_file: str = 'model_metadata.json'):
        self.metadata_file = metadata_file
        self.metadata: Dict[str, Any] = {}

    def save_metadata(self, metadata: Dict[str, Any]):
        """
        Saves metadata to the metadata file.

        Parameters
        ----------
        metadata : dict
            Dictionary containing metadata information.

        """
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to '{self.metadata_file}'.")

    def load_metadata(self) -> Dict[str, Any]:
        """
        Loads metadata from the metadata file.

        Returns
        -------
        metadata : dict
            The loaded metadata.

        Raises
        ------
        FileNotFoundError
            If the metadata file does not exist.

        """
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file '{self.metadata_file}' not found.")

        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Metadata loaded from '{self.metadata_file}'.")
        return self.metadata

    def update_metadata(self, updates: Dict[str, Any]):
        """
        Updates the metadata with new information.

        Parameters
        ----------
        updates : dict
            Dictionary containing metadata updates.

        """
        self.metadata.update(updates)
        logger.info("Metadata updated.")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves the current metadata.

        Returns
        -------
        metadata : dict
            The current metadata.

        """
        return self.metadata

def save_pipeline(pipeline: Any, pipeline_path: str):
    """
    Saves a machine learning pipeline to the specified path using joblib.

    Parameters
    ----------
    pipeline : object
        The pipeline object to save.
    pipeline_path : str
        Path where the pipeline will be saved.

    Examples
    --------
    >>> from gofast.mlops.utils import save_pipeline
    >>> save_pipeline(pipeline, 'pipelines/my_pipeline.pkl')

    """
    from joblib import dump

    dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved to '{pipeline_path}'.")

def load_pipeline(pipeline_path: str) -> Any:
    """
    Loads a machine learning pipeline from the specified path.

    Parameters
    ----------
    pipeline_path : str
        Path to the saved pipeline file.

    Returns
    -------
    pipeline : object
        The loaded pipeline object.

    Examples
    --------
    >>> from gofast.mlops.utils import load_pipeline
    >>> pipeline = load_pipeline('pipelines/my_pipeline.pkl')

    """
    from joblib import load

    pipeline = load(pipeline_path)
    logger.info(f"Pipeline loaded from '{pipeline_path}'.")
    return pipeline

def log_model_summary(model: Any, model_name: str = 'Model'):
    """
    Logs a summary of the model structure.

    Parameters
    ----------
    model : object
        The machine learning model.
    model_name : str, optional
        Name of the model. Defaults to ``'Model'``.

    Examples
    --------
    >>> from gofast.mlops.utils import log_model_summary
    >>> log_model_summary(model, model_name='Neural Network')

    """
    from sklearn.utils import estimator_html_repr

    try:
        summary = estimator_html_repr(model)
        logger.info(f"{model_name} summary:\n{summary}")
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")

def get_model_metadata(model: Any) -> Dict[str, Any]:
    """
    Extracts metadata from a scikit-learn model.

    Parameters
    ----------
    model : object
        The scikit-learn model.

    Returns
    -------
    metadata : dict
        Dictionary containing model metadata, such as parameters and estimator type.

    Examples
    --------
    >>> from gofast.mlops.utils import get_model_metadata
    >>> metadata = get_model_metadata(model)
    >>> print(metadata)

    """
    metadata = {
        'class_name': model.__class__.__name__,
        'module': model.__module__,
        'params': model.get_params()
    }
    logger.info("Model metadata extracted.")
    return metadata
