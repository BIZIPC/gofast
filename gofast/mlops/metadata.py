# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Manage metadata related to models, experiments, datasets, and 
training runs.
"""
import os
import platform
import subprocess
import random 
import json
import time
import zlib
from collections import deque  # noqa
from numbers import Integral, Real
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable

import numpy as np
from sklearn.utils._param_validation import StrOptions, HasMethods

from ._config import INSTALL_DEPENDENCIES, USE_CONDA 
from ..api.property import BaseClass
from ..compat.sklearn import validate_params, Interval 
from ..tools.depsutils import ensure_pkg 
from .._gofastlog import gofastlog 
logger=gofastlog.get_gofast_logger(__name__)

__all__ = [
    'MetadataManager', 
    'LineageTracker', 
    'AuditLogger', 
    'ReproducibilityEnsurer', 
    'PerformanceTracker', 
    'log_metadata',
    'retrieve',
    'compare',
    'audit',
    'sync_with_cloud',
    'validate_schema',
    'track_experiment',
    'prune_old',
]

class MetadataManager(BaseClass):
    """
    Centralized metadata management system for tracking model artifacts,
    data versions, pipeline configurations, and more. Integrates with
    distributed systems and external metadata stores.

    Parameters
    ----------
    metadata_store : {'aws', 'gcp', 'local'}
        The type of metadata store to synchronize with. Options are:
        ``'aws'`` for Amazon Web Services, ``'gcp'`` for Google Cloud
        Platform, and ``'local'`` for local storage.
    schema : dict, optional
        Schema to enforce consistency in metadata storage. If ``None``,
        a default schema is used.
    local_backup_path : str, default='metadata_backup.json'
        Local path to save metadata as a backup in case cloud sync
        fails.
    versioning : bool, default=True
        Enable versioning for metadata entries.
    encryption_key : str or None, default=None
        Key to encrypt metadata when saving to local or cloud storage.
    compression_enabled : bool, default=False
        Enable compression for large metadata objects.
    retry_policy : dict, optional
        Retry policy for cloud sync. Should contain ``'retries'`` and
        ``'backoff'`` keys. Defaults to ``{'retries': 3, 'backoff': 2}``.
    cloud_sync_frequency : int, default=5
        Number of changes before triggering automatic cloud sync.
    auto_load_backup : bool, default=True
        Automatically load local metadata backup on initialization.
    cache_enabled : bool, default=True
        Enable in-memory caching for frequently accessed metadata.
    cloud_bucket_name : str or None, default=None
        Name of the cloud bucket for storing metadata. Required for
        cloud sync with AWS or GCP.

    Notes
    -----
    The :class:`MetadataManager` class provides a centralized system for
    managing and synchronizing metadata across local and cloud storage.
    It supports versioning, encryption, compression, and caching.

    The class can synchronize metadata with cloud services like AWS S3
    or GCP Cloud Storage, and handles retries and backups in case of
    failures.

    Examples
    --------
    >>> from gofast.mlops.metadata import MetadataManager
    >>> manager = MetadataManager(
    ...     metadata_store='aws',
    ...     cloud_bucket_name='my-metadata-bucket',
    ...     encryption_key='my-secret-key'
    ... )
    >>> manager.store_metadata('model_version', '1.0.0')
    >>> version = manager.get_metadata('model_version')
    >>> print(version)

    See Also
    --------
    boto3.client : AWS SDK for Python.
    google.cloud.storage.Client : Google Cloud Storage client.

    References
    ----------
    .. [1] AWS S3 Documentation: https://aws.amazon.com/s3/
    .. [2] Google Cloud Storage Documentation:
       https://cloud.google.com/storage/docs/

    """

    @validate_params(
        {
            "metadata_store": [StrOptions({"aws", "gcp", "local"})],
            "schema": [dict, None],
            "local_backup_path": [str],
            "versioning": [bool],
            "encryption_key": [str, None],
            "compression_enabled": [bool],
            "retry_policy": [dict, None],
            "cloud_sync_frequency": [Interval(Integral, 1, None, closed="left")],
            "auto_load_backup": [bool],
            "cache_enabled": [bool],
            "cloud_bucket_name": [str, None],
        }
    )
    def __init__(
        self,
        metadata_store: str,
        schema: Optional[Dict[str, Any]] = None,
        local_backup_path: str = "metadata_backup.json",
        versioning: bool = True,
        encryption_key: Optional[str] = None,
        compression_enabled: bool = False,
        retry_policy: Optional[Dict[str, int]] = None,
        cloud_sync_frequency: int = 5,
        auto_load_backup: bool = True,
        cache_enabled: bool = True,
        cloud_bucket_name: Optional[str] = None,
    ):
        """
        Initializes the MetadataManager with enhanced options.

        Parameters
        ----------
        metadata_store : {'aws', 'gcp', 'local'}
            The type of metadata store to synchronize with.
        schema : dict, optional
            Schema to enforce consistency in metadata storage. If
            ``None``, a default schema is used.
        local_backup_path : str, default='metadata_backup.json'
            Local path to save metadata as a backup in case cloud sync
            fails.
        versioning : bool, default=True
            Enable versioning for metadata entries.
        encryption_key : str or None, default=None
            Key to encrypt metadata when saving to local or cloud
            storage.
        compression_enabled : bool, default=False
            Enable compression for large metadata objects.
        retry_policy : dict, optional
            Retry policy for cloud sync. Should contain ``'retries'``
            and ``'backoff'`` keys.
        cloud_sync_frequency : int, default=5
            Number of changes before triggering automatic cloud sync.
        auto_load_backup : bool, default=True
            Automatically load local metadata backup on initialization.
        cache_enabled : bool, default=True
            Enable in-memory caching for frequently accessed metadata.
        cloud_bucket_name : str or None, default=None
            Name of the cloud bucket for storing metadata. Required for
            cloud sync with AWS or GCP.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.

        Notes
        -----
        The class supports encryption using the ``cryptography`` package
        and compression using the ``zlib`` module.

        """
        self.metadata_store = metadata_store
        self.schema = schema or self._default_schema()
        self.metadata = {}
        self.local_backup_path = local_backup_path
        self.versioning = versioning
        self.compression_enabled = compression_enabled
        self.retry_policy = retry_policy or {"retries": 3, "backoff": 2}
        self.cloud_sync_frequency = cloud_sync_frequency
        self.encryption_key = encryption_key
        self.cache_enabled = cache_enabled
        self.change_count = 0  # Tracks number of changes for auto sync
        self.cloud_bucket_name = cloud_bucket_name

        if self.metadata_store in {"aws", "gcp"} and not self.cloud_bucket_name:
            raise ValueError(
                "cloud_bucket_name is required when metadata_store "
                "is 'aws' or 'gcp'."
            )

        # Optional in-memory cache for frequently accessed metadata
        self.cache = {} if self.cache_enabled else None

        # Initialize encryption if key is provided
        if self.encryption_key:
            self._initialize_encryption()

        # Auto load from backup
        if auto_load_backup:
            self._load_local_backup()

    def store_metadata(self, key: str, value: Any):
        """
        Stores metadata with the given key and value, with schema
        validation and optional versioning.

        Parameters
        ----------
        key : str
            The metadata key (e.g., ``'model_version'``,
            ``'data_version'``).
        value : Any
            The metadata value to store.

        Raises
        ------
        ValueError
            If the key does not match the schema.

        Notes
        -----
        If versioning is enabled, previous versions of the metadata are
        stored with versioned keys.

        """
        if key not in self.schema:
            raise ValueError(f"Metadata key '{key}' does not match the schema.")

        # Versioning: Store previous version if enabled
        if self.versioning and key in self.metadata:
            previous_versions = [
                k for k in self.metadata.keys() if k.startswith(f"{key}_v")
            ]
            version_number = len(previous_versions) + 1
            previous_version_key = f"{key}_v{version_number}"
            self.metadata[previous_version_key] = self.metadata[key]
            logger.info(
                f"Stored previous version of '{key}' as '{previous_version_key}'."
            )

        # Update the metadata
        self.metadata[key] = value
        logger.info(f"Stored metadata: '{key}' -> {value}")

        # Save a local backup
        self._save_local_backup()

        # Update in-memory cache if enabled
        if self.cache_enabled:
            self.cache[key] = value

        # Auto-sync with cloud if changes exceed sync frequency
        self.change_count += 1
        if self.change_count >= self.cloud_sync_frequency:
            self.sync_with_cloud()
            self.change_count = 0

    def get_metadata(self, key: str) -> Any:
        """
        Retrieves metadata by key, with optional cache support.

        Parameters
        ----------
        key : str
            The metadata key.

        Returns
        -------
        value : Any
            The stored metadata value or ``None`` if the key is not
            found.

        Notes
        -----
        If caching is enabled and the key is in the cache, the value is
        retrieved from the cache.

        """
        if self.cache_enabled and key in self.cache:
            logger.info(f"Fetching '{key}' from cache.")
            return self.cache[key]
        return self.metadata.get(key, None)

    def _default_schema(self) -> Dict[str, Any]:
        """
        Returns the default schema for metadata storage.

        Returns
        -------
        schema : dict
            Default schema for metadata.

        Notes
        -----
        The default schema includes keys like ``'model_version'``,
        ``'data_version'``, etc.

        """
        return {
            "model_version": str,
            "data_version": str,
            "training_params": dict,
            "evaluation_metrics": dict,
            "deployment_config": dict,
            "experiment_id": str,
        }

    def _initialize_encryption(self):
        """
        Initializes the encryption cipher using the provided encryption
        key.

        Raises
        ------
        ValueError
            If the encryption key is invalid.

        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography``
        package for symmetric encryption.

        """
        try:
            from cryptography.fernet import Fernet

            self.cipher = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _compress_metadata(self, metadata: Dict[str, Any]) -> bytes:
        """
        Compresses metadata using zlib.

        Parameters
        ----------
        metadata : dict
            The metadata to compress.

        Returns
        -------
        compressed_data : bytes
            Compressed metadata.

        Notes
        -----
        The metadata is serialized to JSON and then compressed using
        zlib.

        """
        logger.info("Compressing metadata before storage.")
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        return zlib.compress(metadata_bytes)

    def _encrypt_metadata(self, metadata: bytes) -> bytes:
        """
        Encrypts metadata using the provided encryption key.

        Parameters
        ----------
        metadata : bytes
            The metadata to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted metadata.

        Notes
        -----
        If encryption is not enabled, the original metadata is returned.

        """
        if not self.cipher:
            return metadata
        logger.info("Encrypting metadata before storage.")
        return self.cipher.encrypt(metadata)

    def _save_local_backup(self):
        """
        Saves the metadata to a local backup file in case cloud sync
        fails, with optional compression and encryption.

        Notes
        -----
        The metadata is saved to the path specified by
        ``local_backup_path``. Compression and encryption are applied
        if enabled.

        Raises
        ------
        Exception
            If an error occurs during saving.

        """
        try:
            metadata_to_save = self.metadata

            # Compress metadata if enabled
            if self.compression_enabled:
                metadata_to_save = self._compress_metadata(metadata_to_save)
            else:
                metadata_to_save = json.dumps(metadata_to_save).encode("utf-8")

            # Encrypt metadata if enabled
            if self.encryption_key:
                metadata_to_save = self._encrypt_metadata(metadata_to_save)

            write_mode = (
                "wb" if self.encryption_key or self.compression_enabled else "w"
            )
            with open(self.local_backup_path, write_mode) as backup_file:
                backup_file.write(metadata_to_save)
            logger.info(
                f"Metadata backup saved locally at '{self.local_backup_path}'."
            )
        except Exception as e:
            logger.error(f"Failed to save local backup: {e}")

    def sync_with_cloud(self):
        """
        Synchronizes metadata with the cloud metadata store (AWS, GCP,
        etc.) with retry policy.

        Notes
        -----
        The method attempts to synchronize the metadata with the
        specified cloud store. If it fails, it retries according to the
        ``retry_policy``.

        Raises
        ------
        Exception
            If synchronization fails after all retries.

        """
        retries = 0
        success = False
        while retries < self.retry_policy["retries"] and not success:
            try:
                if self.metadata_store == "aws":
                    self._sync_aws()
                elif self.metadata_store == "gcp":
                    self._sync_gcp()
                else:
                    raise ValueError(
                        f"Unsupported metadata store: '{self.metadata_store}'"
                    )
                success = True
            except Exception as e:
                retries += 1
                logger.error(f"Failed to sync with cloud (Attempt {retries}): {e}")
                time.sleep(self.retry_policy["backoff"])
                if retries >= self.retry_policy["retries"]:
                    logger.error("Max retry attempts reached. Sync failed.")
                    self._save_local_backup()
                    raise

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for AWS cloud synchronization.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _sync_aws(self):
        """
        Syncs metadata with AWS Metadata Store (e.g., S3).

        Raises
        ------
        Exception
            If an error occurs during synchronization.

        Notes
        -----
        The metadata is uploaded to an S3 bucket specified by
        ``cloud_bucket_name`` under the key ``'metadata/metadata.json'``.

        """
        try:
            import boto3

            s3_client = boto3.client("s3")
            s3_client.put_object(
                Body=json.dumps(self.metadata),
                Bucket=self.cloud_bucket_name,
                Key="metadata/metadata.json",
            )
            logger.info("Metadata successfully synced to AWS S3.")
        except Exception as e:
            logger.error(f"Error syncing with AWS S3: {e}")
            raise

    @ensure_pkg(
        "google.cloud.storage",
        extra=(
            "The 'google-cloud-storage' package is required for GCP cloud "
            "synchronization."
        ),
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _sync_gcp(self):
        """
        Syncs metadata with GCP Metadata Store (e.g., Cloud Storage).

        Raises
        ------
        Exception
            If an error occurs during synchronization.

        Notes
        -----
        The metadata is uploaded to a Cloud Storage bucket specified by
        ``cloud_bucket_name`` under the blob ``'metadata/metadata.json'``.

        """
        try:
            from google.cloud import storage

            storage_client = storage.Client()
            bucket = storage_client.bucket(self.cloud_bucket_name)
            blob = bucket.blob("metadata/metadata.json")
            blob.upload_from_string(
                json.dumps(self.metadata), content_type="application/json"
            )
            logger.info("Metadata successfully synced to GCP Cloud Storage.")
        except Exception as e:
            logger.error(f"Error syncing with GCP Cloud Storage: {e}")
            raise

    def _load_local_backup(self):
        """
        Loads metadata from a local backup file if available, with
        optional decryption and decompression.

        Notes
        -----
        The method attempts to load the metadata from the path specified
        by ``local_backup_path``. Decryption and decompression are
        applied if enabled.

        Raises
        ------
        Exception
            If an error occurs during loading.

        """
        if not os.path.exists(self.local_backup_path):
            logger.warning("No local backup found.")
            return
        try:
            read_mode = (
                "rb" if self.encryption_key or self.compression_enabled else "r"
            )
            with open(self.local_backup_path, read_mode) as backup_file:
                data = backup_file.read()

                # Decrypt metadata if encryption is enabled
                if self.encryption_key:
                    data = self.cipher.decrypt(data)

                # Decompress metadata if compression is enabled
                if self.compression_enabled:
                    data = zlib.decompress(data)

                # Load metadata from JSON
                self.metadata = json.loads(data.decode("utf-8"))
            logger.info("Metadata loaded from local backup.")
        except Exception as e:
            logger.error(f"Failed to load local backup: {e}")

     
class LineageTracker(BaseClass):
    """
    Tracks the entire lineage of machine learning models, including raw
    data, transformations, hyperparameters, environment configurations,
    dependencies, and model deployment histories.

    Parameters
    ----------
    versioning : bool, default=True
        Enable version control for lineage tracking.
    lineage_store : str or None, default=None
        Path to store the lineage logs. Can be a local file path or an
        S3 bucket path (e.g., 's3://my-bucket/lineage.json').
    compression_enabled : bool, default=False
        Enable compression for lineage data using zlib.
    encryption_key : str or None, default=None
        Key to encrypt lineage data when persisting. Should be a valid
        Fernet key.
    cloud_sync_enabled : bool, default=True
        Enable automatic cloud synchronization of lineage data.
    alert_on_version_change : bool, default=False
        Enable alerts on version changes in data or models.
    max_log_size : int, default=1000
        Maximum number of lineage entries before triggering a sync.
    external_metadata_manager : object or None, default=None
        External metadata manager for richer lineage tracking. Should
        implement a ``store_metadata`` method.
    retry_policy : dict or None, default=None
        Retry policy for cloud sync. Should contain ``'retries'`` and
        ``'backoff'`` keys. Defaults to ``{'retries': 3, 'backoff': 2}``.
    tagging_enabled : bool, default=True
        Enable tagging of lineage entries for easy filtering.

    Notes
    -----
    The :class:`LineageTracker` class provides comprehensive tracking of
    the lineage of machine learning models, including data versions,
    transformations, hyperparameters, and deployment histories. It
    supports versioning, encryption, compression, cloud synchronization,
    and alerting mechanisms.

    Examples
    --------
    >>> from gofast.mlops.metadata import LineageTracker
    >>> tracker = LineageTracker(
    ...     lineage_store='s3://my-bucket/lineage.json',
    ...     encryption_key='my-secret-key',  # Should be a valid Fernet key
    ...     compression_enabled=True,
    ...     alert_on_version_change=True
    ... )
    >>> tracker.log_data_ingestion(
    ...     data_version='v1.0.0',
    ...     source='s3://data-bucket/dataset.csv',
    ...     dependencies=['s3://data-bucket/features.csv'],
    ...     tags=['experiment1']
    ... )
    >>> tracker.log_model_training(
    ...     model_version='v1.0.0',
    ...     hyperparameters={'learning_rate': 0.01},
    ...     environment={'python_version': '3.8'},
    ...     tags=['experiment1']
    ... )
    >>> tracker.log_deployment(
    ...     model_version='v1.0.0',
    ...     deployment_time='2023-10-01 12:00:00',
    ...     environment={'cloud_provider': 'AWS'},
    ...     access_permissions={'deployed_by': 'user1'},
    ...     tags=['production']
    ... )

    See Also
    --------
    MetadataManager : For managing metadata of models and data.
    ExperimentTracker : For tracking experiments and runs.

    References
    ----------
    .. [1] "Data Lineage for Machine Learning", MLflow Documentation.
       https://mlflow.org/docs/latest/lineage.html
    .. [2] "Tracking Model Lineage with Amazon SageMaker",
       AWS Machine Learning Blog.
       https://aws.amazon.com/blogs/machine-learning/tracking-model-lineage-with-amazon-sagemaker/

    """

    @validate_params({
        'versioning': [bool],
        'lineage_store': [str, None],
        'compression_enabled': [bool],
        'encryption_key': [str, None],
        'cloud_sync_enabled': [bool],
        'alert_on_version_change': [bool],
        'max_log_size': [Interval(Integral, 1, None, closed='left')],
        'external_metadata_manager': [HasMethods(['store_metadata']), None],
        'retry_policy': [dict, None],
        'tagging_enabled': [bool],
    })
    def __init__(
        self,
        versioning: bool = True,
        lineage_store: Optional[str] = None,
        compression_enabled: bool = False,
        encryption_key: Optional[str] = None,
        cloud_sync_enabled: bool = True,
        alert_on_version_change: bool = False,
        max_log_size: int = 1000,
        external_metadata_manager: Optional[Any] = None,
        retry_policy: Optional[Dict[str, int]] = None,
        tagging_enabled: bool = True,
    ):
        """
        Initializes the LineageTracker with enhanced options.

        Parameters
        ----------
        versioning : bool, default=True
            Enable version control for lineage tracking.
        lineage_store : str or None, default=None
            Path to store the lineage logs. Can be a local file path or an
            S3 bucket path (e.g., 's3://my-bucket/lineage.json').
        compression_enabled : bool, default=False
            Enable compression for lineage data using zlib.
        encryption_key : str or None, default=None
            Key to encrypt lineage data when persisting. Should be a valid
            Fernet key.
        cloud_sync_enabled : bool, default=True
            Enable automatic cloud synchronization of lineage data.
        alert_on_version_change : bool, default=False
            Enable alerts on version changes in data or models.
        max_log_size : int, default=1000
            Maximum number of lineage entries before triggering a sync.
        external_metadata_manager : object or None, default=None
            External metadata manager for richer lineage tracking. Should
            implement a ``store_metadata`` method.
        retry_policy : dict or None, default=None
            Retry policy for cloud sync. Should contain ``'retries'`` and
            ``'backoff'`` keys. Defaults to ``{'retries': 3, 'backoff': 2}``.
        tagging_enabled : bool, default=True
            Enable tagging of lineage entries for easy filtering.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.

        Notes
        -----
        The class supports encryption using the ``cryptography`` package
        and compression using the ``zlib`` module.

        """
        self.lineage = []
        self.versioning = versioning
        self.lineage_store = lineage_store
        self.compression_enabled = compression_enabled
        self.encryption_key = encryption_key
        self.cloud_sync_enabled = cloud_sync_enabled
        self.alert_on_version_change = alert_on_version_change
        self.max_log_size = max_log_size
        self.external_metadata_manager = external_metadata_manager
        self.retry_policy = retry_policy or {'retries': 3, 'backoff': 2}
        self.tagging_enabled = tagging_enabled

        # Initialize encryption if key is provided
        if self.encryption_key:
            self._initialize_encryption()

        # Validate that external_metadata_manager has required methods
        if self.external_metadata_manager and not hasattr(
            self.external_metadata_manager, 'store_metadata'
        ):
            raise ValueError(
                "external_metadata_manager must implement a 'store_metadata' method."
            )
            
    def record_lineage(self, metadata_type: str, lineage_info: Dict[str, Any]):
        """
        Records lineage information for a specific metadata type.

        Parameters
        ----------
        metadata_type : str
            The type of metadata (e.g., 'model', 'dataset').
        lineage_info : dict
            The lineage information to record.

        Notes
        -----
        This method allows recording custom lineage information associated
        with a metadata type, which can be retrieved later.

        Examples
        --------
        >>> tracker.record_lineage('model', {'parent_model': 'model_v1'})
        """
        log_entry = {
            'metadata_type': metadata_type,
            'lineage_info': lineage_info,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.lineage.append(log_entry)
        logger.info(f"Recorded lineage for metadata type '{metadata_type}'.")

        # Persist the lineage if necessary
        self._persist_lineage()

    def retrieve_lineage(self, metadata_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the lineage information for a specific metadata type.

        Parameters
        ----------
        metadata_type : str
            The type of metadata whose lineage is to be retrieved.

        Returns
        -------
        lineage_info : dict or None
            The lineage information if found, otherwise None.

        Notes
        -----
        This method searches the lineage entries for the specified metadata
        type and returns the associated lineage information.

        Examples
        --------
        >>> lineage_info = tracker.retrieve_lineage('model')
        """
        for entry in reversed(self.lineage):
            if entry.get('metadata_type') == metadata_type:
                return entry.get('lineage_info')
        logger.warning(
            f"No lineage information found for metadata type '{metadata_type}'.")
        return None

    def log_data_ingestion(
        self,
        data_version: str,
        source: str,
        dependencies: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Logs the data ingestion process, capturing data version, source,
        dependencies, and optional tags.

        Parameters
        ----------
        data_version : str
            The version of the dataset.
        source : str
            The data source (e.g., 's3://bucket/dataset.csv').
        dependencies : list of str or None, default=None
            Dependencies between different data sources (e.g., feature datasets).
        tags : list of str or None, default=None
            Optional tags for easy filtering of lineage entries.

        Notes
        -----
        This method logs details about data ingestion, which is crucial
        for tracing the origin of the data used in model training.

        Examples
        --------
        >>> tracker.log_data_ingestion(
        ...     data_version='v1.0.0',
        ...     source='s3://data-bucket/dataset.csv',
        ...     dependencies=['s3://data-bucket/features.csv'],
        ...     tags=['experiment1']
        ... )

        """
        log_entry = {
            'stage': 'data_ingestion',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_version': data_version,
            'source': source,
            'dependencies': dependencies or [],
            'tags': tags if self.tagging_enabled else []
        }

        if self.versioning:
            self._add_version_info(log_entry)

        self.lineage.append(log_entry)
        logger.info(f"Logged data ingestion: {data_version} from {source}")

        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata('data_version', data_version)

        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()

        if self.alert_on_version_change:
            self._send_alert(f"Data version changed: {data_version}")

        self._persist_lineage()

    def log_model_training(
        self,
        model_version: str,
        hyperparameters: Dict[str, Any],
        environment: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ):
        """
        Logs the model training process, capturing the model version,
        hyperparameters, environment, and optional tags.

        Parameters
        ----------
        model_version : str
            Version of the trained model.
        hyperparameters : dict
            Hyperparameter configurations used during training.
        environment : dict
            Environment details (e.g., Python version, library versions).
        tags : list of str or None, default=None
            Optional tags for easy filtering of lineage entries.

        Notes
        -----
        This method logs details about model training, including
        hyperparameters and environment configurations, which are vital
        for reproducibility.

        Examples
        --------
        >>> tracker.log_model_training(
        ...     model_version='v1.0.0',
        ...     hyperparameters={'learning_rate': 0.01},
        ...     environment={'python_version': '3.8'},
        ...     tags=['experiment1']
        ... )

        """
        log_entry = {
            'stage': 'model_training',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': model_version,
            'hyperparameters': hyperparameters,
            'environment': environment,
            'tags': tags if self.tagging_enabled else []
        }

        if self.versioning:
            self._add_version_info(log_entry)

        self.lineage.append(log_entry)
        logger.info(f"Logged model training: {model_version} with hyperparameters {hyperparameters}")

        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata('model_version', model_version)

        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()

        if self.alert_on_version_change:
            self._send_alert(f"Model version changed: {model_version}")

        self._persist_lineage()

    def log_deployment(
        self,
        model_version: str,
        deployment_time: str,
        environment: Dict[str, Any],
        access_permissions: Dict[str, str],
        tags: Optional[List[str]] = None,
    ):
        """
        Logs the deployment of a model, capturing the model version,
        deployment time, environment, access permissions, and optional
        tags.

        Parameters
        ----------
        model_version : str
            The version of the deployed model.
        deployment_time : str
            Timestamp of deployment in format '%Y-%m-%d %H:%M:%S'.
        environment : dict
            Deployment environment details (e.g., cloud service, resources used).
        access_permissions : dict
            Information on access control (e.g., who deployed, access level).
        tags : list of str or None, default=None
            Optional tags for easy filtering of lineage entries.

        Notes
        -----
        This method logs details about model deployment, which is important
        for tracking model versions in production environments.

        Examples
        --------
        >>> tracker.log_deployment(
        ...     model_version='v1.0.0',
        ...     deployment_time='2023-10-01 12:00:00',
        ...     environment={'cloud_provider': 'AWS'},
        ...     access_permissions={'deployed_by': 'user1'},
        ...     tags=['production']
        ... )

        """
        log_entry = {
            'stage': 'deployment',
            'timestamp': deployment_time,
            'model_version': model_version,
            'environment': environment,
            'access_permissions': access_permissions,
            'tags': tags if self.tagging_enabled else []
        }

        if self.versioning:
            self._add_version_info(log_entry)

        self.lineage.append(log_entry)
        logger.info(f"Logged model deployment: {model_version} at {deployment_time}")

        if self.external_metadata_manager:
            self.external_metadata_manager.store_metadata('deployment_version', model_version)

        if len(self.lineage) >= self.max_log_size:
            self._flush_lineage()

        if self.alert_on_version_change:
            self._send_alert(f"Model deployed: {model_version}")

        self._persist_lineage()

    def _flush_lineage(self):
        """
        Flushes the lineage to storage when the max log size is reached.

        Notes
        -----
        This method is called automatically when the number of lineage
        entries reaches ``max_log_size``. It persists the lineage data
        to the configured storage.

        """
        logger.info(f"Flushing lineage log after reaching max log size: {self.max_log_size}")
        self._persist_lineage()

    def _persist_lineage(self):
        """
        Persists the lineage information to external storage, if configured.
        Supports encryption and compression.

        Notes
        -----
        The lineage data is saved to the path specified by
        ``lineage_store``. Compression and encryption are applied if
        enabled. Supports saving to local files or S3 buckets.

        Raises
        ------
        Exception
            If an error occurs during persistence.

        """
        try:
            lineage_data = self.lineage

            if self.compression_enabled:
                lineage_data = self._compress_lineage(lineage_data)
            else:
                lineage_data = json.dumps(lineage_data, indent=4).encode('utf-8')

            if self.encryption_key:
                lineage_data = self._encrypt_lineage(lineage_data)

            if self.lineage_store:
                if self.lineage_store.startswith('s3://'):
                    self._save_to_s3(lineage_data)
                else:
                    write_mode = 'wb' if self.encryption_key or self.compression_enabled else 'w'
                    with open(self.lineage_store, write_mode) as f:
                        f.write(lineage_data)
                    logger.info(f"Lineage persisted to '{self.lineage_store}'")
            else:
                logger.info("Lineage persistence skipped due to missing lineage store configuration.")

        except Exception as e:
            logger.error(f"Error persisting lineage: {e}")

    def _compress_lineage(self, lineage: List[Dict[str, Any]]) -> bytes:
        """
        Compresses lineage data using zlib.

        Parameters
        ----------
        lineage : list of dict
            The lineage data to compress.

        Returns
        -------
        compressed_data : bytes
            Compressed lineage data.

        Notes
        -----
        The lineage data is serialized to JSON and then compressed using
        zlib.

        """
        logger.info("Compressing lineage data.")
        lineage_bytes = json.dumps(lineage).encode('utf-8')
        return zlib.compress(lineage_bytes)

    def _encrypt_lineage(self, lineage: bytes) -> bytes:
        """
        Encrypts lineage data using the provided encryption key.

        Parameters
        ----------
        lineage : bytes
            The lineage data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted lineage data.

        Notes
        -----
        If encryption is not enabled, the original lineage data is returned.

        """
        if not self.cipher:
            return lineage
        logger.info("Encrypting lineage data.")
        return self.cipher.encrypt(lineage)

    def _initialize_encryption(self):
        """
        Initializes the encryption cipher using the provided encryption key.

        Raises
        ------
        ValueError
            If the encryption key is invalid.

        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography``
        package for symmetric encryption.

        """
        try:
            from cryptography.fernet import Fernet

            self.cipher = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _send_alert(self, message: str):
        """
        Sends an alert (e.g., email, Slack message) when version changes occur.

        Parameters
        ----------
        message : str
            The alert message to send.

        Notes
        -----
        This method should integrate with an actual alerting system. Here,
        we implement sending an email alert using SMTP.

        """
        # Implementing an email alert system using smtplib
        import smtplib
        from email.mime.text import MIMEText

        # Configure your email settings
        smtp_server = 'smtp.example.com'
        smtp_port = 587
        sender_email = 'alert@example.com'
        receiver_email = 'admin@example.com'
        password = 'your-email-password'

        # Create the email message
        msg = MIMEText(message)
        msg['Subject'] = 'Lineage Tracker Alert'
        msg['From'] = sender_email
        msg['To'] = receiver_email

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for AWS S3 operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(self, lineage_data: bytes):
        """
        Saves lineage data to an S3 bucket.

        Parameters
        ----------
        lineage_data : bytes
            The lineage data to save.

        Notes
        -----
        The S3 path is specified by ``lineage_store``. The method extracts
        the bucket name and key from the S3 path.

        Raises
        ------
        Exception
            If an error occurs during the S3 operation.

        """
        import boto3
        try:
            s3_client = boto3.client('s3')
            bucket_name, key = self._parse_s3_path(self.lineage_store)
            s3_client.put_object(
                Body=lineage_data,
                Bucket=bucket_name,
                Key=key
            )
            logger.info(f"Lineage successfully saved to S3: '{self.lineage_store}'")
        except Exception as e:
            logger.error(f"Failed to save lineage to S3: {e}")
            raise

    def _parse_s3_path(self, s3_path: str) -> (str, str):
        """
        Parses an S3 path into bucket name and key.

        Parameters
        ----------
        s3_path : str
            The full S3 path (e.g., 's3://my-bucket/path/to/file.json').

        Returns
        -------
        bucket_name : str
            The name of the S3 bucket.
        key : str
            The S3 object key (path within the bucket).

        Notes
        -----
        This method removes the 's3://' prefix and splits the path into
        bucket name and key.

        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key

    def _add_version_info(self, log_entry: Dict[str, Any]):
        """
        Adds versioning information to each log entry, supporting version control.

        Parameters
        ----------
        log_entry : dict
            The log entry to which versioning information will be added.

        Notes
        -----
        The version is determined based on the number of entries in the
        lineage log.

        """
        # Simple versioning system based on log order
        version = f"v{len(self.lineage) + 1}"  
        log_entry['version'] = version
        logger.info(f"Added versioning to log entry: {version}")


class AuditLogger:
    """
    Logs decisions and changes during the machine learning lifecycle for
    auditability and compliance. Includes support for persistent storage,
    encryption, compression, tagging, and alerting.

    Parameters
    ----------
    storage_path : str or None, default=None
        Path for persistent storage (e.g., file path or S3 URI). If
        ``None``, logs are not persisted to storage.
    encryption_key : str or None, default=None
        Encryption key for encrypting audit logs. Should be a valid
        Fernet key.
    compression_enabled : bool, default=False
        Enable compression of logs using zlib.
    auto_archive : bool, default=True
        Automatically archive old logs based on the retention policy.
    archive_path : str or None, default=None
        Path for archived logs. Required if ``auto_archive`` is ``True``.
    retention_policy : int, default=30
        Number of days to retain logs before archiving or deletion.
    tagging_enabled : bool, default=True
        Enable tagging of logs for easy filtering.
    role_based_access_control : dict or None, default=None
        Role-based access control configuration. Should be a dictionary
        mapping roles to a list of permitted actions.
    alert_on_severity : {'low', 'medium', 'high'}, default='high'
        Alert when logs of certain severity are added.
    max_logs_before_sync : int, default=1000
        Maximum number of logs before persisting to storage.
    retry_policy : dict, default={'retries': 3, 'backoff': 2}
        Retry policy for log persistence. Should contain ``'retries'``
        and ``'backoff'`` keys.
    log_severity : bool, default=True
        Enable logging of severity levels ('low', 'medium', 'high').
    log_deletion_enabled : bool, default=True
        Enable log deletion based on the retention policy.

    Notes
    -----
    The :class:`AuditLogger` class provides a comprehensive solution for
    logging decisions and changes during the machine learning lifecycle.
    It supports features like encryption, compression, auto-archiving,
    and alerting based on severity levels.

    Examples
    --------
    >>> from gofast.mlops.metadata import AuditLogger
    >>> logger = AuditLogger(
    ...     storage_path='audit_logs.json',
    ...     encryption_key='my-secret-key',  # Should be a valid Fernet key
    ...     compression_enabled=True,
    ...     alert_on_severity='high'
    ... )
    >>> logger.log_decision(
    ...     decision='model_selection',
    ...     user='data_scientist_1',
    ...     timestamp='2024-10-10 12:00:00',
    ...     rationale='Selected model X due to better performance.',
    ...     severity='high',
    ...     tags=['modeling', 'selection']
    ... )
    >>> audit_logs = logger.get_audit_log()
    >>> print(audit_logs)

    See Also
    --------
    LineageTracker : For tracking the lineage of machine learning models.
    MetadataManager : For managing metadata of models and data.

    References
    ----------
    .. [1] "Audit Logging for Compliance", AWS Documentation.
       https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-user-guide.html
    .. [2] "Audit Logging in Machine Learning", Microsoft Azure Docs.
       https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-audit-trail

    """

    @validate_params(
        {
            'storage_path': [str, None],
            'encryption_key': [str, None],
            'compression_enabled': [bool],
            'auto_archive': [bool],
            'archive_path': [str, None],
            'retention_policy': [Interval(Integral, 1, None, closed='left')],
            'tagging_enabled': [bool],
            'role_based_access_control': [dict, None],
            'alert_on_severity': [StrOptions({'low', 'medium', 'high'})],
            'max_logs_before_sync': [Interval(Integral, 1, None, closed='left')],
            'retry_policy': [dict],
            'log_severity': [bool],
            'log_deletion_enabled': [bool],
        }
    )
    def __init__(
        self,
        storage_path: Optional[str] = None,
        encryption_key: Optional[str] = None,
        compression_enabled: bool = False,
        auto_archive: bool = True,
        archive_path: Optional[str] = None,
        retention_policy: int = 30,
        tagging_enabled: bool = True,
        role_based_access_control: Optional[Dict[str, List[str]]] = None,
        alert_on_severity: str = "high",
        max_logs_before_sync: int = 1000,
        retry_policy: Dict[str, int] = None,
        log_severity: bool = True,
        log_deletion_enabled: bool = True,
    ):
        """
        Initializes the AuditLogger with enhanced features.

        Parameters
        ----------
        storage_path : str or None, default=None
            Path for persistent storage (e.g., file path or S3 URI). If
            ``None``, logs are not persisted to storage.
        encryption_key : str or None, default=None
            Encryption key for encrypting audit logs. Should be a valid
            Fernet key.
        compression_enabled : bool, default=False
            Enable compression of logs using zlib.
        auto_archive : bool, default=True
            Automatically archive old logs based on the retention policy.
        archive_path : str or None, default=None
            Path for archived logs. Required if ``auto_archive`` is ``True``.
        retention_policy : int, default=30
            Number of days to retain logs before archiving or deletion.
        tagging_enabled : bool, default=True
            Enable tagging of logs for easy filtering.
        role_based_access_control : dict or None, default=None
            Role-based access control configuration.
        alert_on_severity : {'low', 'medium', 'high'}, default='high'
            Alert when logs of certain severity are added.
        max_logs_before_sync : int, default=1000
            Maximum number of logs before persisting to storage.
        retry_policy : dict, default={'retries': 3, 'backoff': 2}
            Retry policy for log persistence. Should contain ``'retries'``
            and ``'backoff'`` keys.
        log_severity : bool, default=True
            Enable logging of severity levels ('low', 'medium', 'high').
        log_deletion_enabled : bool, default=True
            Enable log deletion based on the retention policy.

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.

        Notes
        -----
        The class supports encryption using the ``cryptography`` package
        and compression using the ``zlib`` module.

        """
        self.logs = []
        self.storage_path = storage_path
        self.compression_enabled = compression_enabled
        self.encryption_key = encryption_key
        self.auto_archive = auto_archive
        self.archive_path = archive_path
        self.retention_policy = retention_policy
        self.tagging_enabled = tagging_enabled
        self.role_based_access_control = role_based_access_control or {}
        self.alert_on_severity = alert_on_severity
        self.max_logs_before_sync = max_logs_before_sync
        self.retry_policy = retry_policy or {'retries': 3, 'backoff': 2}
        self.log_severity = log_severity
        self.log_deletion_enabled = log_deletion_enabled

        # Initialize encryption if key is provided
        if self.encryption_key:
            self._initialize_encryption()

        if self.auto_archive and not self.archive_path:
            raise ValueError(
                "archive_path must be provided when auto_archive is enabled."
            )

    def log_decision(
        self,
        decision: str,
        user: str,
        timestamp: str,
        rationale: str,
        severity: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Logs a decision made during the ML process, including the user,
        timestamp, rationale, severity, and tags.

        Parameters
        ----------
        decision : str
            The decision made (e.g., 'model_selection').
        user : str
            The user who made the decision.
        timestamp : str
            The time when the decision was made, in format '%Y-%m-%d %H:%M:%S'.
        rationale : str
            The rationale behind the decision.
        severity : {'low', 'medium', 'high'}, optional
            The severity level of the decision.
        tags : list of str, optional
            Optional tags for filtering and organization.

        Notes
        -----
        This method logs critical decisions for auditability and compliance.
        It supports tagging and severity levels for easy filtering and alerting.

        Examples
        --------
        >>> logger.log_decision(
        ...     decision='data_cleanup',
        ...     user='data_engineer_1',
        ...     timestamp='2024-10-10 12:30:00',
        ...     rationale='Removed outliers from dataset.',
        ...     severity='medium',
        ...     tags=['data', 'cleanup']
        ... )
        """
        # Validate severity if provided
        if severity and severity not in {'low', 'medium', 'high'}:
            raise ValueError(
                "severity must be one of 'low', 'medium', or 'high'."
            )

        log_entry = {
            'decision': decision,
            'user': user,
            'timestamp': timestamp,
            'rationale': rationale,
            'severity': severity if self.log_severity else None,
            'tags': tags if self.tagging_enabled else []
        }

        self.logs.append(log_entry)
        logger.info(
            f"Logged decision: {decision} by {user} at {timestamp}, "
            f"rationale: {rationale}, severity: {severity}"
        )

        # Check for severity-based alerts
        if severity and severity.lower() == self.alert_on_severity:
            self._send_alert(f"Critical decision logged: {decision} by {user}")

        if len(self.logs) >= self.max_logs_before_sync:
            self._sync_logs()

        self._archive_logs_if_needed()

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Returns the complete audit log.

        Returns
        -------
        logs : list of dict
            A list of decision logs.

        Notes
        -----
        This method retrieves all the logs currently stored in memory.
        It does not load logs from persistent storage.

        Examples
        --------
        >>> audit_logs = logger.get_audit_log()
        >>> print(audit_logs)
        """
        return self.logs


    def log_change(self, metadata_type: str, user: str, change_description: str):
        """
        Logs a change made to the metadata.

        Parameters
        ----------
        metadata_type : str
            The type of metadata that was changed.
        user : str
            The user who made the change.
        change_description : str
            A description of the change.

        Notes
        -----
        This method records a change to the metadata, including who made
        the change and a description of the change.

        Examples
        --------
        >>> logger.log_change('model', 'test_user', 'Updated model parameters.')
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_decision(
            decision='metadata_change',
            user=user,
            timestamp=timestamp,
            rationale=change_description,
            severity='medium',
            tags=[metadata_type]
        )
        logger.info(f"Logged change to metadata type '{metadata_type}' by user '{user}'.")

    def retrieve_logs(self, metadata_type: str) -> List[Dict[str, Any]]:
        """
        Retrieves audit logs related to a specific metadata type.

        Parameters
        ----------
        metadata_type : str
            The type of metadata whose audit logs are to be retrieved.

        Returns
        -------
        logs : list of dict
            A list of audit log entries related to the specified metadata type.

        Notes
        -----
        This method filters the audit logs to return entries that are tagged
        with the specified metadata type.

        Examples
        --------
        >>> logs = logger.retrieve_logs('model')
        """
        filtered_logs = [
            log for log in self.logs
            if self.tagging_enabled and metadata_type in log.get('tags', [])
        ]
        if not filtered_logs:
            logger.warning(f"No audit logs found for metadata type '{metadata_type}'.")
        return filtered_logs

    def _initialize_encryption(self):
        """
        Initializes the encryption cipher using the provided encryption key.

        Raises
        ------
        ValueError
            If the encryption key is invalid.

        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography``
        package for symmetric encryption.

        """
        try:
            from cryptography.fernet import Fernet

            self.cipher = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _sync_logs(self):
        """
        Syncs logs to storage, with optional compression and encryption.

        Notes
        -----
        This method persists the logs to the configured storage path.
        Compression and encryption are applied if enabled.

        Raises
        ------
        Exception
            If an error occurs during log synchronization.

        """
        try:
            log_data = self.logs

            # Compress logs if enabled
            if self.compression_enabled:
                log_data = self._compress_logs(log_data)
            else:
                log_data = json.dumps(log_data, indent=4).encode('utf-8')

            # Encrypt logs if enabled
            if self.encryption_key:
                log_data = self._encrypt_logs(log_data)

            if self.storage_path:
                if self.storage_path.startswith('s3://'):
                    self._save_to_s3(log_data)
                else:
                    write_mode = 'wb' if self.encryption_key or self.compression_enabled else 'w'
                    with open(self.storage_path, write_mode) as f:
                        f.write(log_data)
                    logger.info(f"Logs successfully synced to '{self.storage_path}'")

            self.logs.clear()  # Clear logs after sync

        except Exception as e:
            logger.error(f"Error syncing logs: {e}")
            self._retry_sync_logs()

    def _retry_sync_logs(self):
        """
        Implements retry logic for syncing logs to storage.

        Notes
        -----
        The method retries log synchronization according to the
        ``retry_policy``. If all retries fail, an error is logged.

        """
        retries = 0
        while retries < self.retry_policy['retries']:
            try:
                logger.info(f"Retrying log sync (Attempt {retries + 1})...")
                self._sync_logs()
                break
            except Exception as e: # noqa
                retries += 1
                time.sleep(self.retry_policy['backoff'])
                if retries == self.retry_policy['retries']:
                    logger.error("Max retry attempts reached. Log sync failed.")

    def _archive_logs_if_needed(self):
        """
        Archives logs if auto-archiving is enabled and retention policy is met.

        Notes
        -----
        This method checks the age of the logs and archives those that
        exceed the retention policy.

        """
        if not self.auto_archive:
            return

        # Check if the archive path is set
        if not self.archive_path:
            logger.warning("Archive path not provided, skipping archiving.")
            return

        # Calculate the cutoff date for retention
        cutoff_date = datetime.now() - timedelta(days=self.retention_policy)

        # Filter logs older than the retention policy
        logs_to_archive = [
            log for log in self.logs
            if self._parse_timestamp(log['timestamp']) < cutoff_date
        ]

        if not logs_to_archive:
            logger.info("No logs to archive based on retention policy.")
            return

        # Archive the logs
        logger.info(
            f"Archiving {len(logs_to_archive)} logs older than "
            f"{self.retention_policy} days."
        )

        # Prepare the archive filename based on the current date
        archive_file_name = os.path.join(
            self.archive_path,
            f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        try:
            # Compress the logs if compression is enabled
            if self.compression_enabled:
                archived_data = self._compress_logs(logs_to_archive)
            else:
                archived_data = json.dumps(
                    logs_to_archive, indent=4
                ).encode('utf-8')

            # Encrypt the logs if encryption is enabled
            if self.encryption_key:
                archived_data = self._encrypt_logs(archived_data)

            # Write the archived logs to the archive file
            with open(archive_file_name, 'wb') as archive_file:
                archive_file.write(archived_data)

            logger.info(f"Logs successfully archived to '{archive_file_name}'.")

            # Remove archived logs from the current logs
            self.logs = [
                log for log in self.logs if log not in logs_to_archive
            ]

        except Exception as e:
            logger.error(f"Failed to archive logs: {e}")

    def _delete_old_logs(self):
        """
        Deletes logs that exceed the retention policy, if enabled.

        Notes
        -----
        This method removes logs from memory that exceed the retention
        policy.

        """
        if not self.log_deletion_enabled:
            return

        logger.info(
            f"Deleting logs older than {self.retention_policy} days."
        )

        # Calculate the cutoff date for retention
        cutoff_date = datetime.now() - timedelta(days=self.retention_policy)

        # Filter logs that exceed the retention policy
        logs_to_delete = [
            log for log in self.logs
            if self._parse_timestamp(log['timestamp']) < cutoff_date
        ]

        if not logs_to_delete:
            logger.info("No logs to delete based on retention policy.")
            return

        try:
            # Delete the logs exceeding the retention period
            self.logs = [
                log for log in self.logs if log not in logs_to_delete
            ]
            logger.info(
                f"Deleted {len(logs_to_delete)} logs that exceeded "
                "retention policy."
            )
        except Exception as e:
            logger.error(f"Failed to delete logs: {e}")

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """
        Parses a timestamp string into a datetime object.

        Parameters
        ----------
        timestamp : str
            The timestamp string (e.g., '2024-10-10 12:00:00').

        Returns
        -------
        dt : datetime
            A datetime object representing the parsed timestamp.

        Raises
        ------
        ValueError
            If the timestamp format is incorrect.

        Notes
        -----
        The expected timestamp format is '%Y-%m-%d %H:%M:%S'.

        """
        try:
            return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            logger.error(f"Error parsing timestamp: {e}")
            raise

    def _send_alert(self, message: str):
        """
        Sends an alert (e.g., email, Slack) when a critical decision is
        logged.

        Parameters
        ----------
        message : str
            The alert message to send.

        Notes
        -----
        This method should integrate with an actual alerting system.
        Here, we implement sending an email alert using SMTP.

        """
        # Implementing an email alert system using smtplib
        import smtplib
        from email.mime.text import MIMEText

        # Configure your email settings
        smtp_server = 'smtp.example.com'
        smtp_port = 587
        sender_email = 'alert@example.com'
        receiver_email = 'admin@example.com'
        password = 'your-email-password'

        # Create the email message
        msg = MIMEText(message)
        msg['Subject'] = 'Audit Logger Alert'
        msg['From'] = sender_email
        msg['To'] = receiver_email

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, password)
                server.send_message(msg)
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _compress_logs(self, logs: List[Dict[str, Any]]) -> bytes:
        """
        Compresses logs using zlib.

        Parameters
        ----------
        logs : list of dict
            The logs to compress.

        Returns
        -------
        compressed_data : bytes
            Compressed log data.

        Notes
        -----
        The logs are serialized to JSON and then compressed using zlib.

        """
        logger.info("Compressing logs.")
        log_bytes = json.dumps(logs).encode('utf-8')
        return zlib.compress(log_bytes)

    def _encrypt_logs(self, logs: bytes) -> bytes:
        """
        Encrypts logs using the provided encryption key.

        Parameters
        ----------
        logs : bytes
            The log data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted log data.

        Notes
        -----
        If encryption is not enabled, the original logs are returned.

        """
        if not self.cipher:
            return logs
        logger.info("Encrypting logs.")
        return self.cipher.encrypt(logs)

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for AWS S3 operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(self, log_data: bytes):
        """
        Saves log data to an S3 bucket.

        Parameters
        ----------
        log_data : bytes
            The log data to save.

        Notes
        -----
        The S3 path is specified by ``storage_path``. The method extracts
        the bucket name and key from the S3 path.

        Raises
        ------
        Exception
            If an error occurs during the S3 operation.

        """
        import boto3
        try:
            s3_client = boto3.client('s3')
            bucket_name, key = self._parse_s3_path(self.storage_path)
            s3_client.put_object(
                Body=log_data,
                Bucket=bucket_name,
                Key=key
            )
            logger.info(f"Logs successfully saved to S3: '{self.storage_path}'")
        except Exception as e:
            logger.error(f"Failed to save logs to S3: {e}")
            raise

    def _parse_s3_path(self, s3_path: str) -> (str, str):
        """
        Parses an S3 path into bucket name and key.

        Parameters
        ----------
        s3_path : str
            The full S3 path (e.g., 's3://my-bucket/path/to/file.json').

        Returns
        -------
        bucket_name : str
            The name of the S3 bucket.
        key : str
            The S3 object key (path within the bucket).

        Notes
        -----
        This method removes the 's3://' prefix and splits the path into
        bucket name and key.

        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key


class ReproducibilityEnsurer:
    """
    Ensures that machine learning models are reproducible by capturing environment
    details, library versions, hardware configurations, random seeds, and more.
    Facilitates the export and comparison of configurations to maintain consistency
    across different runs and deployments.

    .. math::
        \\text{Reproducibility} = f(\\text{Environment}, \\text{Library Versions}, \\text{Hardware}, \\text{Random Seeds})

    Where:
    - :math:`\\text{Environment}` includes OS details and Python version.
    - :math:`\\text{Library Versions}` includes versions of NumPy, PyTorch, TensorFlow, etc.
    - :math:`\\text{Hardware}` includes CPU and GPU details.
    - :math:`\\text{Random Seeds}` ensures deterministic behavior.

    Parameters
    ----------
    capture_hardware_info : bool, default=True
        Capture hardware details such as GPUs and CPUs.

    storage_backend : str or None, default=None
        Path for storing the configuration. Can be a local file path or an S3 URI
        (e.g., ``'s3://my-bucket/config.json'``). If ``None``, the configuration is
        not persisted to storage.

    encryption_key : str or None, default=None
        Encryption key for securing the configuration. Should be a valid Fernet key.

    compression_enabled : bool, default=False
        Enable compression of the configuration data using ``zlib``.

    versioning_enabled : bool, default=True
        Enable version control for tracking configuration changes.

    Notes
    -----
    The :class:`ReproducibilityEnsurer` class provides a comprehensive solution for
    ensuring the reproducibility of machine learning models by capturing and managing
    essential configuration details. It supports features like encryption, compression,
    versioning, and storage to facilitate consistent model training and deployment.

    Examples
    --------
    >>> from gofast.mlops.metadata import ReproducibilityEnsurer
    >>> ensurer = ReproducibilityEnsurer(
    ...     storage_backend='config.json',
    ...     encryption_key='my-fernet-key',  # Should be a valid Fernet key
    ...     compression_enabled=True,
    ...     versioning_enabled=True
    ... )
    >>> ensurer.set_random_seed(42)
    >>> ensurer.export_config()
    >>> other_env = {...}  # Some other environment configuration
    >>> differences = ensurer.compare_environments(other_env)
    >>> print(differences)

    See Also
    --------
    MetadataManager : For managing metadata of models and data.
    LineageTracker : For tracking the lineage of machine learning models.

    References
    ----------
    .. [1] "Reproducibility in Machine Learning", Towards Data Science.
       https://towardsdatascience.com/reproducibility-in-machine-learning-5b4d7c6c6c5f

    .. [2] "Ensuring Reproducibility in TensorFlow", TensorFlow Documentation.
       https://www.tensorflow.org/guide/keras/save_and_serialize#reproducibility

    """

    @validate_params({
        'capture_hardware_info': [bool],
        'storage_backend': [str, None],
        'encryption_key': [str, None],
        'compression_enabled': [bool],
        'versioning_enabled': [bool],
    })
    def __init__(
        self,
        capture_hardware_info: bool = True,
        storage_backend: Optional[str] = None,
        encryption_key: Optional[str] = None,
        compression_enabled: bool = False,
        versioning_enabled: bool = True
    ):
        """
        Initializes the ReproducibilityEnsurer with enhanced options.

        Parameters
        ----------
        capture_hardware_info : bool, default=True
            Capture hardware details such as GPUs and CPUs.

        storage_backend : str or None, default=None
            Path for storing the configuration. Can be a local file path or an
            S3 URI (e.g., ``'s3://my-bucket/config.json'``). If ``None``, the
            configuration is not persisted to storage.

        encryption_key : str or None, default=None
            Encryption key for securing the configuration. Should be a valid
            Fernet key.

        compression_enabled : bool, default=False
            Enable compression of the configuration data using ``zlib``.

        versioning_enabled : bool, default=True
            Enable version control for tracking configuration changes.

        Notes
        -----
        The class supports encryption using the ``cryptography`` package and
        compression using the ``zlib`` module. It captures environment details,
        library versions, hardware configurations, and random seeds to ensure
        reproducibility across different runs and deployments.

        """
        self.environment = self._capture_environment(capture_hardware_info)
        self.random_seed = None
        self.storage_backend = storage_backend
        self.encryption_key = encryption_key
        self.compression_enabled = compression_enabled
        self.versioning_enabled = versioning_enabled
        self.config_versions = [] if versioning_enabled else None
        self.cipher = None

        # Initialize encryption if key is provided
        if self.encryption_key:
            self._initialize_encryption()

    def set_random_seed(self, seed: int):
        """
        Sets the random seed for reproducibility across multiple libraries.

        Parameters
        ----------
        seed : int
            The random seed to use.

        Notes
        -----
        Setting the random seed ensures deterministic behavior in random operations
        across libraries like ``random``, ``numpy``, ``torch``, and ``tensorflow``.
        If PyTorch or TensorFlow are not installed, the method will skip setting
        seeds for them and log a warning.

        Examples
        --------
        >>> ensurer.set_random_seed(42)
        >>> # Now, operations like np.random.rand() will be reproducible
        >>> np.random.rand(3)
        array([0.37454012, 0.95071431, 0.73199394])

        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Set seed for PyTorch if installed
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            logger.warning("PyTorch not installed. Skipping torch.manual_seed().")

        # Set seed for TensorFlow if installed
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            logger.warning("TensorFlow not installed. Skipping tf.random.set_seed().")

        logger.info(f"Random seed set to: {seed}")

        # Update environment with the new seed
        self.environment['random_seed'] = seed

    def export_config(self, file_path: Optional[str] = None):
        """
        Exports the captured environment and configuration to a file or storage backend.

        Parameters
        ----------
        file_path : str or None, default=None
            Path to the configuration file (local or cloud). If ``None``, uses the
            ``storage_backend`` provided during initialization.

        Notes
        -----
        This method serializes the environment configuration, applies optional compression
        and encryption, and persists it to the specified storage backend or file path.
        If the storage path starts with ``'s3://'``, it will attempt to save the
        configuration to AWS S3.

        Examples
        --------
        >>> ensurer.export_config(file_path='config.json')
        >>> # Or use the storage backend set during initialization
        >>> ensurer.export_config()

        """
        config_data = self.environment.copy()

        # Handle compression
        if self.compression_enabled:
            config_data = self._compress_config(config_data)

        # Handle encryption
        if self.encryption_key:
            config_data = self._encrypt_config(config_data)

        # Determine the save path
        save_path = file_path or self.storage_backend

        if save_path:
            if save_path.startswith('s3://'):
                # Save to S3
                self._save_to_s3(config_data, s3_path=save_path)
            else:
                # Save locally
                try:
                    write_mode = 'wb' if isinstance(config_data, bytes) else 'w'
                    with open(save_path, write_mode) as f:
                        if isinstance(config_data, bytes):
                            f.write(config_data)
                        else:
                            f.write(json.dumps(config_data, indent=4))
                    logger.info(f"Configuration exported to '{save_path}'")
                except Exception as e:
                    logger.error(f"Failed to export configuration: {e}")
                    raise RuntimeError(f"Failed to export configuration: {e}")
        else:
            logger.warning("No storage backend or file path provided for configuration export.")

        # Save version history if versioning is enabled
        if self.versioning_enabled:
            self._save_version(config_data)

    def _capture_environment(self, capture_hardware_info: bool) -> Dict[str, Any]:
        """
        Captures environment details, including OS, library versions, hardware,
        and installed packages.

        Parameters
        ----------
        capture_hardware_info : bool
            Capture hardware details such as GPUs and CPUs.

        Returns
        -------
        env_info : dict
            Dictionary containing environment details.

        Notes
        -----
        This method gathers information about the operating system, Python version,
        library versions, and hardware configurations to ensure reproducibility.

        """
        env_info = {
            'python_version': platform.python_version(),
            'os': platform.system(),
            'os_version': platform.version(),
            'numpy_version': np.__version__,
            'installed_packages': self._get_installed_packages(),
            'random_seed': self.random_seed
        }

        # Try to get versions of PyTorch and TensorFlow if installed
        try:
            import torch
            env_info['torch_version'] = torch.__version__
        except ImportError:
            env_info['torch_version'] = 'Not installed'

        try:
            import tensorflow as tf
            env_info['tensorflow_version'] = tf.__version__
        except ImportError:
            env_info['tensorflow_version'] = 'Not installed'

        if capture_hardware_info:
            env_info['hardware'] = self._capture_hardware_info()

        return env_info

    def _get_installed_packages(self) -> str:
        """
        Retrieves the list of installed packages using ``pip freeze``.

        Returns
        -------
        installed_packages : str
            A string containing the list of installed packages.

        Notes
        -----
        This method uses the ``subprocess`` module to execute ``pip freeze`` and
        capture the output. If ``pip`` is not available or an error occurs, it
        returns an error message.

        """
        try:
            # Attempt to get the list of installed packages using `pip freeze`
            installed_packages = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
            return installed_packages
        except Exception as e:
            logger.error(f"Failed to capture installed packages: {e}")
            return "Error capturing installed packages"

    def _capture_hardware_info(self) -> Dict[str, Any]:
        """
        Captures hardware details, such as CPU and GPU information.

        Returns
        -------
        hardware_info : dict
            Dictionary containing CPU and GPU details.

        Notes
        -----
        This method gathers information about the CPU and GPU using the ``platform``
        and ``subprocess`` modules. If GPU information cannot be retrieved, it logs
        an error message.

        """
        hardware_info = {}
        try:
            cpu_info = platform.processor()
            hardware_info['cpu'] = cpu_info
        except Exception as e:
            logger.error(f"Error capturing CPU info: {e}")
            hardware_info['cpu'] = 'Error capturing CPU info'

        try:
            gpu_info = self._get_gpu_info()
            hardware_info['gpu'] = gpu_info
        except Exception as e:
            logger.error(f"Error capturing GPU info: {e}")
            hardware_info['gpu'] = 'Error capturing GPU info'

        return hardware_info

    def _get_gpu_info(self) -> str:
        """
        Captures GPU information using ``nvidia-smi``.

        Returns
        -------
        gpu_info : str
            String containing GPU details or an error message.

        Notes
        -----
        This method attempts to execute ``nvidia-smi`` to retrieve GPU information.
        If ``nvidia-smi`` is not available or an error occurs, it returns an error
        message.

        """
        try:
            gpu_info = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode('utf-8')
            return gpu_info
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to capture GPU info: {e.output.decode('utf-8')}")
            return "Error capturing GPU info"
        except Exception as e:
            logger.error(f"Failed to capture GPU info: {e}")
            return "Error capturing GPU info"

    def _initialize_encryption(self):
        """
        Initializes the encryption cipher using the provided encryption key.

        Raises
        ------
        ImportError
            If the ``cryptography`` package is not installed.

        ValueError
            If the encryption key is invalid.

        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography`` package for
        symmetric encryption.

        """
        try:
            from cryptography.fernet import Fernet
            self.cipher = Fernet(self.encryption_key)
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Please install it using 'pip install cryptography'."
            )
        except Exception as e:
            raise ValueError(f"Invalid encryption key: {e}")

    def _compress_config(self, config_data: Dict[str, Any]) -> bytes:
        """
        Compresses the configuration data using ``zlib``.

        Parameters
        ----------
        config_data : dict
            The configuration data to compress.

        Returns
        -------
        compressed_data : bytes
            Compressed configuration data.

        Notes
        -----
        The configuration data is serialized to JSON and then compressed using
        ``zlib`` to reduce storage size.

        """
        logger.info("Compressing configuration data.")
        config_json = json.dumps(config_data).encode('utf-8')
        return zlib.compress(config_json)

    def _encrypt_config(self, config_data: Any) -> bytes:
        """
        Encrypts the configuration data using the provided encryption key.

        Parameters
        ----------
        config_data : dict or bytes
            The configuration data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted configuration data.

        Notes
        -----
        If encryption is not enabled, the original configuration data is returned
        without modification.

        """
        if not self.cipher:
            return config_data

        logger.info("Encrypting configuration data.")
        if isinstance(config_data, bytes):
            data_to_encrypt = config_data
        else:
            data_to_encrypt = json.dumps(config_data).encode('utf-8')
        return self.cipher.encrypt(data_to_encrypt)

    def _save_version(self, config_data: Any):
        """
        Saves a version of the configuration for version control.

        Parameters
        ----------
        config_data : Any
            The configuration data to version.

        Notes
        -----
        This method appends the current configuration data along with a version
        number and timestamp to the version history.

        """
        if self.config_versions is not None:
            version_info = {
                'version': len(self.config_versions) + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'config': self.environment.copy()  # Save the unmodified environment
            }
            self.config_versions.append(version_info)
            logger.info(f"Saved configuration version {version_info['version']} at {version_info['timestamp']}")

    def compare_environments(self, other_environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares the current environment with another environment configuration.

        Parameters
        ----------
        other_environment : dict
            The environment configuration to compare against.

        Returns
        -------
        differences : dict
            Differences between the two environments.

        Notes
        -----
        This method identifies discrepancies between the current environment and
        another provided environment configuration. It is useful for diagnosing
        inconsistencies that may affect model reproducibility.

        Examples
        --------
        >>> other_env = ensurer.export_config(file_path='other_config.json')
        >>> differences = ensurer.compare_environments(other_env)
        >>> print(differences)
        {'numpy_version': {'current': '1.21.0', 'other': '1.19.5'}, ...}

        """
        differences = {}
        for key, value in self.environment.items():
            if key in other_environment and other_environment[key] != value:
                differences[key] = {'current': value, 'other': other_environment[key]}
            elif key not in other_environment:
                differences[key] = {'current': value, 'other': 'Key not present'}
        for key in other_environment:
            if key not in self.environment:
                differences[key] = {'current': 'Key not present', 'other': other_environment[key]}

        logger.info(f"Differences found between environments: {differences}")
        return differences

    @ensure_pkg(
        "boto3",
        extra="The 'boto3' package is required for AWS S3 operations.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def _save_to_s3(self, config_data: Any, s3_path: str):
        """
        Saves configuration data to an S3 bucket.

        Parameters
        ----------
        config_data : Any
            The configuration data to save.

        s3_path : str
            The full S3 path (e.g., ``'s3://my-bucket/path/to/file.json'``).

        Notes
        -----
        The S3 path is specified by ``storage_backend`` or ``file_path``. The method
        extracts the bucket name and key from the S3 path and uploads the data using
        ``boto3``.

        Raises
        ------
        RuntimeError
            If an error occurs during the S3 operation.

        """
        import boto3
        try:
            s3_client = boto3.client('s3')
            bucket_name, key = self._parse_s3_path(s3_path)
            s3_client.put_object(
                Body=config_data if isinstance(config_data, bytes) else json.dumps(config_data, indent=4),
                Bucket=bucket_name,
                Key=key
            )
            logger.info(f"Configuration successfully saved to S3: '{s3_path}'")
        except Exception as e:
            logger.error(f"Failed to save configuration to S3: {e}")
            raise RuntimeError(f"Failed to save configuration to S3: {e}") from e

    def _parse_s3_path(self, s3_path: str) -> (str, str):
        """
        Parses an S3 path into bucket name and key.

        Parameters
        ----------
        s3_path : str
            The full S3 path (e.g., ``'s3://my-bucket/path/to/file.json'``).

        Returns
        -------
        bucket_name : str
            The name of the S3 bucket.

        key : str
            The S3 object key (path within the bucket).

        Notes
        -----
        This method removes the ``'s3://'`` prefix and splits the path into bucket
        name and key.

        """
        s3_path = s3_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1)
        return bucket_name, key


class PerformanceTracker:
    """
    Tracks model performance metrics over time, capturing real-time and historical
    data for evaluation and alerting on degradation. Supports real-time monitoring,
    customizable alerting, rolling averages, compression, encryption, and visualization.
    
    .. math::
        \\text{Performance Degradation} = \\text{Previous Value} - \\text{Current Value}
    
    The class monitors metrics and alerts when the degradation exceeds a specified
    threshold. It also supports performance improvement notifications when metrics
    surpass upper bounds.
    
    Parameters
    ----------
    alert_threshold : float
        The threshold for performance degradation alerts. If the difference between
        the previous and current metric values exceeds this threshold, an alert is
        triggered.
    
    alert_method : callable or None, default=None
        Custom alert method (e.g., email, Slack). Should be a callable that accepts
        a string message. If ``None``, no alerts are sent.
    
    metrics_to_track : list of str or None, default=None
        List of metrics to track. If ``None``, defaults to ``['accuracy']``.
    
    use_rolling_avg : bool, default=False
        Use rolling average to smooth out metric fluctuations.
    
    window_size : int, default=5
        Size of the window for rolling averages. Only applicable if
        ``use_rolling_avg`` is ``True``.
    
    baseline_performance : dict of str to float or None, default=None
        Baseline performance for metrics. Used for comparison and alerting.
    
    upper_bound_performance : dict of str to float or None, default=None
        Upper bound performance for metrics. Alerts are triggered if performance
        exceeds these values.
    
    storage_backend : str or None, default=None
        Path to save performance metrics. Can be a local file path or an S3 URI
        (e.g., ``'s3://my-bucket/metrics.json'``). If ``None``, metrics are not
        persisted to storage.
    
    encryption_key : str or None, default=None
        Encryption key for securing performance data. Should be a valid Fernet key.
    
    compression_enabled : bool, default=False
        Enable compression of performance metrics using ``zlib``.
    
    real_time_monitoring : bool, default=False
        Enable real-time monitoring of metrics.
    
    real_time_interval : int, default=10
        Interval for real-time tracking in seconds.
    
    Notes
    -----
    The :class:`PerformanceTracker` class provides advanced capabilities for
    monitoring and alerting on model performance metrics. It supports features like
    rolling averages to smooth out fluctuations, customizable alerting mechanisms,
    and data persistence with optional compression and encryption.
    
    Examples
    --------
    >>> from gofast.mlops.metadata import PerformanceTracker
    >>> def my_alert_method(message):
    ...     print(f"Alert: {message}")
    ...
    >>> tracker = PerformanceTracker(
    ...     alert_threshold=0.05,
    ...     alert_method=my_alert_method,
    ...     metrics_to_track=['accuracy', 'loss'],
    ...     use_rolling_avg=True,
    ...     window_size=3,
    ...     storage_backend='performance_metrics.json',
    ...     compression_enabled=True,
    ...     encryption_key='my-fernet-key',  # Should be a valid Fernet key
    ...     real_time_monitoring=False
    ... )
    >>> tracker.log_performance('v1.0', 'accuracy', 0.95)
    >>> tracker.log_performance('v1.0', 'accuracy', 0.90)
    >>> tracker.visualize_performance('v1.0', 'accuracy')
    
    See Also
    --------
    ReproducibilityEnsurer : Ensures that machine learning models are reproducible.
    MetadataManager : Manages metadata of models and data.
    
    References
    ----------
    .. [1] "Model Performance Monitoring", MLOps Community.
       https://mlops.community/model-performance-monitoring/
    
    .. [2] "Real-Time Data Streaming", Apache Kafka Documentation.
       https://kafka.apache.org/documentation/
    
    """

    @validate_params({
        'alert_threshold': [float],
        'alert_method': [callable, None],
        'metrics_to_track': [list, None],
        'use_rolling_avg': [bool],
        'window_size': [Integral],
        'baseline_performance': [Dict[str, float], None],
        'upper_bound_performance': [Dict[str, float], None],
        'storage_backend': [str, None],
        'encryption_key': [str, None],
        'compression_enabled': [bool],
        'real_time_monitoring': [bool],
        'real_time_interval': [Integral],
    }, 
        prefer_skip_nested_validation=False
        )
    def __init__(
        self,
        alert_threshold: float,
        alert_method: Optional[Callable] = None,
        metrics_to_track: Optional[List[str]] = None,
        use_rolling_avg: bool = False,
        window_size: int = 5,
        baseline_performance: Optional[Dict[str, float]] = None,
        upper_bound_performance: Optional[Dict[str, float]] = None,
        storage_backend: Optional[str] = None,
        encryption_key: Optional[str] = None,
        compression_enabled: bool = False,
        real_time_monitoring: bool = False,
        real_time_interval: int = 10
    ):
        """
        Initializes the PerformanceTracker with advanced options.

        Parameters
        ----------
        alert_threshold : float
            The threshold for performance degradation alerts.

        alert_method : callable or None, default=None
            Custom alert method (e.g., email, Slack). Should be a callable that
            accepts a string message.

        metrics_to_track : list of str or None, default=None
            List of metrics to track. Defaults to ``['accuracy']`` if ``None``.

        use_rolling_avg : bool, default=False
            Use rolling average to smooth out metric fluctuations.

        window_size : int, default=5
            Size of the window for rolling averages.

        baseline_performance : dict of str to float or None, default=None
            Baseline performance for metrics.

        upper_bound_performance : dict of str to float or None, default=None
            Upper bound performance for metrics.

        storage_backend : str or None, default=None
            Path to save performance metrics. Can be a local file path or an S3
            URI.

        encryption_key : str or None, default=None
            Encryption key for securing performance data.

        compression_enabled : bool, default=False
            Enable compression of performance metrics using ``zlib``.

        real_time_monitoring : bool, default=False
            Enable real-time monitoring of metrics.

        real_time_interval : int, default=10
            Interval for real-time tracking in seconds.

        Raises
        ------
        ValueError
            If invalid parameters are provided.

        Notes
        -----
        If ``encryption_key`` is provided, the class uses the ``cryptography``
        package for data encryption. Ensure that a valid Fernet key is used.

        """
        self.performance_metrics: Dict[str, Dict[str, List[float]]] = {}
        self.alert_threshold = alert_threshold
        self.alert_method = alert_method
        self.metrics_to_track = metrics_to_track or ['accuracy']
        self.use_rolling_avg = use_rolling_avg
        self.window_size = window_size
        self.rolling_window = {
            metric: deque(maxlen=window_size) for metric in self.metrics_to_track
        }
        self.baseline_performance = baseline_performance or {}
        self.upper_bound_performance = upper_bound_performance or {}
        self.storage_backend = storage_backend
        self.compression_enabled = compression_enabled
        self.encryption_key = encryption_key
        self.real_time_monitoring = real_time_monitoring
        self.real_time_interval = real_time_interval
        self.cipher = None

        # Initialize encryption if key is provided
        if self.encryption_key:
            try:
                from cryptography.fernet import Fernet
                self.cipher = Fernet(self.encryption_key)
            except ImportError:
                raise ImportError(
                    "The 'cryptography' package is required for encryption. "
                    "Please install it using 'pip install cryptography'."
                )
            except Exception as e:
                raise ValueError(f"Invalid encryption key: {e}")

        # Start real-time monitoring if enabled
        if self.real_time_monitoring:
            self._start_real_time_monitoring()

    def log_performance(self, model_version: str, metric: str, value: float):
        """
        Logs the performance of a model for a given metric.

        Parameters
        ----------
        model_version : str
            The version of the model.

        metric : str
            The performance metric (e.g., ``'accuracy'``).

        value : float
            The value of the metric.

        Notes
        -----
        If rolling average is enabled, the method computes the average over the
        specified window size before logging the metric.

        Examples
        --------
        >>> tracker.log_performance('v1.0', 'accuracy', 0.92)

        """
        if model_version not in self.performance_metrics:
            self.performance_metrics[model_version] = {}
        if metric not in self.performance_metrics[model_version]:
            self.performance_metrics[model_version][metric] = []

        # Apply rolling average if enabled
        if self.use_rolling_avg:
            self.rolling_window[metric].append(value)
            value_smoothed = np.mean(self.rolling_window[metric])
            logger.info(f"Using rolling average for {metric}: {value_smoothed}")
            self.performance_metrics[model_version][metric].append(value_smoothed)
        else:
            self.performance_metrics[model_version][metric].append(value)

        logger.info(f"Logged performance for model {model_version}: {metric} -> {value}")

        # Check for degradation or improvement
        self._check_performance_change(model_version, metric)

        # Persist performance metrics if storage backend is provided
        if self.storage_backend:
            self._persist_metrics()

    def _check_performance_change(self, model_version: str, metric: str):
        """
        Checks if the current performance has degraded or improved based on thresholds.

        Parameters
        ----------
        model_version : str
            The version of the model.

        metric : str
            The performance metric to compare.

        Notes
        -----
        The method compares the latest metric value with the previous one and
        triggers alerts if performance degradation exceeds the ``alert_threshold``.
        It also checks against upper bound performance for significant improvements.

        """
        values = self.performance_metrics[model_version][metric]
        current_value = values[-1]
        previous_value = values[-2] if len(values) > 1 else current_value

        # Check for performance degradation
        if previous_value - current_value > self.alert_threshold:
            logger.warning(
                f"Performance degradation detected in model {model_version} for {metric}."
            )
            if self.alert_method:
                self.alert_method(
                    f"Degradation detected: {metric} in model {model_version}."
                )

        # Check for performance improvement beyond upper bound
        if metric in self.upper_bound_performance and current_value > self.upper_bound_performance[metric]:
            logger.info(
                f"Performance improvement detected in model {model_version} for {metric}."
            )
            if self.alert_method:
                self.alert_method(
                    f"Improvement detected: {metric} in model {model_version}."
                )

    def get_performance(self, model_version: str) -> Dict[str, List[float]]:
        """
        Retrieves the performance metrics for a specific model version.

        Parameters
        ----------
        model_version : str
            The version of the model.

        Returns
        -------
        performance : dict of str to list of float
            The performance metrics for the model.

        Examples
        --------
        >>> metrics = tracker.get_performance('v1.0')
        >>> print(metrics)
        {'accuracy': [0.92, 0.93], 'loss': [0.1, 0.08]}

        """
        return self.performance_metrics.get(model_version, {})

    def _persist_metrics(self):
        """
        Persists performance metrics to the storage backend with optional compression and encryption.

        Notes
        -----
        The method serializes the performance metrics to JSON, applies compression and
        encryption if enabled, and saves the data to the specified storage backend.

        Raises
        ------
        RuntimeError
            If the metrics cannot be persisted due to an error.

        """
        try:
            metrics_data = json.dumps(self.performance_metrics).encode('utf-8')

            # Compress data if compression is enabled
            if self.compression_enabled:
                metrics_data = zlib.compress(metrics_data)
                logger.info("Compressed performance metrics data.")

            # Encrypt data if encryption is enabled
            if self.encryption_key:
                metrics_data = self._encrypt_metrics(metrics_data)

            # Save to storage backend (e.g., file)
            write_mode = 'wb' if isinstance(metrics_data, bytes) else 'w'
            with open(self.storage_backend, write_mode) as f:
                if isinstance(metrics_data, bytes):
                    f.write(metrics_data)
                else:
                    f.write(metrics_data)
            logger.info(f"Performance metrics saved to {self.storage_backend}")

        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
            raise RuntimeError(f"Failed to persist metrics: {e}")

    def _encrypt_metrics(self, metrics_data: bytes) -> bytes:
        """
        Encrypts performance metrics using the provided encryption key.

        Parameters
        ----------
        metrics_data : bytes
            The metrics data to encrypt.

        Returns
        -------
        encrypted_data : bytes
            Encrypted metrics data.

        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography`` package for symmetric
        encryption.

        """
        if not self.cipher:
            return metrics_data
        logger.info("Encrypting performance metrics data.")
        return self.cipher.encrypt(metrics_data)

    def _start_real_time_monitoring(self):
        """
        Starts real-time performance monitoring with the specified interval.

        Notes
        -----
        The method starts a background thread that logs performance metrics at regular
        intervals. This is useful for real-time monitoring of models in production.

        """
        import threading

        def _monitor():
            while self.real_time_monitoring:
                time.sleep(self.real_time_interval)
                # Simulate real-time metric logging
                for model_version in self.performance_metrics.keys():
                    for metric in self.metrics_to_track:
                        current_value = np.random.rand()  # Placeholder for actual metric
                        self.log_performance(model_version, metric, current_value)

        logger.info(
            f"Starting real-time monitoring with {self.real_time_interval}-second intervals."
        )
        threading.Thread(target=_monitor, daemon=True).start()

    @ensure_pkg(
        "matplotlib",
        extra="The 'matplotlib' package is required for visualization.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA,
    )
    def visualize_performance(self, model_version: str, metric: str):
        """
        Visualizes the performance of a specific metric over time using Matplotlib.

        Parameters
        ----------
        model_version : str
            The version of the model.

        metric : str
            The performance metric to visualize.

        Notes
        -----
        The method generates a line plot showing how the metric value changes over time.
        Ensure that the ``matplotlib`` library is installed.

        Examples
        --------
        >>> tracker.visualize_performance('v1.0', 'accuracy')

        """
        import matplotlib.pyplot as plt  # Imported here due to optional dependency

        if ( 
            model_version not in self.performance_metrics 
            or metric not in self.performance_metrics[model_version]
            ):
            logger.warning(
                f"No data available for model {model_version} and metric {metric}."
            )
            return

        values = self.performance_metrics[model_version][metric]
        plt.plot(values)
        plt.title(f'Performance of {metric} for Model {model_version}')
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()

    def load_metrics(self):
        """
        Loads previously saved performance metrics from the storage backend.
        
        Notes
        -----
        This method reads the performance metrics from the specified storage
        backend, applies decryption and decompression if enabled, and updates
        the internal state of the tracker.
        
        Raises
        ------
        RuntimeError
            If the metrics cannot be loaded due to an error.
        
        Examples
        --------
        >>> tracker.load_metrics()
        
        """
        try:
            read_mode = 'rb' if self.encryption_key or self.compression_enabled else 'r'
            with open(self.storage_backend, read_mode) as f:
                metrics_data = f.read()
            
            # Decrypt data if encryption is enabled
            if self.encryption_key:
                metrics_data = self._decrypt_metrics(metrics_data)
                logger.info("Decrypted performance metrics data.")
            
            # Decompress data if compression is enabled
            if self.compression_enabled:
                metrics_data = zlib.decompress(metrics_data)
                logger.info("Decompressed performance metrics data.")
            
            # Update internal metrics
            self.performance_metrics = json.loads(metrics_data.decode('utf-8'))
            logger.info(f"Performance metrics loaded from {self.storage_backend}")
        
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            raise RuntimeError(f"Failed to load metrics: {e}")

    def _decrypt_metrics(self, metrics_data: bytes) -> bytes:
        """
        Decrypts performance metrics using the provided encryption key.
        
        Parameters
        ----------
        metrics_data : bytes
            The metrics data to decrypt.
        
        Returns
        -------
        decrypted_data : bytes
            Decrypted metrics data.
        
        Notes
        -----
        Uses the :class:`Fernet` cipher from the ``cryptography`` package for symmetric
        decryption.
        
        """
        if not self.cipher:
            return metrics_data
        logger.info("Decrypting performance metrics data.")
        return self.cipher.decrypt(metrics_data)

    def stop_real_time_monitoring(self):
        """
        Stops the real-time performance monitoring process.
        
        Notes
        -----
        This method sets the real-time monitoring flag to ``False``, which signals
        the background thread to terminate gracefully.
        
        Examples
        --------
        >>> tracker.stop_real_time_monitoring()
        
        """
        if self.real_time_monitoring:
            self.real_time_monitoring = False
            logger.info("Real-time monitoring has been stopped.")
        else:
            logger.warning("Real-time monitoring is not active.")

    def adjust_alert_threshold(self, new_threshold: float):
        """
        Adjusts the alert threshold dynamically during runtime.
        
        Parameters
        ----------
        new_threshold : float
            The new alert threshold to set.
        
        Notes
        -----
        This method allows for dynamic tuning of the alert sensitivity based on
        operational needs.
        
        Examples
        --------
        >>> tracker.adjust_alert_threshold(0.03)
        
        """
        self.alert_threshold = new_threshold
        logger.info(f"Alert threshold adjusted to {new_threshold}")

    def export_metrics(self, export_format: str = 'json',
                       destination: Optional[str] = None):
        """
        Exports performance metrics to different formats or external systems.
        
        Parameters
        ----------
        export_format : str, default='json'
            The format to export the metrics in. Options include ``'json'``, ``'csv'``.
        
        destination : str or None, default=None
            The destination path or system where the metrics will be exported.
            If ``None``, the metrics are returned as a string.
        
        Returns
        -------
        exported_data : str or None
            The exported metrics data if no destination is provided.
        
        Raises
        ------
        ValueError
            If an unsupported export format is specified.
        
        Notes
        -----
        This method supports exporting metrics in various formats, which can be
        extended to include more formats or integration with external systems.
        
        Examples
        --------
        >>> csv_data = tracker.export_metrics(export_format='csv')
        >>> print(csv_data)
        
        """
        if export_format == 'json':
            exported_data = json.dumps(self.performance_metrics, indent=4)
        elif export_format == 'csv':
            import csv
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            # Write CSV headers
            writer.writerow(['Model Version', 'Metric', 'Values'])
            # Write metric data
            for model_version, metrics in self.performance_metrics.items():
                for metric, values in metrics.items():
                    writer.writerow([model_version, metric, values])
            exported_data = output.getvalue()
        else:
            logger.error(f"Unsupported export format: {export_format}")
            raise ValueError(f"Unsupported export format: {export_format}")
        
        if destination:
            try:
                with open(destination, 'w') as f:
                    f.write(exported_data)
                logger.info(f"Metrics exported to {destination} in {export_format} format.")
            except Exception as e:
                logger.error(f"Failed to export metrics to {destination}: {e}")
                raise RuntimeError(f"Failed to export metrics: {e}")
        else:
            return exported_data

    def generate_summary(self, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates summary statistics for the tracked performance metrics.
        
        Parameters
        ----------
        model_version : str or None, default=None
            The version of the model to generate a summary for. If ``None``,
            generates a summary for all models.
        
        Returns
        -------
        summary : dict
            A dictionary containing summary statistics like mean, median, and
            standard deviation for each metric.
        
        Notes
        -----
        This method computes statistical summaries for the performance metrics,
        which can be useful for reporting and analysis.
        
        Examples
        --------
        >>> summary = tracker.generate_summary('v1.0')
        >>> print(summary)
        
        """
        import statistics
        summary = {}
        models = [model_version] if model_version else self.performance_metrics.keys()
        
        for model in models:
            if model not in self.performance_metrics:
                logger.warning(f"No data available for model {model}.")
                continue
            summary[model] = {}
            for metric, values in self.performance_metrics[model].items():
                summary_stats = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                summary[model][metric] = summary_stats
                logger.info(f"Generated summary for model {model}, metric {metric}: {summary_stats}")
        
        return summary

# ---functions ----------------

@validate_params({
    'metadata': [dict],
    'metadata_type': [str],
    'versioning_enabled': [bool],
    'version': [Integral, None],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'encryption_key': [str, None],
    'compression_enabled': [bool],
    'cloud_sync_enabled': [bool],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def log_metadata(
    metadata: Dict[str, Any],
    metadata_type: str,
    versioning_enabled: bool = True,
    version: Optional[int] = None,
    storage_backend: str = 'local',
    encryption_key: Optional[str] = None,
    compression_enabled: bool = False,
    cloud_sync_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Logs and stores metadata with options for versioning, encryption,
    compression, and cloud synchronization.

    .. math::
        \\text{Stored Metadata} = f(\\text{Metadata}, \\text{Version},
        \\text{Encryption}, \\text{Compression})

    Where:
    - :math:`\\text{Metadata}` is the input metadata dictionary.
    - :math:`\\text{Version}` is the version number if versioning is enabled.
    - :math:`\\text{Encryption}` is applied if an encryption key is provided.
    - :math:`\\text{Compression}` is applied if compression is enabled.

    Parameters
    ----------
    metadata : dict
        The metadata to be logged.

    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``, ``'experiment'``).

    versioning_enabled : bool, default=True
        Enable versioning for metadata. If ``True``, a version number is added
        to the metadata.

    version : int, optional
        Metadata version number. If ``versioning_enabled`` is ``True`` and
        ``version`` is ``None``, it defaults to ``1``.

    storage_backend : {'local', 's3', 'database'}, default='local'
        Backend for storing metadata.

        - ``'local'``: Stores metadata locally as a file.
        - ``'s3'``: Stores metadata in an AWS S3 bucket.
        - ``'database'``: Stores metadata in a MongoDB database.

    encryption_key : str, optional
        Key for encrypting metadata. Should be a valid Fernet key. If provided,
        metadata will be encrypted before storage.

    compression_enabled : bool, default=False
        Enable compression for metadata storage using ``zlib``. If ``True``,
        metadata will be compressed before storage.

    cloud_sync_enabled : bool, default=False
        Enable synchronization of metadata with cloud storage services. If
        ``True``, metadata will be synced with the specified cloud service
        after storage.

    bucket_name : str, optional
        Name of the S3 bucket where metadata will be stored. Required if
        ``storage_backend`` is ``'s3'``.

    mongo_db_uri : str, optional
        MongoDB URI for database connection. Required if ``storage_backend`` is
        ``'database'``.

    Returns
    -------
    str
        A success message indicating that the metadata was logged.

    Raises
    ------
    ValueError
        If an unsupported ``storage_backend`` is provided or required
        parameters are missing.

    RuntimeError
        If compression, encryption, or storage operations fail.

    Notes
    -----
    - The function supports encryption using the ``cryptography`` package and
      compression using the ``zlib`` module.
    - Ensure that external dependencies like ``boto3`` and ``pymongo`` are
      installed if using cloud or database storage backends.

    .. math::
        \\text{Metadata Storage Process} =
        \\text{Encrypt}(\\text{Compress}(\\text{Serialize}(\\text{Metadata})))

    Examples
    --------
    >>> from gofast.mlops.metadata import log_metadata
    >>> metadata = {
    ...     'accuracy': 0.95,
    ...     'loss': 0.05,
    ...     'epoch': 10
    ... }
    >>> message = log_metadata(
    ...     metadata=metadata,
    ...     metadata_type='model',
    ...     encryption_key='your-fernet-key',  # Should be a valid Fernet key
    ...     compression_enabled=True,
    ...     storage_backend='s3',
    ...     bucket_name='my-model-metadata-bucket'
    ... )
    >>> print(message)
    Metadata logged successfully for model (version: 1)

    See Also
    --------
    boto3.client : AWS SDK for Python, used for interacting with S3.
    cryptography.fernet.Fernet : For symmetric encryption of metadata.
    pymongo.MongoClient : MongoDB client for database operations.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "Cryptography Documentation", Cryptography.io.
       https://cryptography.io/en/latest/

    """
    # Handle versioning
    if versioning_enabled:
        if version is None:
            version = 1
        metadata['version'] = version

    # Serialize metadata to JSON string
    metadata_json = json.dumps(metadata)

    # Compress metadata if enabled
    if compression_enabled:
        try:
            logger.info("Compressing metadata.")
            metadata_bytes = zlib.compress(metadata_json.encode('utf-8'))
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Compression failed: {e}")
    else:
        metadata_bytes = metadata_json.encode('utf-8')

    # Encrypt metadata if encryption key is provided
    if encryption_key:
        try:
            logger.info("Encrypting metadata.")
            metadata_bytes = _encrypt_metadata(metadata_bytes, encryption_key)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")

    # Store metadata in the specified backend
    try:
        if storage_backend == 'local':
            _store_locally(metadata_bytes, metadata_type, versioning_enabled, version)
        elif storage_backend == 's3':
            if not bucket_name:
                raise ValueError("bucket_name must be provided when using 's3' as storage_backend.")
            _store_in_s3(metadata_bytes, metadata_type, versioning_enabled, version, bucket_name)
        elif storage_backend == 'database':
            if not mongo_db_uri:
                raise ValueError("mongo_db_uri must be provided when using 'database' as storage_backend.")
            _store_in_database(metadata_json, metadata_type, mongo_db_uri)
        else:
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
    except Exception as e:
        logger.error(f"Failed to store metadata: {e}")
        raise RuntimeError(f"Failed to store metadata: {e}")

    # Optional cloud sync
    if cloud_sync_enabled:
        _sync_with_cloud(metadata_bytes, storage_backend, metadata_type, 
                         versioning_enabled, version, bucket_name)

    return f"Metadata logged successfully for {metadata_type} (version: {version})"


def _encrypt_metadata(metadata_bytes: bytes, encryption_key: str) -> bytes:
    """
    Encrypts the metadata using the provided encryption key.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to encrypt.

    encryption_key : str
        The Fernet encryption key.

    Returns
    -------
    encrypted_data : bytes
        The encrypted metadata.

    Raises
    ------
    ImportError
        If the ``cryptography`` package is not installed.

    RuntimeError
        If encryption fails.

    Notes
    -----
    Uses the :class:`Fernet` cipher from the ``cryptography`` package for
    symmetric encryption.

    """
    try:
        from cryptography.fernet import Fernet
        cipher = Fernet(encryption_key)
        encrypted_data = cipher.encrypt(metadata_bytes)
        return encrypted_data
    except ImportError:
        raise ImportError(
            "The 'cryptography' package is required for encryption. "
            "Please install it using 'pip install cryptography'."
        )
    except Exception as e:
        logger.error(f"Error during encryption: {e}")
        raise RuntimeError(f"Encryption failed: {e}")


def _store_locally(
    metadata_bytes: bytes,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int]
):
    """
    Stores the metadata locally as a file.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store.

    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    versioning_enabled : bool
        Indicates if versioning is enabled.

    version : int or None
        Version number of the metadata.

    Raises
    ------
    RuntimeError
        If file writing fails.

    """
    file_name = f"{metadata_type}"
    if versioning_enabled and version is not None:
        file_name += f"_v{version}"
    file_name += ".json"

    try:
        with open(file_name, 'wb') as f:
            f.write(metadata_bytes)
        logger.info(f"Metadata successfully stored locally at {file_name}.")
    except Exception as e:
        logger.error(f"Error storing metadata locally: {e}")
        raise RuntimeError(f"Error storing metadata locally: {e}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_s3(
    metadata_bytes: bytes,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int],
    bucket_name: str
):
    """
    Stores the metadata in an AWS S3 bucket.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store.

    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    versioning_enabled : bool
        Indicates if versioning is enabled.

    version : int or None
        Version number of the metadata.

    bucket_name : str
        Name of the S3 bucket where metadata will be stored.

    Raises
    ------
    RuntimeError
        If S3 storage fails.

    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if versioning_enabled and version is not None:
            key += f"_v{version}"
        key += ".json"
        s3_client.put_object(Body=metadata_bytes, Bucket=bucket_name, Key=key)
        logger.info(f"Metadata successfully stored in S3 at s3://{bucket_name}/{key}.")
    except Exception as e:
        logger.error(f"Error storing metadata in S3: {e}")
        raise RuntimeError(f"Error storing metadata in S3: {e}")


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_database(metadata_json: str, metadata_type: str, mongo_db_uri: str):
    """
    Stores the metadata in a MongoDB database.

    Parameters
    ----------
    metadata_json : str
        The metadata to store as a JSON string.

    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    mongo_db_uri : str
        MongoDB URI for database connection.

    Raises
    ------
    RuntimeError
        If database storage fails.

    """
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]
        metadata_dict = json.loads(metadata_json)
        collection.insert_one(metadata_dict)
        logger.info(f"Metadata successfully stored in MongoDB collection '{metadata_type}'.")
    except Exception as e:
        logger.error(f"Error storing metadata in database: {e}")
        raise RuntimeError(f"Error storing metadata in database: {e}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _sync_with_cloud(
    metadata_bytes: bytes,
    storage_backend: str,
    metadata_type: str,
    versioning_enabled: bool,
    version: Optional[int],
    bucket_name: Optional[str]
):
    """
    Syncs metadata with a cloud storage service.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to sync.

    storage_backend : str
        The storage backend used (e.g., ``'local'``, ``'s3'``, ``'database'``).

    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    versioning_enabled : bool
        Indicates if versioning is enabled.

    version : int or None
        Version number of the metadata.

    bucket_name : str or None
        Name of the S3 bucket where metadata will be synced.

    Raises
    ------
    RuntimeError
        If cloud synchronization fails.

    Notes
    -----
    This implementation syncs the metadata to an AWS S3 bucket. Modify this
    function to sync with other cloud services as needed.

    """
    if storage_backend != 'local':
        logger.info("Cloud synchronization is only applicable for local storage backend.")
        return

    if not bucket_name:
        raise ValueError("bucket_name must be provided for cloud synchronization.")

    try:
        import boto3
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if versioning_enabled and version is not None:
            key += f"_v{version}"
        key += ".json"
        s3_client.put_object(Body=metadata_bytes, Bucket=bucket_name, Key=key)
        logger.info(f"Metadata successfully synced to cloud at s3://{bucket_name}/{key}.")
    except Exception as e:
        logger.error(f"Error during cloud synchronization: {e}")
        raise RuntimeError(f"Error during cloud synchronization: {e}")

@validate_params({
    'metadata_type': [str],
    'version': [Integral, None],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'decryption_key': [str, None],
    'decompression_enabled': [bool],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def retrieve(
    metadata_type: str,
    version: Optional[int] = None,
    storage_backend: str = 'local',
    decryption_key: Optional[str] = None,
    decompression_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves metadata based on filtering criteria, with options for decompression
    and decryption.

    .. math::
        \\text{Retrieved Metadata} = f(\\text{Storage Backend}, \\text{Metadata Type},
        \\text{Version}, \\text{Decryption}, \\text{Decompression})

    Where:
    - :math:`\\text{Storage Backend}` is the system from which metadata is retrieved
      (e.g., local storage, AWS S3, database).
    - :math:`\\text{Metadata Type}` specifies the category of metadata to retrieve
      (e.g., ``'model'``, ``'dataset'``).
    - :math:`\\text{Version}` is the specific version of the metadata to retrieve.
    - :math:`\\text{Decryption}` is applied if a decryption key is provided.
    - :math:`\\text{Decompression}` is applied if decompression is enabled.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to retrieve (e.g., ``'model'``, ``'dataset'``).

    version : int, optional
        The version of the metadata to retrieve. If ``None``, retrieves the latest
        version.

    storage_backend : {'local', 's3', 'database'}, default='local'
        Backend to retrieve metadata from.

        - ``'local'``: Retrieves metadata from the local filesystem.
        - ``'s3'``: Retrieves metadata from an AWS S3 bucket.
        - ``'database'``: Retrieves metadata from a MongoDB database.

    decryption_key : str, optional
        Key for decrypting metadata. Should be a valid Fernet key. If provided,
        metadata will be decrypted after retrieval.

    decompression_enabled : bool, default=False
        Enable decompression for metadata retrieval using ``zlib``. If ``True``,
        metadata will be decompressed after retrieval.

    bucket_name : str, optional
        Name of the S3 bucket to retrieve metadata from. Required if
        ``storage_backend`` is ``'s3'``.

    mongo_db_uri : str, optional
        MongoDB URI for database connection. Required if ``storage_backend`` is
        ``'database'``.

    Returns
    -------
    dict
        The retrieved metadata as a dictionary.

    Raises
    ------
    ValueError
        If an unsupported ``storage_backend`` is provided or required parameters
        are missing.

    FileNotFoundError
        If the specified metadata file or record does not exist.

    RuntimeError
        If decryption or decompression fails, or if JSON decoding fails.

    Notes
    -----
    - The function supports decryption using the ``cryptography`` package and
      decompression using the ``zlib`` module.
    - Ensure that external dependencies like ``boto3`` and ``pymongo`` are installed
      if using cloud or database storage backends.

    .. math::
        \\text{Decrypted Metadata} = \\text{Decrypt}(\\text{Metadata})
        \\\\
        \\text{Decompressed Metadata} = \\text{Decompress}(\\text{Metadata})

    Examples
    --------
    >>> from gofast.mlops.metadata import retrieve
    >>> # Retrieve the latest version of model metadata from local storage
    >>> metadata = retrieve(metadata_type='model')
    >>> print(metadata)
    {'accuracy': 0.95, 'loss': 0.05, 'epoch': 10, 'version': 1}

    >>> # Retrieve version 2 of dataset metadata from AWS S3 with decryption and decompression
    >>> metadata = retrieve(
    ...     metadata_type='dataset',
    ...     version=2,
    ...     storage_backend='s3',
    ...     decryption_key='your-fernet-key',
    ...     decompression_enabled=True,
    ...     bucket_name='my-dataset-metadata-bucket'
    ... )
    >>> print(metadata)
    {'data_size': '500MB', 'source': 's3://data-bucket/dataset_v2.csv', 'version': 2}

    See Also
    --------
    log_metadata : Function to log and store metadata.
    boto3.client : AWS SDK for Python, used for interacting with S3.
    cryptography.fernet.Fernet : For symmetric decryption of metadata.
    pymongo.MongoClient : For interacting with MongoDB databases.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "Cryptography Documentation", Cryptography.io.
       https://cryptography.io/en/latest/

    .. [3] "PyMongo Documentation", MongoDB.
       https://pymongo.readthedocs.io/en/stable/

    """
    metadata = None

    # Retrieve metadata from local storage
    if storage_backend == 'local':
        file_name = f"{metadata_type}"
        if version:
            file_name += f"_v{version}"
        file_name += ".json"
        try:
            with open(file_name, 'rb' if decryption_key or decompression_enabled else 'r') as f:
                metadata = f.read()
            logger.info(f"Successfully retrieved metadata from local file: {file_name}")
        except FileNotFoundError:
            logger.error(f"Metadata file {file_name} not found.")
            raise FileNotFoundError(f"Metadata file {file_name} not found.")
        except Exception as e:
            logger.error(f"Error reading local metadata file {file_name}: {e}")
            raise RuntimeError(f"Error reading local metadata file {file_name}: {e}")

    # Retrieve metadata from AWS S3
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError("bucket_name must be provided when using 's3' as storage_backend.")
        metadata = _fetch_from_s3(metadata_type, version, bucket_name)

    # Retrieve metadata from MongoDB
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError("mongo_db_uri must be provided when using 'database' as storage_backend.")
        metadata = _fetch_from_database(metadata_type, version, mongo_db_uri)

    else:
        logger.error(f"Unsupported storage backend: {storage_backend}")
        raise ValueError(f"Unsupported storage backend: {storage_backend}")

    # Optional decryption
    if decryption_key:
        try:
            from cryptography.fernet import Fernet
            cipher = Fernet(decryption_key)
            metadata = cipher.decrypt(metadata)
            logger.info("Successfully decrypted metadata.")
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for decryption. "
                "Please install it using 'pip install cryptography'."
            )
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise RuntimeError(f"Decryption failed: {e}")

    # Optional decompression
    if decompression_enabled:
        try:
            metadata = zlib.decompress(metadata)
            logger.info("Successfully decompressed metadata.")
        except zlib.error as e:
            logger.error(f"Decompression failed: {e}")
            raise RuntimeError(f"Decompression failed: {e}")

    # Convert metadata back to a dictionary
    try:
        metadata_dict = json.loads(metadata.decode('utf-8') if isinstance(metadata, bytes) else metadata)
        logger.info("Successfully decoded metadata JSON.")
        return metadata_dict
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode metadata JSON: {e}")
        raise RuntimeError(f"Failed to decode metadata JSON: {e}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _fetch_from_s3(metadata_type: str, version: Optional[int], bucket_name: str) -> bytes:
    """
    Retrieves metadata from an AWS S3 bucket.

    Parameters
    ----------
    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    version : int or None
        Version number of the metadata.

    bucket_name : str
        Name of the S3 bucket to retrieve metadata from.

    Returns
    -------
    bytes
        The retrieved metadata as bytes.

    Raises
    ------
    FileNotFoundError
        If the specified metadata file does not exist in the S3 bucket.

    RuntimeError
        If retrieval from S3 fails.

    Notes
    -----
    Uses the ``boto3`` library to interact with AWS S3.

    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"{metadata_type}/metadata"
        if version:
            key += f"_v{version}"
        key += ".json"
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        data = response['Body'].read()
        logger.info(f"Successfully retrieved metadata from AWS S3: s3://{bucket_name}/{key}")
        return data
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"Metadata file {key} not found in S3 bucket {bucket_name}.")
        raise FileNotFoundError(f"Metadata file {key} not found in S3 bucket {bucket_name}.")
    except Exception as e:
        logger.error(f"Error retrieving metadata from S3: {e}")
        raise RuntimeError(f"Error retrieving metadata from S3: {e}")


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _fetch_from_database(
        metadata_type: str, version: Optional[int], mongo_db_uri: str) -> bytes:
    """
    Retrieves metadata from a MongoDB database.

    Parameters
    ----------
    metadata_type : str
        Type of metadata (e.g., ``'model'``, ``'dataset'``).

    version : int or None
        Version number of the metadata.

    mongo_db_uri : str
        MongoDB URI for database connection.

    Returns
    -------
    bytes
        The retrieved metadata as bytes.

    Raises
    ------
    FileNotFoundError
        If the specified metadata record does not exist in the database.

    RuntimeError
        If retrieval from the database fails.

    Notes
    -----
    Uses the ``pymongo`` library to interact with MongoDB.

    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]
        query = {'version': version} if version else {}
        metadata_record = collection.find_one(query)
        if metadata_record:
            # Remove MongoDB internal fields
            metadata_record.pop('_id', None)
            metadata_json = json.dumps(metadata_record)
            logger.info(
                "Successfully retrieved metadata from "
                f"MongoDB: {metadata_type} version {version}")
            return metadata_json.encode('utf-8')
        else:
            logger.error(f"Metadata record {metadata_type}"
                         f" version {version} not found in MongoDB.")
            raise FileNotFoundError(
                f"Metadata record {metadata_type} version {version}"
                " not found in MongoDB.")
    except Exception as e:
        logger.error(f"Error retrieving metadata from MongoDB: {e}")
        raise RuntimeError(f"Error retrieving metadata from MongoDB: {e}")


@validate_params({
    'metadata_a': [dict],
    'metadata_b': [dict],
    'keys_to_compare': [list, None],
    'ignore_keys': [list, None],
    'recursive': [bool],
    'tolerance': [Real, Integral],
})
def compare(
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    keys_to_compare: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]] = None,
    recursive: bool = False,
    tolerance: float = 0.0
) -> Dict[str, Any]:
    """
    Compares two sets of metadata and highlights the differences.

    .. math::
        \\text{Differences} = \\{ k \\mid k \\in (\\text{Keys}(A) \\cup \\text{Keys}(B)),\\,
        \\text{is\_different}(A[k], B[k]) \\}

    Where:
    - :math:`A` and :math:`B` are the two metadata dictionaries.
    - :math:`k` represents a key in the metadata.
    - :math:`\\text{is\\_different}` is a function that determines if the values differ,
      considering numerical tolerance.

    Parameters
    ----------
    metadata_a : dict
        The first metadata set.

    metadata_b : dict
        The second metadata set.

    keys_to_compare : list of str, optional
        List of specific keys to compare. If ``None``, all keys are compared.

    ignore_keys : list of str, optional
        List of keys to ignore during comparison.

    recursive : bool, default=False
        If ``True``, performs a deep comparison of nested dictionaries.

    tolerance : float, default=0.0
        Numerical tolerance for comparing numeric values. Differences smaller than
        this tolerance are ignored.

    Returns
    -------
    differences : dict
        A dictionary showing the differences between the two metadata sets.
        For each differing key, it provides the value from ``metadata_a`` and
        ``metadata_b``.

    Raises
    ------
    TypeError
        If either ``metadata_a`` or ``metadata_b`` is not a dictionary.

    Notes
    -----
    - This function is useful for tracking changes between different versions of
      metadata, such as model configurations, dataset versions, or experiment 
      setups.
    - By default, it performs a shallow comparison. If ``recursive`` is set to
      ``True``, it will recursively compare nested dictionaries.
    - The ``tolerance`` parameter allows for approximate comparisons 
    of numerical values.

    Examples
    --------
    >>> from gofast.mlops.metadata import compare
    >>> metadata1 = {
    ...     'accuracy': 0.9501,
    ...     'loss': 0.05,
    ...     'epoch': 10,
    ...     'optimizer': 'adam',
    ...     'params': {'learning_rate': 0.001}
    ... }
    >>> metadata2 = {
    ...     'accuracy': 0.9502,
    ...     'loss': 0.05,
    ...     'epoch': 10,
    ...     'optimizer': 'adam',
    ...     'params': {'learning_rate': 0.0015}
    ... }
    >>> # Shallow comparison
    >>> differences = compare(metadata1, metadata2)
    >>> print(differences)
    {'accuracy': {'metadata_a': 0.9501, 'metadata_b': 0.9502},
     'params': {'metadata_a': {'learning_rate': 0.001}, 'metadata_b': {'learning_rate': 0.0015}}}

    >>> # Deep comparison with tolerance
    >>> differences = compare(metadata1, metadata2, recursive=True, tolerance=0.0005)
    >>> print(differences)
    {'params': {'learning_rate': {'metadata_a': 0.001, 'metadata_b': 0.0015}}}

    >>> # Comparing only specific keys
    >>> differences = compare(metadata1, metadata2, keys_to_compare=['epoch'])
    >>> print(differences)
    {}

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on criteria.

    References
    ----------
    .. [1] "Dictionary Operations in Python", Python Documentation.
       https://docs.python.org/3/tutorial/datastructures.html#dictionaries

    .. [2] "Deep Comparison of Dictionaries", Stack Overflow.
       https://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python

    """
    differences = _compare_metadata(
        metadata_a,
        metadata_b,
        keys_to_compare,
        ignore_keys,
        recursive,
        tolerance
    )
    return differences


def _compare_metadata(
    metadata_a: Dict[str, Any],
    metadata_b: Dict[str, Any],
    keys_to_compare: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]] = None,
    recursive: bool = False,
    tolerance: float = 0.0
) -> Dict[str, Any]:
    """
    Internal helper function to compare two metadata dictionaries.

    Parameters
    ----------
    metadata_a : dict
        The first metadata set.

    metadata_b : dict
        The second metadata set.

    keys_to_compare : list of str, optional
        List of specific keys to compare.

    ignore_keys : list of str, optional
        List of keys to ignore during comparison.

    recursive : bool, default=False
        If ``True``, performs a deep comparison of nested dictionaries.

    tolerance : float, default=0.0
        Numerical tolerance for comparing numeric values.

    Returns
    -------
    differences : dict
        A dictionary highlighting the differences between ``metadata_a``
        and ``metadata_b``.

    """
    differences = {}
    all_keys = set(metadata_a.keys()).union(set(metadata_b.keys()))

    if keys_to_compare:
        all_keys = all_keys.intersection(set(keys_to_compare))
    if ignore_keys:
        all_keys = all_keys.difference(set(ignore_keys))

    for key in all_keys:
        value_a = metadata_a.get(key, 'N/A')
        value_b = metadata_b.get(key, 'N/A')

        if isinstance(value_a, dict) and isinstance(value_b, dict) and recursive:
            nested_diff = _compare_metadata(
                value_a,
                value_b,
                keys_to_compare=None,
                ignore_keys=None,
                recursive=True,
                tolerance=tolerance
            )
            if nested_diff:
                differences[key] = nested_diff
        else:
            if not _values_are_equal(value_a, value_b, tolerance):
                differences[key] = {
                    'metadata_a': value_a,
                    'metadata_b': value_b
                }

    return differences

def _values_are_equal(value_a: Any, value_b: Any, tolerance: float) -> bool:
    """
    Determines if two values are equal, considering numerical tolerance.

    Parameters
    ----------
    value_a : Any
        The first value to compare.

    value_b : Any
        The second value to compare.

    tolerance : float
        Numerical tolerance for comparing numeric values.

    Returns
    -------
    bool
        ``True`` if values are considered equal, ``False`` otherwise.

    """
    if value_a == value_b:
        return True
    if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
        return abs(value_a - value_b) <= tolerance
    return False



@validate_params({
    'metadata_type': [str],
    'user': [str],
    'change_description': [str],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'version': [Integral, None],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def audit(
    metadata_type: str,
    user: str,
    change_description: str,
    storage_backend: str = 'local',
    version: Optional[int] = None,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Generates an audit trail for metadata changes, tracking user, timestamp,
    and changes made.

    .. math::
        \\text{Audit Log} = f(\\text{Metadata Type}, \\text{User},
        \\text{Change Description}, \\text{Timestamp}, \\text{Version})

    Where:
    - :math:`\\text{Metadata Type}` is the category of metadata being audited
      (e.g., ``'model'``, ``'dataset'``).
    - :math:`\\text{User}` is the identifier of the user who made the change.
    - :math:`\\text{Change Description}` details the modification made to the
      metadata.
    - :math:`\\text{Timestamp}` records the exact time the change was made.
    - :math:`\\text{Version}` denotes the version of the metadata after the change.

    Parameters
    ----------
    metadata_type : str
        The type of metadata (e.g., ``'model'``, ``'dataset'``).

    user : str
        The user who made the change.

    change_description : str
        Description of the change made.

    storage_backend : {'local', 's3', 'database'}, default='local'
        Backend for storing audit logs.

        - ``'local'``: Stores audit logs locally as a file.
        - ``'s3'``: Stores audit logs in an AWS S3 bucket.
        - ``'database'``: Stores audit logs in a MongoDB database.

    version : int, optional
        The version of the metadata being audited. If ``None``, versioning is
        not applied.

    bucket_name : str, optional
        Name of the S3 bucket where audit logs will be stored. Required if
        ``storage_backend`` is ``'s3'``.

    mongo_db_uri : str, optional
        MongoDB URI for database connection. Required if ``storage_backend`` is
        ``'database'``.

    Returns
    -------
    str
        A success message indicating that the audit was logged.

    Raises
    ------
    ValueError
        If an unsupported ``storage_backend`` is provided or required parameters
        are missing.

    RuntimeError
        If storage operations fail.

    Notes
    -----
    - The function supports storing audit logs locally, in AWS S3, or in a
      MongoDB database.
    - Ensure that external dependencies like ``boto3`` and ``pymongo`` are
      installed if using cloud or database storage backends.

    .. math::
        \\text{Audit Log Entry} = \\{
            \\text{"metadata\\_type"}: \\text{Metadata Type},
            \\text{"user"}: \\text{User},
            \\text{"change\\_description"}: \\text{Change Description},
            \\text{"timestamp"}: \\text{Timestamp},
            \\text{"version"}: \\text{Version}
        \\}

    Examples
    --------
    >>> from gofast.mlops.metadata import audit
    >>> message = audit(
    ...     metadata_type='model',
    ...     user='alice',
    ...     change_description='Updated model architecture to include dropout layers.',
    ...     version=2,
    ...     storage_backend='local'
    ... )
    >>> print(message)
    Audit log successfully recorded for model by alice.

    >>> message = audit(
    ...     metadata_type='dataset',
    ...     user='bob',
    ...     change_description='Added new data points to the training set.',
    ...     storage_backend='s3',
    ...     bucket_name='my-audit-log-bucket'
    ... )
    >>> print(message)
    Audit log successfully recorded for dataset by bob.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "PyMongo Documentation", MongoDB.
       https://pymongo.readthedocs.io/en/stable/

    .. [3] "Python datetime Module", Python Documentation.
       https://docs.python.org/3/library/datetime.html

    """
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create audit log entry
    audit_log = {
        'metadata_type': metadata_type,
        'user': user,
        'change_description': change_description,
        'timestamp': timestamp,
        'version': version
    }

    # Store audit log in the specified backend
    if storage_backend == 'local':
        result = _save_to_local(audit_log, metadata_type)
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError(
                "bucket_name must be provided when using 's3' as storage_backend.")
        result = _save_to_s3(audit_log, metadata_type, bucket_name)
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError(
                "mongo_db_uri must be provided when using 'database' as storage_backend.")
        result = _save_to_database(audit_log, metadata_type, mongo_db_uri)
    else:
        logger.error(f"Unsupported storage backend: {storage_backend}")
        raise ValueError(f"Unsupported storage backend: {storage_backend}")

    return result


def _save_to_local(
    audit_log: Dict[str, Any],
    metadata_type: str
) -> str:
    """
    Saves the audit log to a local JSON file.

    Parameters
    ----------
    audit_log : dict
        The audit log entry to save.

    metadata_type : str
        The type of metadata being audited.

    Returns
    -------
    str
        A success message indicating that the audit was logged locally.

    Raises
    ------
    RuntimeError
        If file writing fails.

    """
    file_name = f"{metadata_type}_audit.log"
    try:
        with open(file_name, 'a') as f:
            f.write(json.dumps(audit_log) + '\n')
        logger.info(f"Successfully recorded audit log in local file: {file_name}")
        return f"Audit log successfully recorded for {metadata_type} by {audit_log['user']}."
    except Exception as e:
        logger.error(f"Failed to write audit log to local file: {e}")
        raise RuntimeError(f"Failed to write audit log to local file: {e}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _save_to_s3(
    audit_log: Dict[str, Any],
    metadata_type: str,
    bucket_name: str
) -> str:
    """
    Saves the audit log to an AWS S3 bucket.

    Parameters
    ----------
    audit_log : dict
        The audit log entry to save.

    metadata_type : str
        The type of metadata being audited.

    bucket_name : str
        Name of the S3 bucket where audit logs will be stored.

    Returns
    -------
    str
        A success message indicating that the audit was logged in S3.

    Raises
    ------
    RuntimeError
        If writing to S3 fails.

    Notes
    -----
    Uses the ``boto3`` library to interact with AWS S3.

    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"audit_logs/{metadata_type}_audit.log"

        # Retrieve existing audit logs if they exist
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            existing_logs = response['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            existing_logs = ''
            logger.info("No existing audit log found in S3."
                        f" Creating a new log at s3://{bucket_name}/{key}.")

        # Append new audit log
        updated_logs = existing_logs + json.dumps(audit_log) + '\n'

        # Write updated logs back to S3
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=updated_logs.encode('utf-8'))
        logger.info(f"Successfully recorded audit log in S3 at s3://{bucket_name}/{key}")
        return f"Audit log successfully recorded for {metadata_type} by {audit_log['user']}."
    except Exception as e:
        logger.error(f"Failed to write audit log to S3: {e}")
        raise RuntimeError(f"Failed to write audit log to S3: {e}")

@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _save_to_database(
    audit_log: Dict[str, Any],
    metadata_type: str,
    mongo_db_uri: str
) -> str:
    """
    Saves the audit log to a MongoDB database.

    Parameters
    ----------
    audit_log : dict
        The audit log entry to save.

    metadata_type : str
        The type of metadata being audited.

    mongo_db_uri : str
        MongoDB URI for database connection.

    Returns
    -------
    str
        A success message indicating that the audit was logged in the database.

    Raises
    ------
    RuntimeError
        If writing to the database fails.

    Notes
    -----
    Uses the ``pymongo`` library to interact with MongoDB.

    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_audit_db']
        collection = db[f"{metadata_type}_audit_logs"]
        collection.insert_one(audit_log)
        logger.info(f"Successfully recorded audit log in MongoDB"
                    f" collection '{metadata_type}_audit_logs'.")
        return f"Audit log successfully recorded for {metadata_type} by {audit_log['user']}."
    except Exception as e:
        logger.error(f"Failed to write audit log to MongoDB: {e}")
        raise RuntimeError(f"Failed to write audit log to MongoDB: {e}")


@validate_params({
    'metadata': [dict],
    'cloud_provider': [StrOptions({'aws', 'gcp'})],
    'retries': [Integral],
    'batch_size': [Integral],
    'bucket_name': [str],
    'aws_credentials': [dict, None],
    'gcp_credentials': [dict, None],
})
def sync_with_cloud(
    metadata: Dict[str, Any],
    cloud_provider: str,
    retries: int = 3,
    batch_size: int = 100,
    bucket_name: str = None,
    aws_credentials: Optional[Dict[str, str]] = None,
    gcp_credentials: Optional[Dict[str, str]] = None
) -> str:
    """
    Synchronizes metadata with cloud storage (e.g., AWS S3, GCP Cloud Storage)
    with retry logic and batch updates.

    .. math::
        \\text{Sync Process} = f(\\text{Metadata}, \\text{Cloud Provider},
        \\text{Retries}, \\text{Batch Size})

    Where:
    - :math:`\\text{Metadata}` is the dictionary containing metadata to sync.
    - :math:`\\text{Cloud Provider}` specifies the target cloud platform
      (e.g., ``'aws'``, ``'gcp'``).
    - :math:`\\text{Retries}` denotes the number of retry attempts in case of failure.
    - :math:`\\text{Batch Size}` defines how many records to sync in each batch.

    Parameters
    ----------
    metadata : dict
        The metadata to synchronize.

    cloud_provider : {'aws', 'gcp'}
        The cloud provider to sync with.

        - ``'aws'``: Synchronize with AWS S3.
        - ``'gcp'``: Synchronize with Google Cloud Storage.

    retries : int, default=3
        Number of retry attempts in case of synchronization failure.

    batch_size : int, default=100
        The number of records to sync in each batch.

    bucket_name : str
        The name of the cloud storage bucket.

    aws_credentials : dict, optional
        AWS credentials for authentication. Required if ``cloud_provider`` is ``'aws'``.

        Example:
        ``{'aws_access_key_id': '...', 'aws_secret_access_key': '...'}``

    gcp_credentials : dict, optional
        GCP credentials for authentication. Required if ``cloud_provider`` is ``'gcp'``.

        Example:
        ``{'service_account_json': 'path/to/your/service_account.json'}``

    Returns
    -------
    str
        Success message if the synchronization is successful.

    Raises
    ------
    ValueError
        If an unsupported ``cloud_provider`` is provided or if required parameters
        are missing.

    RuntimeError
        If synchronization fails after the specified number of retries.

    Notes
    -----
    - The function supports synchronization with AWS S3 and Google Cloud Storage.
    - Ensure that external dependencies like ``boto3`` and
      ``google-cloud-storage`` are installed if using cloud storage backends.
    - Authentication with cloud providers must be properly configured, either via
      credentials passed as parameters or through environment variables.

    .. math::
        \\text{Synchronization} =
        \\begin{cases}
            \\text{AWS S3 Sync} & \\text{if } \\text{cloud\\_provider} = \\text{'aws'} \\\\
            \\text{GCP Storage Sync} & \\text{if } \\text{cloud\\_provider} = \\text{'gcp'}
        \\end{cases}

    Examples
    --------
    >>> from gofast.mlops.metadata import sync_with_cloud
    >>> metadata = {
    ...     'model_name': 'resnet50',
    ...     'accuracy': 0.95,
    ...     'epoch': 10,
    ...     'loss': 0.05
    ... }
    >>> # Sync metadata to AWS S3
    >>> message = sync_with_cloud(
    ...     metadata=metadata,
    ...     cloud_provider='aws',
    ...     bucket_name='your-aws-s3-bucket-name',
    ...     aws_credentials={
    ...         'aws_access_key_id': 'YOUR_ACCESS_KEY_ID',
    ...         'aws_secret_access_key': 'YOUR_SECRET_ACCESS_KEY'
    ...     }
    ... )
    >>> print(message)
    Metadata synced successfully with AWS.

    >>> # Sync metadata to Google Cloud Storage
    >>> message = sync_with_cloud(
    ...     metadata=metadata,
    ...     cloud_provider='gcp',
    ...     bucket_name='your-gcp-bucket-name',
    ...     gcp_credentials={
    ...         'service_account_json': '/path/to/your/service_account.json'
    ...     }
    ... )
    >>> print(message)
    Metadata synced successfully with GCP.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.
    audit : Generates an audit trail for metadata changes.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "Google Cloud Storage Documentation", Google Cloud.
       https://cloud.google.com/storage/docs/

    .. [3] "boto3 Documentation", AWS.
       https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

    .. [4] "Google Cloud Storage Client Library for Python", Google Cloud.
       https://googleapis.dev/python/storage/latest/index.html

    """
    result = _sync_with_cloud_inner(
        metadata=metadata,
        cloud_provider=cloud_provider,
        retries=retries,
        batch_size=batch_size,
        bucket_name=bucket_name,
        aws_credentials=aws_credentials,
        gcp_credentials=gcp_credentials
    )
    return result


def _sync_with_cloud_inner(
    metadata: Dict[str, Any],
    cloud_provider: str,
    retries: int,
    batch_size: int,
    bucket_name: str,
    aws_credentials: Optional[Dict[str, str]],
    gcp_credentials: Optional[Dict[str, str]]
) -> str:
    """
    Internal helper function to synchronize metadata with the specified cloud provider.

    Parameters
    ----------
    metadata : dict
        The metadata to synchronize.

    cloud_provider : str
        The cloud provider to sync with.

    retries : int
        Number of retry attempts.

    batch_size : int
        Number of records per batch.

    bucket_name : str
        Name of the cloud storage bucket.

    aws_credentials : dict or None
        AWS credentials for authentication.

    gcp_credentials : dict or None
        GCP credentials for authentication.

    Returns
    -------
    str
        Success message upon successful synchronization.

    Raises
    ------
    ValueError
        If required parameters are missing.

    RuntimeError
        If synchronization fails after the specified number of retries.

    """
    if cloud_provider == 'aws' and not aws_credentials:
        logger.error("AWS credentials must be provided for AWS synchronization.")
        raise ValueError("AWS credentials must be provided for AWS synchronization.")
    if cloud_provider == 'gcp' and not gcp_credentials:
        logger.error("GCP credentials must be provided for GCP synchronization.")
        raise ValueError("GCP credentials must be provided for GCP synchronization.")

    # Break metadata into batches
    metadata_items = list(metadata.items())
    total_items = len(metadata_items)
    logger.info(f"Starting synchronization of {total_items}"
                f" metadata items to {cloud_provider.upper()}.")

    for attempt in range(1, retries + 1):
        try:
            for i in range(0, total_items, batch_size):
                metadata_batch = dict(metadata_items[i:i + batch_size])
                _sync_metadata(
                    metadata_batch,
                    cloud_provider,
                    bucket_name,
                    aws_credentials,
                    gcp_credentials
                )
            logger.info(f"Metadata synced successfully with {cloud_provider.upper()}.")
            return f"Metadata synced successfully with {cloud_provider.upper()}."
        except Exception as e:
            logger.error(f"Attempt {attempt} failed with error: {e}")
            if attempt < retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Exponential backoff
            else:
                logger.error(
                    f"Failed to sync metadata with {cloud_provider.upper()}"
                    f" after {retries} attempts.")
                raise RuntimeError(
                    f"Failed to sync metadata with {cloud_provider.upper()}"
                    f" after {retries} attempts."
                ) from e

def _sync_metadata(
    metadata_batch: Dict[str, Any],
    cloud_provider: str,
    bucket_name: str,
    aws_credentials: Optional[Dict[str, str]],
    gcp_credentials: Optional[Dict[str, str]]
):
    """
    Synchronizes a batch of metadata with the specified cloud provider.

    Parameters
    ----------
    metadata_batch : dict
        A batch of metadata to synchronize.

    cloud_provider : str
        The cloud provider to sync with.

    bucket_name : str
        The name of the cloud storage bucket.

    aws_credentials : dict or None
        AWS credentials for authentication.

    gcp_credentials : dict or None
        GCP credentials for authentication.

    Raises
    ------
    ValueError
        If an unsupported cloud provider is specified.

    RuntimeError
        If synchronization with the cloud provider fails.

    """
    if cloud_provider == 'aws':
        _sync_to_aws(metadata_batch, bucket_name, aws_credentials)
    elif cloud_provider == 'gcp':
        _sync_to_gcp(metadata_batch, bucket_name, gcp_credentials)
    else:
        logger.error(f"Unsupported cloud provider: {cloud_provider}")
        raise ValueError(f"Unsupported cloud provider: {cloud_provider}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _sync_to_aws(metadata_batch: Dict[str, Any], bucket_name: str,
                 aws_credentials: Dict[str, str]):
    """
    Synchronizes a batch of metadata with AWS S3.

    Parameters
    ----------
    metadata_batch : dict
        A batch of metadata to synchronize.

    bucket_name : str
        The name of the AWS S3 bucket.

    aws_credentials : dict
        AWS credentials for authentication.

    Raises
    ------
    RuntimeError
        If synchronization with AWS S3 fails.

    """
    import boto3
    import botocore.exceptions

    try:
        s3_client = boto3.client('s3', **aws_credentials)
        # Unique file key based on timestamp
        file_key = f"metadata/metadata_sync_{int(time.time())}.json"  
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=json.dumps(metadata_batch))
        logger.info(f"Successfully synced batch to AWS S3: {file_key}")
    except botocore.exceptions.ClientError as e:
        logger.error(f"Failed to sync batch to AWS S3: {e}")
        raise RuntimeError(f"Failed to sync batch to AWS S3: {e}") from e


@ensure_pkg(
    "google",
    extra="The 'google-cloud-storage' package is required for GCP Storage operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
    dist_name="google-cloud-storage", 
    infer_dist_name=True
)
def _sync_to_gcp(metadata_batch: Dict[str, Any], bucket_name: str, 
                 gcp_credentials: Dict[str, str]):
    """
    Synchronizes a batch of metadata with Google Cloud Storage.

    Parameters
    ----------
    metadata_batch : dict
        A batch of metadata to synchronize.

    bucket_name : str
        The name of the GCP Cloud Storage bucket.

    gcp_credentials : dict
        GCP credentials for authentication.

    Raises
    ------
    RuntimeError
        If synchronization with GCP Cloud Storage fails.

    """
    from google.cloud import storage  # GCP Storage SDK
    from google.auth.exceptions import GoogleAuthError

    try:
        client = storage.Client.from_service_account_json(
            gcp_credentials['service_account_json'])
        bucket = client.bucket(bucket_name)
         # Unique blob name based on timestamp
        blob_name = f"metadata/metadata_sync_{int(time.time())}.json" 
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(metadata_batch))
        logger.info(f"Successfully synced batch to GCP Cloud Storage: {blob_name}")
    except (GoogleAuthError, Exception) as e:
        logger.error(f"Failed to sync batch to GCP Cloud Storage: {e}")
        raise RuntimeError(f"Failed to sync batch to GCP Cloud Storage: {e}") from e


@validate_params({
    'metadata': [dict],
    'schema': [dict],
    'auto_correct': [bool],
    'correction_log': [str, None],
})
def validate_schema(
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    auto_correct: bool = False,
    correction_log: Optional[str] = None
) -> bool:
    """
    Validates metadata against a predefined schema and auto-corrects common errors.

    .. math::
        \\text{Validation Result} = \\text{Validate}(\\text{Metadata}, \\text{Schema}) \\cup \\text{Auto-Correction}

    Where:
    - :math:`\\text{Metadata}` is the input metadata dictionary.
    - :math:`\\text{Schema}` is the predefined JSON schema.
    - :math:`\\text{Auto-Correction}` is the process of fixing common validation errors if enabled.

    Parameters
    ----------
    metadata : dict
        The metadata to be validated.

    schema : dict
        The JSON schema to validate the metadata against.

    auto_correct : bool, default=False
        Automatically correct common errors in the metadata based on the schema.

    correction_log : str, optional
        Path to a log file where corrections will be recorded.
        If ``auto_correct`` is ``True``, corrections will be logged here.

    Returns
    -------
    bool
        ``True`` if the metadata is valid (or becomes valid after auto-correction),
        ``False`` otherwise.

    Raises
    ------
    ValueError
        If required parameters are missing or if schema is invalid.

    RuntimeError
        If validation fails and auto-correction is not enabled,
        or if auto-correction fails to rectify the metadata.

    Notes
    -----
    - The function leverages the ``jsonschema`` library for validation.
    - Auto-correction attempts to fix common issues such as missing required fields
      by assigning default values defined in the schema.
    - Only shallow corrections are performed; nested structures require additional logic.
    - Ensure that external dependencies like ``jsonschema`` are installed.

    .. math::
        \\text{Corrected Metadata} = \\text{AutoCorrect}(\\text{Metadata}, \\text{Schema})

    Examples
    --------
    >>> from gofast.mlops.metadata import validate_schema
    >>> metadata = {
    ...     "accuracy": 0.95,
    ...     "loss": 0.05,
    ...     "epoch": 10
    ... }
    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "accuracy": {"type": "number"},
    ...         "loss": {"type": "number"},
    ...         "epoch": {"type": "integer"},
    ...         "optimizer": {"type": "string", "default": "adam"}
    ...     },
    ...     "required": ["accuracy", "loss", "epoch", "optimizer"]
    ... }
    >>> is_valid = validate_schema(
    ...     metadata=metadata,
    ...     schema=schema,
    ...     auto_correct=True,
    ...     correction_log='corrections.log'
    ... )
    >>> print(is_valid)
    True
    >>> # The 'optimizer' field was missing and auto-corrected with the default value.

    See Also
    --------
    jsonschema.validate : Validates an instance against a schema.
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.

    References
    ----------
    .. [1] "JSON Schema Validation," json-schema.org.
       https://json-schema.org/understanding-json-schema/reference/validation.html

    .. [2] "jsonschema Documentation," Python jsonschema.
       https://python-jsonschema.readthedocs.io/en/stable/

    """
    is_valid = _validate_schema_inner(
        metadata=metadata,
        schema=schema,
        auto_correct=auto_correct,
        correction_log=correction_log
    )
    return is_valid


def _validate_schema_inner(
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    auto_correct: bool,
    correction_log: Optional[str]
) -> bool:
    """
    Validates and optionally auto-corrects metadata based on the provided schema.

    Parameters
    ----------
    metadata : dict
        The metadata to be validated.

    schema : dict
        The JSON schema to validate against.

    auto_correct : bool
        Flag to enable auto-correction of metadata.

    correction_log : str or None
        Path to the correction log file.

    Returns
    -------
    bool
        Validation status after processing.

    """
    try:
        _validate_with_jsonschema(metadata, schema)
        logger.info("Metadata is valid according to the schema.")
        return True

    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")

        if auto_correct:
            logger.info("Attempting to auto-correct the metadata.")
            corrected_metadata = _auto_correct_metadata(metadata, schema)

            # Validate again after correction
            try:
                _validate_with_jsonschema(corrected_metadata, schema)
                logger.info("Metadata is valid after auto-correction.")
                if correction_log:
                    _log_corrections(corrected_metadata, correction_log)
                return True
            except Exception as e2:
                logger.error(f"Auto-correction failed: {e2}")
                return False

        return False


@ensure_pkg(
    "jsonschema",
    extra="The 'jsonschema' package is required for metadata validation.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _validate_with_jsonschema(metadata: Dict[str, Any], schema: Dict[str, Any]):
    """
    Validates metadata using the jsonschema library.

    Parameters
    ----------
    metadata : dict
        The metadata to validate.

    schema : dict
        The JSON schema to validate against.

    Raises
    ------
    jsonschema.exceptions.ValidationError
        If the metadata does not conform to the schema.

    """
    import jsonschema
    jsonschema.validate(instance=metadata, schema=schema)


def _auto_correct_metadata(metadata: Dict[str, Any], schema: Dict[str, Any]
                           ) -> Dict[str, Any]:
    """
    Automatically corrects common errors in the metadata based on the schema.

    Parameters
    ----------
    metadata : dict
        The metadata to be corrected.

    schema : dict
        The JSON schema defining the correct structure.

    Returns
    -------
    dict
        The corrected metadata.

    """
    corrected_metadata = metadata.copy()

    # Add missing required fields with default values from the schema
    for key, value in schema.get("properties", {}).items():
        if key not in corrected_metadata and "default" in value:
            corrected_metadata[key] = value["default"]
            logger.info(f"Added missing key '{key}' with default value '{value['default']}'.")

    # Remove unexpected fields that are not defined in the schema
    for key in list(corrected_metadata.keys()):
        if key not in schema.get("properties", {}):
            logger.info(f"Removed unexpected key '{key}' from metadata.")
            del corrected_metadata[key]

    return corrected_metadata


def _log_corrections(metadata: Dict[str, Any], correction_log: str):
    """
    Logs the corrections made to the metadata.

    Parameters
    ----------
    metadata : dict
        The corrected metadata.

    correction_log : str
        The path to the log file for storing corrections.

    Raises
    ------
    RuntimeError
        If writing to the correction log fails.

    """
    try:
        with open(correction_log, 'a') as f:
            f.write(json.dumps(metadata, indent=4) + '\n')
        logger.info(f"Corrections logged to {correction_log}.")
    except Exception as e:
        logger.error(f"Failed to log corrections: {e}")
        raise RuntimeError(f"Failed to log corrections: {e}")


@validate_params({
    'experiment_id': [str],
    'configuration': [dict],
    'hyperparameters': [dict],
    'performance_metrics': [dict],
    'training_logs': [str, None],
    'versioning_enabled': [bool],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'encryption_key': [str, None],
    'compression_enabled': [bool],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def track_experiment(
    experiment_id: str,
    configuration: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    training_logs: Optional[str] = None,
    versioning_enabled: bool = True,
    storage_backend: str = 'local',
    encryption_key: Optional[str] = None,
    compression_enabled: bool = False,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Tracks experiment-specific metadata including configurations, hyperparameters,
    performance metrics, and training logs, with versioning and automated storage.

    .. math::
        \\text{Experiment Metadata} = f(\\text{Configuration}, \\text{Hyperparameters},
        \\text{Performance Metrics}, \\text{Training Logs}, \\text{Version})

    Where:
    - :math:`\\text{Configuration}` includes settings and parameters for the experiment.
    - :math:`\\text{Hyperparameters}` are the tunable parameters used during training.
    - :math:`\\text{Performance Metrics}` are the results obtained from the experiment.
    - :math:`\\text{Training Logs}` provide detailed logs of the training process.
    - :math:`\\text{Version}` tracks the iteration of the experiment metadata.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment.

    configuration : dict
        Configuration settings for the experiment.

    hyperparameters : dict
        Hyperparameters used in the experiment.

    performance_metrics : dict
        Performance metrics obtained from the experiment.

    training_logs : str, optional
        Path to training logs. If provided, the contents will be included in the metadata.

    versioning_enabled : bool, default=True
        Enable versioning for experiment metadata. If ``True``, a version number is added
        to the metadata.

    storage_backend : {'local', 's3', 'database'}, default='local'
        Backend for storing experiment metadata.

        - ``'local'``: Stores metadata locally as a file.
        - ``'s3'``: Stores metadata in an AWS S3 bucket.
        - ``'database'``: Stores metadata in a MongoDB database.

    encryption_key : str, optional
        Key for encrypting metadata. Should be a valid Fernet key. If provided,
        metadata will be encrypted before storage.

    compression_enabled : bool, default=False
        Enable compression for experiment metadata storage using zlib. If ``True``,
        metadata will be compressed before storage.

    bucket_name : str, optional
        Name of the S3 bucket where metadata will be stored. Required if
        ``storage_backend`` is ``'s3'``.

    mongo_db_uri : str, optional
        MongoDB URI for database connection. Required if ``storage_backend`` is
        ``'database'``.

    Returns
    -------
    str
        A success message indicating that the experiment metadata was tracked.

    Raises
    ------
    ValueError
        If an unsupported ``storage_backend`` is provided or required parameters
        are missing.

    RuntimeError
        If storing metadata fails in the specified backend.

    Notes
    -----
    - The function supports encryption using the ``cryptography`` package and
      compression using the ``zlib`` module.
    - Ensure that external dependencies like ``boto3`` and ``pymongo`` are installed
      if using cloud or database storage backends.
    - Authentication credentials and sensitive configurations should be handled securely.

    .. math::
        \\text{Stored Metadata} = \\text{Encrypt}(\\text{Compress}(\\text{Metadata}))

    Examples
    --------
    >>> from gofast.mlops.metadata import track_experiment
    >>> from cryptography.fernet import Fernet
    >>>
    >>> # Experiment configuration, hyperparameters, and performance metrics
    >>> configuration = {"optimizer": "Adam", "learning_rate": 0.001}
    >>> hyperparameters = {"batch_size": 32, "epochs": 10}
    >>> performance_metrics = {"accuracy": 0.93, "loss": 0.07}
    >>>
    >>> # Track experiment metadata in local storage
    >>> message = track_experiment(
    ...     experiment_id="exp_001",
    ...     configuration=configuration,
    ...     hyperparameters=hyperparameters,
    ...     performance_metrics=performance_metrics,
    ...     training_logs="training_log.txt",
    ...     storage_backend='local',
    ...     compression_enabled=False,
    ...     versioning_enabled=True
    ... )
    >>> print(message)
    Experiment metadata tracked successfully for exp_001.
    >>>
    >>> # Track experiment metadata in AWS S3 with encryption
    >>> encryption_key = Fernet.generate_key()
    >>> message = track_experiment(
    ...     experiment_id="exp_002",
    ...     configuration=configuration,
    ...     hyperparameters=hyperparameters,
    ...     performance_metrics=performance_metrics,
    ...     training_logs="training_log.txt",
    ...     storage_backend='s3',
    ...     encryption_key=encryption_key.decode(),
    ...     compression_enabled=True,
    ...     versioning_enabled=True,
    ...     bucket_name='my-experiment-bucket'
    ... )
    >>> print(message)
    Experiment metadata tracked successfully for exp_002.
    >>>
    >>> # Track experiment metadata in MongoDB
    >>> message = track_experiment(
    ...     experiment_id="exp_003",
    ...     configuration=configuration,
    ...     hyperparameters=hyperparameters,
    ...     performance_metrics=performance_metrics,
    ...     training_logs=None,
    ...     storage_backend='database',
    ...     versioning_enabled=True,
    ...     mongo_db_uri='mongodb://localhost:27017/'
    ... )
    >>> print(message)
    Experiment metadata tracked successfully for exp_003.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "PyMongo Documentation", MongoDB.
       https://pymongo.readthedocs.io/en/stable/

    .. [3] "Cryptography Documentation", Cryptography.io.
       https://cryptography.io/en/latest/

    .. [4] "zlib Compression Library", Python Documentation.
       https://docs.python.org/3/library/zlib.html

    """
    # Create experiment metadata dictionary
    experiment_metadata = {
        'experiment_id': experiment_id,
        'configuration': configuration,
        'hyperparameters': hyperparameters,
        'performance_metrics': performance_metrics,
        'version': 1 if versioning_enabled else None,
        'timestamp': datetime.utcnow().isoformat()
    }

    # Include training logs if provided
    if training_logs:
        try:
            with open(training_logs, 'r') as f:
                experiment_metadata['training_logs'] = f.read()
            logger.info("Training logs included in the experiment metadata.")
        except Exception as e:
            logger.error(f"Failed to read training logs: {e}")
            raise RuntimeError(f"Failed to read training logs: {e}")

    # Convert metadata to JSON string
    metadata_json = json.dumps(experiment_metadata)

    # Optional compression of the metadata
    if compression_enabled:
        try:
            metadata_bytes = zlib.compress(metadata_json.encode('utf-8'))
            logger.info("Experiment metadata compressed.")
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Compression failed: {e}")
    else:
        metadata_bytes = metadata_json.encode('utf-8')

    # Optional encryption of the metadata
    if encryption_key:
        try:
            from cryptography.fernet import Fernet
            cipher = Fernet(encryption_key)
            metadata_bytes = cipher.encrypt(metadata_bytes)
            logger.info("Experiment metadata encrypted.")
        except ImportError:
            raise ImportError(
                "The 'cryptography' package is required for encryption. "
                "Please install it using 'pip install cryptography'."
            )
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"Encryption failed: {e}")

    # Store metadata in the specified backend
    try:
        if storage_backend == 'local':
            return _store_locally_track_e(metadata_bytes, experiment_id,
                                  versioning_enabled)
        elif storage_backend == 's3':
            if not bucket_name:
                raise ValueError(
                    "bucket_name must be provided when using 's3'"
                    " as storage_backend.")
            return _store_in_aws_s3(metadata_bytes, experiment_id, 
                                    versioning_enabled, bucket_name)
        elif storage_backend == 'database':
            if not mongo_db_uri:
                raise ValueError(
                    "mongo_db_uri must be provided when"
                    " using 'database' as storage_backend.")
            return _store_in_mongodb(
                experiment_metadata, experiment_id, versioning_enabled, mongo_db_uri)
        else:
            logger.error(f"Unsupported storage backend: {storage_backend}")
            raise ValueError(f"Unsupported storage backend: {storage_backend}")
    except Exception as e:
        logger.error(f"Failed to store experiment metadata: {e}")
        raise RuntimeError(f"Failed to store experiment metadata: {e}") from e


def _store_locally_track_e(
    metadata_bytes: bytes,
    experiment_id: str,
    versioning_enabled: bool
) -> str:
    """
    Stores experiment metadata locally as a JSON file.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store, possibly compressed and/or encrypted.

    experiment_id : str
        Unique identifier for the experiment.

    versioning_enabled : bool
        Indicates whether versioning is enabled.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    RuntimeError
        If writing to the local file system fails.

    """
    file_name = f"experiment_{experiment_id}"
    if versioning_enabled:
        file_name += "_v1"
    file_name += ".json"

    try:
        with open(file_name, 'wb') as f:
            f.write(metadata_bytes)
        logger.info(f"Experiment metadata stored locally in {file_name}.")
        return f"Experiment metadata tracked successfully for {experiment_id}."
    except Exception as e:
        logger.error(f"Failed to store experiment metadata locally: {e}")
        raise RuntimeError(f"Failed to store experiment metadata locally: {e}")


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_aws_s3(
    metadata_bytes: bytes,
    experiment_id: str,
    versioning_enabled: bool,
    bucket_name: str
) -> str:
    """
    Stores experiment metadata in AWS S3.

    Parameters
    ----------
    metadata_bytes : bytes
        The metadata to store, possibly compressed and/or encrypted.

    experiment_id : str
        Unique identifier for the experiment.

    versioning_enabled : bool
        Indicates whether versioning is enabled.

    bucket_name : str
        Name of the S3 bucket where metadata will be stored.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    RuntimeError
        If writing to S3 fails.

    Notes
    -----
    Uses the ``boto3`` library to interact with AWS S3.

    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        key = f"experiment_metadata/experiment_{experiment_id}"
        if versioning_enabled:
            key += "_v1"
        key += ".json"
        s3_client.put_object(Bucket=bucket_name, Key=key, Body=metadata_bytes)
        logger.info(f"Experiment metadata stored in AWS S3 under key: {key}")
        return f"Experiment metadata tracked successfully for {experiment_id}."
    except Exception as e:
        logger.error(f"Failed to store experiment metadata in AWS S3: {e}")
        raise RuntimeError(f"Failed to store experiment metadata in AWS S3: {e}")


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _store_in_mongodb(
    experiment_metadata: Dict[str, Any],
    experiment_id: str,
    versioning_enabled: bool,
    mongo_db_uri: str
) -> str:
    """
    Stores experiment metadata in MongoDB.

    Parameters
    ----------
    experiment_metadata : dict
        The metadata to store.

    experiment_id : str
        Unique identifier for the experiment.

    versioning_enabled : bool
        Indicates whether versioning is enabled.

    mongo_db_uri : str
        MongoDB URI for database connection.

    Returns
    -------
    str
        Success message.

    Raises
    ------
    RuntimeError
        If writing to the database fails.

    Notes
    -----
    Uses the ``pymongo`` library to interact with MongoDB.

    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['experiment_db']
        collection = db['experiment_metadata']
        document = experiment_metadata.copy()
        if versioning_enabled:
            document['version'] = 1
        collection.insert_one(document)
        logger.info(f"Experiment metadata stored in MongoDB for experiment {experiment_id}.")
        return f"Experiment metadata tracked successfully for {experiment_id}."
    except Exception as e:
        logger.error(f"Failed to store experiment metadata in MongoDB: {e}")
        raise RuntimeError(f"Failed to store experiment metadata in MongoDB: {e}")


@validate_params({
    'metadata_type': [str],
    'retention_days': [Interval(Integral, 1, None, closed="left",  inclusive=True)],
    'storage_backend': [StrOptions({'local', 's3', 'database'})],
    'preserve_versions': [list, None],
    'bucket_name': [str, None],
    'mongo_db_uri': [str, None],
})
def prune_old(
    metadata_type: str,
    retention_days: int,
    storage_backend: str = 'local',
    preserve_versions: Optional[List[int]] = None,
    bucket_name: Optional[str] = None,
    mongo_db_uri: Optional[str] = None
) -> str:
    """
    Prunes old or redundant metadata based on a retention policy, while keeping key
    versions for reproducibility.

    .. math::
        \\text{Pruned Metadata} = \\{
            m \\in \\text{Metadata} \\mid m.\\text{last\\_modified} < \\text{cutoff\\_date}
            \\land m.\\text{version} \\notin \\text{Preserve Versions}
        \\}

    Where:
    - :math:`\\text{Metadata}` is the collection of all metadata records.
    - :math:`\\text{cutoff\\_date}` is calculated as the current date minus the retention period.
    - :math:`\\text{Preserve Versions}` is a set of version numbers that should not be pruned.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to prune (e.g., ``'model'``, ``'dataset'``, ``'experiment'``).

    retention_days : int
        Number of days to retain metadata before pruning. Metadata older than this
        period will be considered for pruning.

    storage_backend : {'local', 's3', 'database'}, default='local'
        Backend for storing metadata.

        - ``'local'``: Prunes metadata stored locally as files.
        - ``'s3'``: Prunes metadata stored in an AWS S3 bucket.
        - ``'database'``: Prunes metadata stored in a MongoDB database.

    preserve_versions : list of int, optional
        List of version numbers to preserve for reproducibility. Metadata entries
        with these versions will not be pruned regardless of their age.

    bucket_name : str, optional
        Name of the S3 bucket where metadata is stored. Required if
        ``storage_backend`` is ``'s3'``.

    mongo_db_uri : str, optional
        MongoDB URI for database connection. Required if ``storage_backend`` is
        ``'database'``.

    Returns
    -------
    str
        A success message indicating the pruning process completion.

    Raises
    ------
    ValueError
        If an unsupported ``storage_backend`` is provided or required parameters
        are missing.

    RuntimeError
        If pruning operations fail due to backend-specific issues.

    Notes
    -----
    - The function supports pruning metadata from local storage, AWS S3, and MongoDB.
    - Ensure that external dependencies like ``boto3`` and ``pymongo`` are installed if
      using cloud or database storage backends.
    - Authentication credentials and sensitive configurations should be handled securely.

    .. math::
        \\text{Cutoff Date} = \\text{Current Date} - \\text{Retention Days}

    Examples
    --------
    >>> from gofast.mlops.metadata import prune_old

    # Prune local metadata older than 30 days, preserving versions 1 and 2
    >>> message = prune_old(
    ...     metadata_type='model',
    ...     retention_days=30,
    ...     storage_backend='local',
    ...     preserve_versions=[1, 2]
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 30 days.

    # Prune AWS S3 metadata older than 60 days without preserving any versions
    >>> message = prune_old(
    ...     metadata_type='dataset',
    ...     retention_days=60,
    ...     storage_backend='s3',
    ...     bucket_name='my-metadata-bucket'
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 60 days.

    # Prune MongoDB metadata older than 90 days, preserving version 3
    >>> message = prune_old(
    ...     metadata_type='experiment',
    ...     retention_days=90,
    ...     storage_backend='database',
    ...     preserve_versions=[3],
    ...     mongo_db_uri='mongodb://localhost:27017/'
    ... )
    >>> print(message)
    Old metadata pruned successfully based on a retention policy of 90 days.

    See Also
    --------
    log_metadata : Logs and stores metadata with various options.
    retrieve : Retrieves stored metadata based on specified criteria.
    audit : Generates an audit trail for metadata changes.

    References
    ----------
    .. [1] "AWS S3 Documentation", Amazon Web Services.
       https://aws.amazon.com/s3/

    .. [2] "PyMongo Documentation", MongoDB.
       https://pymongo.readthedocs.io/en/stable/

    .. [3] "Python os Module", Python Documentation.
       https://docs.python.org/3/library/os.html

    """
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    preserve_versions = preserve_versions or []

    if storage_backend == 'local':
        result = _prune_local(metadata_type, cutoff_date, preserve_versions,
                              retention_days)
    elif storage_backend == 's3':
        if not bucket_name:
            raise ValueError(
                "bucket_name must be provided when using 's3' as storage_backend.")
        result = _prune_s3(metadata_type, cutoff_date, preserve_versions,
                           retention_days, bucket_name)
    elif storage_backend == 'database':
        if not mongo_db_uri:
            raise ValueError("mongo_db_uri must be provided when using"
                             " 'database' as storage_backend.")
        result = _prune_database(metadata_type, cutoff_date, preserve_versions,
                                 retention_days, mongo_db_uri)
    else:
        logger.error(f"Unsupported storage backend: {storage_backend}")
        raise ValueError(f"Unsupported storage backend: {storage_backend}")

    return result


def _prune_local(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int
) -> str:
    """
    Prunes local metadata files based on the retention policy.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to prune.

    cutoff_date : datetime
        The cutoff date; metadata older than this will be pruned.

    preserve_versions : list of int
        Versions to preserve during pruning.

    retention_days : int
        Number of days to retain metadata before pruning.

    Returns
    -------
    str
        Success message upon successful pruning.

    Raises
    ------
    RuntimeError
        If pruning operations fail.

    """
    try:
        metadata_files = [f for f in os.listdir() if f.startswith(metadata_type)]
        for file in metadata_files:
            file_path = os.path.join(os.getcwd(), file)
            file_modified_timestamp = os.path.getmtime(file_path)
            file_modified_date = datetime.fromtimestamp(file_modified_timestamp)

            # Extract version number from filename (e.g., 'model_v1.json')
            version = None
            if '_v' in file:
                try:
                    version_str = file.split('_v')[-1].split('.')[0]
                    version = int(version_str)
                except (IndexError, ValueError):
                    logger.warning(
                        f"Unable to parse version from filename: {file}."
                        " Skipping version check.")

            # Prune old files except for preserved versions
            if file_modified_date < cutoff_date and (version not in preserve_versions):
                os.remove(file_path)
                logger.info(f"Pruned metadata file: {file_path}")

        return( 
            f"Old metadata pruned successfully based on a"
            f" retention policy of {retention_days} days."
            )
    except Exception as e:
        logger.error(f"Failed to prune local metadata: {e}")
        raise RuntimeError(f"Failed to prune local metadata: {e}") from e


@ensure_pkg(
    "boto3",
    extra="The 'boto3' package is required for AWS S3 operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _prune_s3(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int,
    bucket_name: str
) -> str:
    """
    Prunes metadata stored in AWS S3 based on the retention policy.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to prune.

    cutoff_date : datetime
        The cutoff date; metadata older than this will be pruned.

    preserve_versions : list of int
        Versions to preserve during pruning.

    retention_days : int
        Number of days to retain metadata before pruning.

    bucket_name : str
        Name of the S3 bucket where metadata is stored.

    Returns
    -------
    str
        Success message upon successful pruning.

    Raises
    ------
    RuntimeError
        If pruning operations fail.

    Notes
    -----
    Uses the ``boto3`` library to interact with AWS S3.

    """
    import boto3
    try:
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{metadata_type}/")

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    file_last_modified = obj['LastModified']

                    # Extract version number from file_key (e.g., 'model/metadata_v1.json')
                    version = None
                    if '_v' in file_key:
                        try:
                            version_str = file_key.split('_v')[-1].split('.')[0]
                            version = int(version_str)
                        except (IndexError, ValueError):
                            logger.warning(
                                f"Unable to parse version from file key: {file_key}."
                                " Skipping version check.")

                    # Prune old S3 objects except for preserved versions
                    if file_last_modified.replace(tzinfo=None) < cutoff_date and (
                            version not in preserve_versions):
                        s3_client.delete_object(Bucket=bucket_name, Key=file_key)
                        logger.info(
                            f"Pruned S3 metadata file: s3://{bucket_name}/{file_key}")

        return( 
            f"Old metadata pruned successfully based on a retention policy"
            f" of {retention_days} days."
            )
    
    except Exception as e:
        logger.error(f"Failed to prune AWS S3 metadata: {e}")
        raise RuntimeError(f"Failed to prune AWS S3 metadata: {e}") from e


@ensure_pkg(
    "pymongo",
    extra="The 'pymongo' package is required for MongoDB operations.",
    auto_install=INSTALL_DEPENDENCIES,
    use_conda=USE_CONDA,
)
def _prune_database(
    metadata_type: str,
    cutoff_date: datetime,
    preserve_versions: List[int],
    retention_days: int,
    mongo_db_uri: str
) -> str:
    """
    Prunes metadata stored in MongoDB based on the retention policy.

    Parameters
    ----------
    metadata_type : str
        The type of metadata to prune.

    cutoff_date : datetime
        The cutoff date; metadata older than this will be pruned.

    preserve_versions : list of int
        Versions to preserve during pruning.

    retention_days : int
        Number of days to retain metadata before pruning.

    mongo_db_uri : str
        MongoDB URI for database connection.

    Returns
    -------
    str
        Success message upon successful pruning.

    Raises
    ------
    RuntimeError
        If pruning operations fail.

    Notes
    -----
    Uses the ``pymongo`` library to interact with MongoDB.

    """
    from pymongo import MongoClient
    try:
        client = MongoClient(mongo_db_uri)
        db = client['metadata_db']
        collection = db[metadata_type]

        # Create query to find metadata older than cutoff_date
        query = {
            "last_modified": {"$lt": cutoff_date}
        }
        if preserve_versions:
            query["version"] = {"$nin": preserve_versions}

        # Delete the matching records
        result = collection.delete_many(query)
        logger.info(f"Pruned {result.deleted_count} metadata records"
                    f" from MongoDB for type '{metadata_type}'.")

        return ( 
            f"Old metadata pruned successfully based on a retention"
            f" policy of {retention_days} days."
            )
    except Exception as e:
        logger.error(f"Failed to prune MongoDB metadata: {e}")
        raise RuntimeError(f"Failed to prune MongoDB metadata: {e}") from e
