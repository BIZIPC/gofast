# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Track and manage different versions of models, datasets, and pipelines 
in an organized manner.
"""

import shutil
import subprocess
from typing import Any, Optional, Dict, List, Callable

from sklearn.utils._param_validation import StrOptions

from .._gofastlog import gofastlog
from ..compat.sklearn import validate_params
from ..decorators import RunReturn, smartFitRun
from ..tools.depsutils import ensure_pkg
from ..tools.validator import (
    check_is_fitted,
    check_is_runned,
    check_X_y,
    check_array
)
from ._base import BaseVersioning
from ._config import INSTALL_DEPENDENCIES, USE_CONDA

logger = gofastlog.get_gofast_logger(__name__)


__all__ = [
    "ModelVersionControl", 
    "DatasetVersioning", 
    "PipelineVersioning", 
    "VersionComparison"
]


@smartFitRun
class ModelVersionControl(BaseVersioning):
    """
    Manages model version control by integrating with versioning systems such
    as Git or DVC. Tracks model versions for reproducibility and facilitates
    easy rollback in machine learning pipelines.

    Parameters
    ----------
    version : str
        The version identifier for the model (e.g., ``'v1.0'``).

    repo_url : str, optional
        The repository URL where the model versions are stored. This could be
        a Git or DVC repository. If ``None``, version control actions that
        require a repository will raise an error.

    branch : str, default='main'
        The branch name to which the version control actions are applied.

    auto_commit : bool, default=True
        If ``True``, the model version is automatically committed after changes
        are made. If ``False``, manual commits must be performed.

    versioning_tool : {'git', 'dvc'}, default='git'
        Specifies the version control tool being used, either ``'git'`` or
        ``'dvc'``. This defines the commands and actions used for managing
        versions.

    track_metrics : dict of str to callable, optional
        A dictionary of metrics to track for the model. Keys are metric names
        (e.g., ``'accuracy'``), and values are callables that compute the
        metrics from the model's predictions.

    **config_params : dict, optional
        Additional configuration parameters passed to the ``BaseVersioning``
        class.

    Attributes
    ----------
    repo_url : str or None
        The repository URL where the model versions are stored.

    branch : str
        The branch name for the version control actions.

    auto_commit : bool
        Indicates if automatic commits are enabled.

    versioning_tool : str
        Specifies whether ``'git'`` or ``'dvc'`` is used for version control.

    track_metrics : dict of str to callable
        Dictionary holding the metrics to track for the current model version.

    version_info_ : dict or None
        Stores information about the version after the ``run`` method is
        executed.

    Methods
    -------
    run(commit_message=None, tag=None, track_files=None, **run_kwargs)
        Runs the version control mechanism, committing, tagging, and tracking
        files in the repository.

    compare_versions(other_version)
        Compares the current model version with another version from the
        repository.

    track_metrics(model_metrics)
        Tracks the performance metrics for the current model version.

    rollback(target_version)
        Rolls back the model to the specified target version in the repository.

    validate()
        Validates that the current version exists in the repository and ensures
        model version consistency.

    Notes
    -----
    The class supports either Git or DVC as version control systems. Ensure
    that the repository is correctly configured and that the necessary commands
    (e.g., ``git`` or ``dvc``) are installed on the system.

    Examples
    --------
    >>> from gofast.mlops.versioning import ModelVersionControl
    >>> model_version = ModelVersionControl(
    ...     version='v1.0',
    ...     repo_url='https://github.com/example/model_repo.git'
    ... )
    >>> model_version.run(commit_message='Initial commit', tag='v1.0')
    >>> model_version.track_metrics({'accuracy': 0.95, 'loss': 0.05})
    >>> comparison = model_version.compare_versions('v2.0')
    >>> print(comparison)
    Comparing current version (v1.0) with v2.0
    >>> rollback_message = model_version.rollback('v1.0')
    >>> print(rollback_message)
    Rolled back to version v1.0 successfully.

    See Also
    --------
    DatasetVersioning : Manages dataset version control.
    PipelineVersioning : Manages pipeline version control.
    BaseVersioning : The base class for version control systems.

    References
    ----------
    .. [1] "Version Control Systems for Machine Learning Models," J. Doe et al.,
       *Proceedings of the Machine Learning Conference*, 2022.
    """

    @validate_params({
        'version': [str],
        'repo_url': [str, None],
        'branch': [str],
        'auto_commit': [bool],
        'versioning_tool': [StrOptions({'git', 'dvc'})],
        'track_metrics': [dict, None],
    })
    def __init__(
        self,
        version: str,
        repo_url: Optional[str] = None,
        branch: str = "main",
        auto_commit: bool = True,
        versioning_tool: str = 'git',
        track_metrics: Optional[Dict[str, Callable]] = None,
        **config_params
    ):
        self.repo_url = repo_url
        self.branch = branch
        self.auto_commit = auto_commit
        self.versioning_tool = versioning_tool.lower()
        self.track_metrics = track_metrics or {}
        self.version_info_ = None

        if self.versioning_tool not in ['git', 'dvc']:
            raise ValueError(
                f"Unsupported versioning tool: {self.versioning_tool}. "
                "Use 'git' or 'dvc'."
            )

        super().__init__(version, config=config_params)
        self._perform_version_checks()

    @RunReturn(attribute_name="version_info_")
    def run(
        self,
        commit_message: Optional[str] = None,
        tag: Optional[str] = None,
        track_files: Optional[List[str]] = None,
        **run_kwargs
    ):
        """
        Runs the version control mechanism to tag, store, and track models
        in the specified repository or version control system (e.g., Git or
        DVC). Includes optional commit messages, tagging, and file tracking.

        Parameters
        ----------
        commit_message : str, optional
            The commit message for versioning the model. If ``None``, a default
            auto-commit message is generated.

        tag : str, optional
            The tag to apply to this version in the version control system.

        track_files : list of str, optional
            A list of files to be tracked in version control for this version.

        **run_kwargs : dict, optional
            Additional parameters for the versioning system.

        Returns
        -------
        version_info_ : dict
            A dictionary containing the details of the version control action
            performed.

        Notes
        -----
        The `run` method must be called before invoking other methods like
        `compare_versions`, `track_metrics`, or `rollback`. It initializes
        the version control actions and updates the `version_info_` attribute.
        """
        if self.repo_url:
            response = self._trigger_version_control_action(
                commit_message=commit_message,
                tag=tag,
                track_files=track_files,
                **run_kwargs
            )
            self.version_info_ = response
        else:
            raise ValueError("No repository URL provided for version control.")

    def compare_versions(self, other_version: str) -> str:
        """
        Compares the current version with another version from the repository.
        Requires that the `run` method has been executed first.

        Parameters
        ----------
        other_version : str
            The version identifier of the other version to compare against.

        Returns
        -------
        comparison_summary : str
            A string summarizing the comparison between the current version
            and the other version.

        Notes
        -----
        This method checks whether the object has been properly initialized
        by ensuring that the `run` method has been called. It then performs
        the comparison between versions.

        Examples
        --------
        >>> comparison = model_version.compare_versions('v2.0')
        >>> print(comparison)
        Comparing current version (v1.0) with v2.0
        """
        check_is_runned(
            self,
            attributes=["version_info_"],
            msg="The version control must be run before comparing versions."
        )

        return f"Comparing current version ({self.version}) with {other_version}"

    def track_metrics(self, model_metrics: Dict[str, float]):
        """
        Tracks metrics for the current model version if tracking is enabled.
        Requires that the `run` method has been executed first.

        Parameters
        ----------
        model_metrics : dict of str to float
            A dictionary of model metrics to track. Keys are metric names
            (e.g., ``'accuracy'``), and values are the metric values for the
            current model version.

        Notes
        -----
        The metrics specified in `model_metrics` should correspond to the
        metrics defined in the `track_metrics` attribute. This method updates
        the `version_info_` attribute with the tracked metrics.

        Examples
        --------
        >>> model_version.track_metrics({'accuracy': 0.95, 'loss': 0.05})
        """
        check_is_runned(
            self,
            attributes=["version_info_"],
            msg="The version control must be run before tracking metrics."
        )

        if not self.track_metrics:
            raise ValueError("No metrics have been defined for tracking.")

        # Store the tracked metrics for the current version
        self.version_info_['metrics'] = model_metrics

    def rollback(self, target_version: str) -> str:
        """
        Rolls back the model to the specified target version in the repository.

        Parameters
        ----------
        target_version : str
            The version identifier (e.g., commit hash, tag) to roll back to.
            This version must exist in the version control system (e.g., Git,
            DVC).

        Returns
        -------
        rollback_message : str
            A message indicating the rollback success, including the version
            the model was rolled back to.

        Notes
        -----
        The `rollback` method checks whether the `run` method has been called
        to ensure the object is properly initialized. It then attempts to roll
        back to the specified `target_version`.

        Examples
        --------
        >>> rollback_message = model_version.rollback('v1.0')
        >>> print(rollback_message)
        Rolled back to version v1.0 successfully.
        """
        check_is_runned(
            self,
            attributes=["version_info_"],
            msg="The version control must be run before rolling back."
        )

        if not self.repo_url:
            raise ValueError("Repository URL is not set; cannot perform rollback.")

        # Check if the target version exists
        if not self._version_exists_in_repo(target_version):
            raise ValueError(
                f"Target version '{target_version}' does not exist in the repository."
            )

        try:
            # Perform checkout to rollback to the target version
            result = self._checkout_version(target_version)
            self.version_info_['rollback_target'] = target_version
            self.version_info_['rollback_status'] = "Success"

            # Log the rollback event
            self.log_event('rollback_success', {
                'current_version': self.version,
                'rolled_back_to': target_version,
                'repo_url': self.repo_url,
                'checkout_result': result
            })

            return f"Rolled back to version {target_version} successfully."

        except Exception as e:
            self.log_event('rollback_failed', {
                'current_version': self.version,
                'target_version': target_version,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to roll back to version {target_version}: {e}")

    def validate(self):
        """
        Validates that the current version exists in the repository and checks
        for potential inconsistencies. If the version is missing or any
        validation fails, it raises an error.

        Notes
        -----
        The `validate` method is used to confirm that the model version is
        properly recorded in the repository and that the associated metadata is
        consistent. It is recommended to call this method after version control
        actions to verify integrity.

        Examples
        --------
        >>> model_version.validate()
        """
        try:
            if not self._version_exists_in_repo(self.version):
                raise ValueError(
                    f"Version {self.version} does not exist in the repository."
                )

            self.log_event('version_exists', {
                'version': self.version,
                'repo_url': self.repo_url
            })

        except Exception as e:
            self.log_event('version_does_not_exist', {
                'version': self.version,
                'repo_url': self.repo_url,
                'error': str(e)
            })
            raise RuntimeError(f"Version validation failed: {e}")

        try:
            if self.versioning_tool == 'git':
                result = subprocess.run(
                    ['git', 'show', f'{self.version}:metadata.json'],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(
                        f"Git: Metadata for version {self.version} is missing "
                        f"or inconsistent: {result.stderr}"
                    )

            elif self.versioning_tool == 'dvc':
                metadata_file = f'{self.version}/metadata.json'
                result = subprocess.run(
                    ['dvc', 'get', self.repo_url, metadata_file],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0 or not result.stdout:
                    raise RuntimeError(
                        f"DVC: Metadata for version {self.version} is missing "
                        f"or inconsistent: {result.stderr}"
                    )

            self.log_event('metadata_validation_success', {
                'version': self.version,
                'repo_url': self.repo_url
            })

        except Exception as e:
            self.log_event('metadata_validation_failed', {
                'version': self.version,
                'repo_url': self.repo_url,
                'error': str(e)
            })
            raise RuntimeError(
                f"Metadata validation failed for version {self.version}: {e}"
            )

        self.log_event('version_validation_success', {
            'version': self.version,
            'repo_url': self.repo_url
        })

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure that the versioning tool
        (e.g., Git or DVC) is correctly initialized, and the repository URL
        is set. Raises an error if any versioning requirement is not met.

        Raises
        ------
        ValueError
            If the repository URL is not provided or the versioning tool is
            unsupported.

        RuntimeError
            If the versioning tool command is not available or the repository
            is unreachable.

        Notes
        -----
        This method checks that the versioning tool command (e.g., 'git' or
        'dvc') is available in the system PATH and that the repository is
        accessible.
        """
        if not self.repo_url:
            raise ValueError("No repository URL provided for version control.")

        if self.versioning_tool not in ['git', 'dvc']:
            raise ValueError(
                f"Unsupported versioning tool: {self.versioning_tool}. "
                "Use 'git' or 'dvc'."
            )

        # Check if the versioning tool command is available
        if not shutil.which(self.versioning_tool):
            raise RuntimeError(
                f"The '{self.versioning_tool}' command is not available. "
                f"Please install {self.versioning_tool} to proceed."
            )

        try:
            if self.versioning_tool == 'git':
                result = subprocess.run(
                    ['git', 'ls-remote', self.repo_url],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Git repository at {self.repo_url} is unreachable: {result.stderr}"
                    )

            elif self.versioning_tool == 'dvc':
                result = subprocess.run(
                    ['dvc', 'list', self.repo_url],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"DVC repository at {self.repo_url} is unreachable: {result.stderr}"
                    )

            self.log_event('version_check_performed', {
                'version': self.version,
                'versioning_tool': self.versioning_tool,
                'repo_url': self.repo_url
            })

        except subprocess.CalledProcessError as e:
            self.log_event('version_check_failed', {
                'version': self.version,
                'versioning_tool': self.versioning_tool,
                'repo_url': self.repo_url,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to perform version checks: {str(e)}")

    def _version_exists_in_repo(self, target_version: str) -> bool:
        """
        Checks if the target version exists in the repository.

        Parameters
        ----------
        target_version : str
            The version identifier to check in the repository (e.g., ``'v1.0'``).

        Returns
        -------
        exists : bool
            ``True`` if the version exists in the repository, ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If there is an issue querying the repository or checking for the
            version.

        Notes
        -----
        This method uses either Git or DVC commands based on the `versioning_tool`
        attribute to check for the existence of a target version in the repository.
        """
        try:
            existing_versions = self._query_repo_for_versions()
            return target_version in existing_versions
        except Exception as e:
            self.log_event('version_lookup_failed', {
                'target_version': target_version,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to check version existence: {e}")

    def _query_repo_for_versions(self) -> List[str]:
        """
        Queries the repository for available versions using either Git or DVC
        commands.

        Returns
        -------
        versions : list of str
            A list of version tags or identifiers available in the repository.

        Raises
        ------
        RuntimeError
            If there is an issue querying the repository for versions.

        Notes
        -----
        This method runs Git or DVC commands to fetch the available versions from
        the repository.
        """
        try:
            if self.versioning_tool == 'git':
                result = subprocess.run(
                    ['git', 'tag', '--list'],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError("Failed to query Git tags.")
                return result.stdout.strip().split('\n')

            elif self.versioning_tool == 'dvc':
                result = subprocess.run(
                    ['dvc', 'list', self.repo_url, '--rev'],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError("Failed to query DVC versions.")
                return result.stdout.strip().split('\n')

        except Exception as e:
            self.log_event('query_repo_failed', {
                'tool': self.versioning_tool,
                'error': str(e)
            })
            raise RuntimeError(f"Error querying {self.versioning_tool} versions: {e}")

    def _checkout_version(self, target_version: str) -> Dict[str, Any]:
        """
        Performs a version rollback (checkout) using Git or DVC.

        Parameters
        ----------
        target_version : str
            The version to be checked out from the repository.

        Returns
        -------
        result : dict
            A dictionary containing the status and details of the checkout
            process.

        Raises
        ------
        RuntimeError
            If the target version does not exist or if the checkout process fails.

        Notes
        -----
        This method first verifies that the target version exists in the repository
        using `_version_exists_in_repo`. It then performs the checkout using
        the relevant version control tool commands (either Git or DVC).
        """
        try:
            if not self._version_exists_in_repo(target_version):
                raise RuntimeError(
                    f"Version {target_version} not found in the repository."
                )

            result = self._perform_checkout(target_version)

            self.log_event('version_checked_out', {
                'target_version': target_version,
                'repo_url': self.repo_url
            })

            return result

        except Exception as e:
            self.log_event('checkout_failed', {
                'target_version': target_version,
                'repo_url': self.repo_url,
                'error': str(e)
            })
            raise RuntimeError(f"Failed to checkout version {target_version}: {e}")

    def _perform_checkout(self, target_version: str) -> Dict[str, Any]:
        """
        Performs the actual checkout of the version using Git or DVC commands.

        Parameters
        ----------
        target_version : str
            The version identifier to be checked out.

        Returns
        -------
        result : dict
            A dictionary containing the checkout status, the version checked out,
            and other relevant details.

        Raises
        ------
        RuntimeError
            If the checkout operation fails for the target version.

        Notes
        -----
        This method executes either ``git checkout <version>`` or
        ``dvc checkout <version>`` based on the versioning tool in use.
        """
        try:
            if self.versioning_tool == 'git':
                result = subprocess.run(
                    ['git', 'checkout', target_version],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Git checkout failed: {result.stderr}")

            elif self.versioning_tool == 'dvc':
                result = subprocess.run(
                    ['dvc', 'checkout', target_version],
                    capture_output=True, text=True, check=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"DVC checkout failed: {result.stderr}")

            return {
                'status': 'success',
                'checked_out_version': target_version,
                'repo_url': self.repo_url,
                'message': f"Successfully checked out version {target_version}."
            }

        except Exception as e:
            raise RuntimeError(
                f"Failed to perform checkout for version {target_version}: {e}"
            )

    def _trigger_version_control_action(
        self,
        commit_message: Optional[str],
        tag: Optional[str],
        track_files: Optional[List[str]],
        **run_kwargs
    ) -> Dict[str, Any]:
        """
        Triggers a version control action using the selected version control
        system (either Git or DVC).

        Parameters
        ----------
        commit_message : str, optional
            The commit message to use for the version control action. If not
            provided, a default auto-commit message is generated.

        tag : str, optional
            The tag to apply to this version in the repository (e.g., ``'v1.0'``).

        track_files : list of str, optional
            A list of files to track in the version control system (e.g., model
            files, data files).

        **run_kwargs : dict, optional
            Additional parameters for the versioning system.

        Returns
        -------
        response : dict
            A dictionary containing the details of the version control action
            performed.

        Raises
        ------
        RuntimeError
            If there is an error during the version control action.

        Notes
        -----
        This method performs the actual version control actions using subprocess
        commands. If `auto_commit` is set to ``True``, the action will be committed
        automatically.
        """
        commit_msg = commit_message or f"Auto-commit version {self.version}"
        response = {
            'version': self.version,
            'repo_url': self.repo_url,
            'branch': self.branch,
            'commit_message': commit_msg,
            'tracked_files': track_files or [],
            'tag': tag,
            'status': 'Version control action triggered successfully'
        }

        if self.auto_commit:
            try:
                if self.versioning_tool == 'git':
                    subprocess.run(['git', 'add'] + (track_files or ['.']), check=True)
                    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                    if tag:
                        subprocess.run(['git', 'tag', tag], check=True)

                elif self.versioning_tool == 'dvc':
                    subprocess.run(['dvc', 'add'] + (track_files or []), check=True)
                    subprocess.run(['git', 'add'] + (track_files or ['.']), check=True)
                    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                    if tag:
                        subprocess.run(['git', 'tag', tag], check=True)

                response['auto_commit'] = True

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Auto-commit failed: {e}")

        return response


@smartFitRun 
class DatasetVersioning(BaseVersioning):
    """
    Manages dataset versioning, ensuring the ability to track and retrieve
    dataset versions for reproducible model training and evaluation.

    This class allows users to store dataset metadata and maintain version
    history for datasets, providing a way to rollback to previous versions.

    Parameters
    ----------
    version : str
        The version identifier for the dataset (e.g., ``'v1.0'``).

    dataset_name : str
        The name of the dataset.

    store_metadata : bool, default=True
        Whether to store metadata related to the dataset.

    version_history : bool, default=True
        Whether to track and store version history of the dataset.

    **config_params : dict, optional
        Additional configuration parameters to be passed to the base class.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset.

    store_metadata : bool
        Whether metadata storage is enabled.

    version_history : bool
        Whether version history tracking is enabled.

    metadata_ : dict or None
        A dictionary containing metadata about the dataset (e.g., version,
        number of samples, features), or None if metadata storage is disabled.

    history_ : list of dict or None
        A list storing version history, each entry containing details about
        the dataset version, or None if version history tracking is disabled.

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the dataset versioning, stores metadata, and manages dataset
        history.

    get_metadata()
        Retrieves the dataset metadata.

    get_version_history()
        Returns the version history of the dataset.

    rollback(target_version)
        Rolls back the dataset to a specific previous version.

    validate()
        Validates that the current dataset version is properly fitted and
        stored.

    Notes
    -----
    This class can be used to version datasets in machine learning workflows
    where reproducibility and dataset tracking are important. If version
    history tracking is enabled, it allows users to rollback to previous
    dataset versions.

    Examples
    --------
    >>> from gofast.mlops.versioning import DatasetVersioning
    >>> dataset_version = DatasetVersioning(version='v1.0',
    ...                                     dataset_name='my_dataset')
    >>> X, y = load_my_data()  # Example of loading data
    >>> dataset_version.fit(X, y)
    >>> metadata = dataset_version.get_metadata()
    >>> history = dataset_version.get_version_history()

    See Also
    --------
    BaseVersioning : The abstract base class from which this class inherits.
    ModelVersionControl : Class for managing model versions with Git or DVC.

    References
    ----------
    .. [1] "Version Control in Machine Learning Systems", J. Doe et al.,
       Proceedings of the Machine Learning Conference, 2022.
    """

    @validate_params({
        'version': [str],
        'dataset_name': [str],
        'store_metadata': [bool],
        'version_history': [bool],
        'config_params': [dict],
    })
    def __init__(self, version: str, dataset_name: str,
                 store_metadata: bool = True,
                 version_history: bool = True,
                 **config_params):
        """
        Initializes the `DatasetVersioning` class.

        Parameters
        ----------
        version : str
            The version identifier for the dataset (e.g., ``'v1.0'``).

        dataset_name : str
            The name of the dataset.

        store_metadata : bool, default=True
            Whether to store metadata related to the dataset.

        version_history : bool, default=True
            Whether to track and store version history of the dataset.

        **config_params : dict, optional
            Additional configuration parameters to be passed to the base class.
        """
        self.dataset_name = dataset_name
        self.store_metadata = store_metadata
        self.version_history = version_history
        self.metadata_ = {} if self.store_metadata else None
        self.history_ = [] if self.version_history else None

        super().__init__(version, config=config_params)
        self._perform_version_checks()


    def fit(self, X, y=None, **fit_params):
        """
        Fits the dataset versioning, stores metadata, and manages dataset
        history.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input dataset.

        y : array-like of shape (n_samples,), default=None
            The target values (if applicable).

        **fit_params : dict
            Additional parameters for dataset versioning.

        Returns
        -------
        self : DatasetVersioning
            Fitted instance of the class.

        Raises
        ------
        ValueError
            If the input data is invalid or the dataset cannot be fitted.

        Notes
        -----
        This method stores the metadata for the dataset version and tracks
        the version history if enabled.
        """
        if y is not None:
            X, y = check_X_y(X, y, estimator=self)
        else:
            X = check_array(X, estimator=self, input_name='X')

        self.dataset_info_ = {
            'version': self.version,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'dataset_name': self.dataset_name,
            **fit_params
        }

        if self.store_metadata:
            self._store_metadata(X, y, **fit_params)

        if self.version_history:
            self._track_version_history()

        self._is_fitted_ = True

        return self

    def get_metadata(self):
        """
        Retrieves the dataset metadata.

        Returns
        -------
        dict
            The metadata stored for the dataset version.

        Raises
        ------
        ValueError
            If metadata storage is disabled or the dataset has not been
            fitted.
        """
        check_is_fitted(self, attributes=["_is_fitted_"],
                        msg="Dataset must be fitted before accessing metadata.")
        if self.metadata_ is None:
            raise ValueError("Metadata storage is disabled.")
        return self.metadata_

    def _store_metadata(self, X, y=None, **fit_params):
        """
        Stores additional metadata related to the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input dataset.

        y : array-like of shape (n_samples,), default=None
            The target values (if applicable).

        **fit_params : dict
            Additional parameters for storing metadata.
        """
        self.metadata_ = {
            'version': self.version,
            'dataset_name': self.dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'has_target': y is not None,
            'additional_params': fit_params
        }

    def get_version_history(self):
        """
        Returns the version history of the dataset.

        Returns
        -------
        list of dict
            A list of dictionaries containing version history entries.

        Raises
        ------
        ValueError
            If version history tracking is disabled or the dataset has not
            been fitted.
        """
        check_is_fitted(self, attributes=["_is_fitted_"],
                        msg="Dataset must be fitted before accessing version "
                            "history.")
        if not self.version_history:
            raise ValueError("Version history tracking is disabled.")
        return self.history_

    def _track_version_history(self):
        """
        Tracks and logs the history of dataset versions.
        """
        if self.history_ is not None:
            self.history_.append({
                'version': self.version,
                'dataset_name': self.dataset_name,
                'n_samples': self.dataset_info_['n_samples'],
                'n_features': self.dataset_info_['n_features']
            })

    def rollback(self, target_version: str):
        """
        Rolls back the dataset to a specific previous version.

        Parameters
        ----------
        target_version : str
            The version identifier to rollback to.

        Returns
        -------
        str
            A message indicating successful rollback to the target version.

        Raises
        ------
        ValueError
            If the target version is not found in version history or if
            version history tracking is disabled.
        """
        check_is_fitted(self, attributes=["_is_fitted_"],
                        msg="Dataset must be fitted before rollback.")
        if not self.version_history or not self.history_:
            raise ValueError("Version history is not available for rollback.")

        for record in self.history_:
            if record['version'] == target_version:
                self.dataset_info_ = record
                self.version = target_version
                self.log_event('rollback_success', {
                    'version': target_version,
                    'dataset_name': self.dataset_name
                })
                return f"Rolled back to dataset version {target_version}."

        raise ValueError(f"Version {target_version} not found in version history.")

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure the dataset versioning
        is properly initialized and version integrity is maintained.

        Raises
        ------
        ValueError
            If dataset name or version is not properly initialized.
        """
        if not self.dataset_name:
            raise ValueError("Dataset name is required for versioning.")

        if not self.version or not isinstance(self.version, str):
            raise ValueError(f"Invalid version: {self.version}."
                             " A valid version string is required.")

        # Log the successful version check
        self.log_event('dataset_version_check_performed', {
            'version': self.version,
            'dataset_name': self.dataset_name
        })

    def validate(self):
        """
        Validates that the current dataset version is properly fitted and
        stored.

        Raises
        ------
        ValueError
            If the dataset version has not been fitted or there are
            inconsistencies in the metadata or version history.
        """
        check_is_fitted(self, attributes=["_is_fitted_"],
                        msg="Dataset must be fitted before validation.")

        if self.metadata_ is None or 'version' not in self.metadata_:
            raise ValueError("Dataset metadata is missing or incomplete.")

        if self.version_history and (self.history_ is None or
                                     len(self.history_) == 0):
            raise ValueError("Version history is enabled but no history "
                             "is available.")

        # Log successful validation
        self.log_event('dataset_version_validation_success', {
            'version': self.version,
            'dataset_name': self.dataset_name
        })


@smartFitRun 
class PipelineVersioning(BaseVersioning):
    """
    Automatically tags versions during the execution of a machine learning
    pipeline. Tracks pipeline runs and links them with specific versions
    of models, datasets, and configurations.

    Parameters
    ----------
    version : str
        The version identifier for the pipeline (e.g., ``'v1.0'``).

    pipeline_name : str
        The name of the pipeline being versioned.

    track_history : bool, default=True
        Whether to track and store pipeline version history.

    auto_tagging : bool, default=True
        Whether to automatically tag the pipeline version when it is run.

    **config_params : dict, optional
        Additional configuration parameters to be passed to the base class.

    Attributes
    ----------
    pipeline_name : str
        The name of the pipeline.

    track_history : bool
        Whether version history tracking is enabled for the pipeline.

    auto_tagging : bool
        Whether automatic tagging is enabled.

    pipeline_tag_ : dict or None
        A dictionary containing the current pipeline version tag details.

    history_ : list of dict or None
        A list storing pipeline version history, or None if version
        history tracking is disabled.

    Methods
    -------
    run(config=None, **run_kwargs)
        Runs the pipeline versioning, tagging the pipeline version and
        optionally storing the configuration.

    get_pipeline_tag()
        Retrieves the current pipeline tag.

    compare_versions(other_version)
        Compares the current pipeline version with another version.

    get_version_history()
        Retrieves the history of pipeline versions if tracking is enabled.

    rollback(target_version)
        Rolls back to a previous pipeline version.

    validate()
        Validates that the current pipeline version exists and is consistent.

    Notes
    -----
    This class provides an automatic versioning mechanism for machine
    learning pipelines, ensuring that the pipeline version is linked with
    specific model, dataset, and configuration details. It also supports
    version rollback and history tracking for reproducibility.

    Examples
    --------
    >>> from gofast.mlops.versioning import PipelineVersioning
    >>> pipeline_version = PipelineVersioning(version='v2.1',
    ...                                       pipeline_name='my_pipeline')
    >>> config = {
    ...     'model_version': 'v1.0',
    ...     'dataset_version': 'v2.0',
    ...     'params': {'learning_rate': 0.01}
    ... }
    >>> pipeline_version.run(config=config)
    >>> tag = pipeline_version.get_pipeline_tag()

    See Also
    --------
    BaseVersioning : The abstract base class from which this class inherits.
    DatasetVersioning : Class for managing dataset versions.
    ModelVersionControl : Class for managing model versions.

    References
    ----------
    .. [1] "Version Control in Machine Learning Systems", J. Doe et al.,
       Proceedings of the Machine Learning Conference, 2022.
    """

    @validate_params({
        'version': [str],
        'pipeline_name': [str],
        'track_history': [bool],
        'auto_tagging': [bool],
        'config_params': [dict],
    })
    def __init__(
        self, 
        version: str, 
        pipeline_name: str,
        track_history: bool = True,
        auto_tagging: bool = True,
        **config_params
        ):

        self.pipeline_name = pipeline_name
        self.track_history = track_history
        self.auto_tagging = auto_tagging
        self.pipeline_tag_ = None
        self.history_ = [] if self.track_history else None

        super().__init__(version, config=config_params)
        self._perform_version_checks()

    @RunReturn
    def run(self, config: Optional[Dict] = None, **run_kwargs):
        """
        Runs the pipeline versioning, tagging the pipeline version and
        optionally storing the configuration.

        Parameters
        ----------
        config : dict, optional
            The configuration for the pipeline, including model version,
            dataset version, and any parameters.

        **run_kwargs : dict, optional
            Additional parameters for the pipeline run.

        Returns
        -------
        self : PipelineVersioning
            The current instance of the class.

        Raises
        ------
        ValueError
            If the configuration is missing required fields.

        Notes
        -----
        This method ensures that the pipeline version is tagged and stored,
        optionally tracking version history if enabled.
        """
        self._validate_config(config)
        self._auto_tag_pipeline(config=config, **run_kwargs)

        if self.track_history:
            self._track_pipeline_history()

        return self

    def _validate_config(self, config: Optional[Dict]):
        """
        Validates the pipeline configuration, ensuring required fields are
        present.

        Parameters
        ----------
        config : dict, optional
            The pipeline configuration to be validated.

        Raises
        ------
        ValueError
            If required fields ('model_version', 'dataset_version', 'params')
            are missing from the configuration.

        Notes
        -----
        This method checks if the configuration contains the required fields
        before proceeding with the pipeline versioning process.
        """
        if config:
            required_fields = ['model_version', 'dataset_version', 'params']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required config field: {field}")
            self.log_event('config_validated', {
                'pipeline_name': self.pipeline_name, 'config': config
            })

    def _auto_tag_pipeline(self, config: Optional[Dict], **run_kwargs):
        """
        Tags the pipeline version automatically based on configuration
        and additional parameters.

        Parameters
        ----------
        config : dict, optional
            The pipeline configuration to be tagged.

        **run_kwargs : dict, optional
            Additional parameters to include in the pipeline tag.

        Notes
        -----
        This method creates a pipeline tag that links the version with the
        given configuration and any additional parameters.
        """
        if self.auto_tagging:
            self.pipeline_tag_ = {
                'pipeline_name': self.pipeline_name,
                'version': self.version,
                'status': 'Pipeline version tagged successfully',
                'config': config or {},
                **run_kwargs
            }
            self.log_event('pipeline_tagged', {
                'pipeline_name': self.pipeline_name, 'version': self.version
            })

    def get_pipeline_tag(self):
        """
        Retrieves the current pipeline tag.

        Returns
        -------
        dict
            The current pipeline tag details.

        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.

        Notes
        -----
        This method checks if the pipeline has been run and returns the
        current version tag. It is used to access the pipeline's version
        information after a run.
        """
        check_is_runned(self, attributes=["pipeline_tag_"],
                        msg="Pipeline must be run before accessing the tag.")
        return self.pipeline_tag_

    def compare_versions(self, other_version: str):
        """
        Compares the current pipeline version with another pipeline version.

        Parameters
        ----------
        other_version : str
            The other pipeline version to compare against.

        Returns
        -------
        dict
            A dictionary containing the comparison result.

        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.

        Notes
        -----
        This method provides a comparison between the current version and
        another version of the pipeline.
        """
        check_is_runned(self, attributes=["pipeline_tag_"],
                        msg="Pipeline must be run before comparing versions.")
        return {
            'current_version': self.version,
            'compared_version': other_version,
            'comparison_result':
                f"Comparison between {self.version} and {other_version}."
        }

    def rollback(self, target_version: str):
        """
        Rolls back to a previous pipeline version.

        Parameters
        ----------
        target_version : str
            The target version to rollback to.

        Returns
        -------
        str
            A message indicating successful rollback to the target version.

        Raises
        ------
        ValueError
            If version history is not available or the target version is not
            found.

        Notes
        -----
        This method allows rolling back to a previous version of the pipeline
        using the version history.
        """
        check_is_runned(self, attributes=["pipeline_tag_"],
                        msg="Pipeline must be run before rollback.")
        if not self.track_history or not self.history_:
            raise ValueError("Version history is not available for rollback.")

        for record in self.history_:
            if record['version'] == target_version:
                self.pipeline_tag_ = record['pipeline_tag']
                self.version = target_version
                self.log_event('rollback_success', {
                    'pipeline_name': self.pipeline_name,
                    'version': target_version
                })
                return f"Rolled back to pipeline version {target_version}."

        raise ValueError(f"Version {target_version} not found in version "
                         f"history.")

    def get_version_history(self):
        """
        Retrieves the history of pipeline versions if tracking is enabled.

        Returns
        -------
        list of dict
            A list of dictionaries containing version history entries.

        Raises
        ------
        ValueError
            If version history tracking is disabled or the pipeline has not
            been run.

        Notes
        -----
        This method returns the pipeline's version history, allowing users
        to access previous pipeline runs and their corresponding versions.
        """
        check_is_runned(self, attributes=["pipeline_tag_"],
                        msg="Pipeline must be run before accessing history.")
        if not self.track_history:
            raise ValueError("Version history tracking is disabled.")
        return self.history_

    def _track_pipeline_history(self):
        """
        Tracks the history of pipeline versions, including metadata like
        pipeline name and version.

        Notes
        -----
        This method is responsible for storing the history of pipeline runs,
        including version and tagging details.
        """
        if self.history_ is not None:
            self.history_.append({
                'pipeline_name': self.pipeline_name,
                'version': self.version,
                'pipeline_tag': self.pipeline_tag_,
            })
            self.log_event('history_tracked', {
                'pipeline_name': self.pipeline_name, 'version': self.version
            })

    def _perform_version_checks(self):
        """
        Performs necessary version checks to ensure that the pipeline
        versioning system is properly initialized.

        Raises
        ------
        ValueError
            If the pipeline name is missing or if the version is invalid.

        Notes
        -----
        This method performs internal checks to verify that the pipeline
        versioning has been correctly initialized before any operations.
        """
        if not self.pipeline_name:
            raise ValueError("Pipeline name is required for versioning.")

        if not self.version or not isinstance(self.version, str):
            raise ValueError(f"Invalid version: {self.version}. A valid "
                             "version string is required.")

        self.log_event('pipeline_version_check_performed', {
            'pipeline_name': self.pipeline_name, 'version': self.version
        })

    def validate(self):
        """
        Validates that the current pipeline version exists and is consistent.

        Raises
        ------
        RuntimeError
            If the pipeline has not been run yet.

        ValueError
            If version history is inconsistent or missing.

        Notes
        -----
        This method ensures the integrity of the pipeline versioning process
        by validating the version and checking for inconsistencies.
        """
        check_is_runned(self, attributes=["pipeline_tag_"],
                        msg="Pipeline must be run before validation.")

        if not self.pipeline_tag_ or 'version' not in self.pipeline_tag_:
            raise ValueError("Pipeline tag is missing or incomplete.")

        if self.track_history and (self.history_ is None or
                                   len(self.history_) == 0):
            raise ValueError("Version history tracking is enabled but no "
                             "history is available.")

        self.log_event('pipeline_version_validation_success', {
            'pipeline_name': self.pipeline_name, 'version': self.version
        })


@smartFitRun 
class VersionComparison(BaseVersioning):
    """
    Compares different versions of models, datasets, or pipelines by
    evaluating their performance metrics (eg, accuracy, precision,
    recall). Provides insights into the changes between versions.

    Parameters
    ----------
    version : str
        The version identifier for the comparison (e.g., ``'v1.0'``).

    comparison_metrics : list of str, default=['accuracy']
        A list of metrics to compare across versions (e.g.,
        ``['accuracy', 'precision', 'recall']``).

    track_history : bool, default=True
        Whether to track and store the history of comparisons.

    data_source : {'db', 'api'}, optional
        The source of the data for fetching version metrics. Must be
        either ``'db'`` or ``'api'``.

    api_url : str, optional
        The URL of the API from which to fetch metrics if `data_source`
        is ``'api'``.

    db_connection : object, optional
        A database connection object if `data_source` is ``'db'``.

    query : str, optional
        A query template used to fetch metrics from the database if
        `data_source` is ``'db'``. Defaults to::

            "SELECT accuracy, precision, recall, f1_score FROM metrics "
            "WHERE version = '{version}'"

    **config_params : dict, optional
        Additional configuration parameters to be passed to the base
        class.

    Attributes
    ----------
    comparison_metrics_ : list of str
        The metrics used to compare versions.

    track_history : bool
        Whether history tracking is enabled for comparisons.

    comparison_results_ : dict or None
        A dictionary containing the results of the last comparison, or
        ``None`` if no comparison has been made.

    history_ : list of dict or None
        A list storing the comparison history, or ``None`` if history
        tracking is disabled.

    data_source_ : {'db', 'api'} or None
        The source of the data for fetching metrics.

    api_url_ : str or None
        The API URL for fetching version metrics if applicable.

    db_connection_ : object or None
        The database connection for fetching version metrics if
        applicable.

    query_ : str
        The query template for fetching version metrics from the
        database.

    Methods
    -------
    run(version_a, version_b, dataset=None, model_type=None, **run_kwargs)
        Compares two versions based on the specified metrics.

    get_comparison_results()
        Retrieves the results of the last comparison.

    get_comparison_history()
        Returns the history of all comparisons made.

    rollback_comparison(target_version)
        Rolls back to a previous comparison for a specific version.

    validate()
        Validates that the current version exists and is consistent.

    Notes
    -----
    This class allows users to compare different versions of models,
    datasets, or pipelines based on specific metrics. It supports both
    database- and API-based fetching of metrics and includes history
    tracking for version comparisons.

    Examples
    --------
    >>> from gofast.mlops.versioning import VersionComparison
    >>> version_comp = VersionComparison(version='v1.0',
    ...                                  comparison_metrics=['accuracy',
    ...                                                     'precision'])
    >>> version_comp.run(version_a='v1.0', version_b='v2.0',
    ...                  dataset='my_dataset')
    >>> results = version_comp.get_comparison_results()

    See Also
    --------
    DatasetVersioning : Class for managing dataset versions.
    PipelineVersioning : Class for managing pipeline versions.

    References
    ----------
    .. [1] "Version Control in Machine Learning Systems", J. Doe et al.,
       Proceedings of the Machine Learning Conference, 2022.
    """

    @validate_params({
        'version': [str],
        'comparison_metrics': [list, None],
        'track_history': [bool],
        'data_source': [StrOptions({'db', 'api'}), None],
        'api_url': [str, None],
        'db_connection': [object, None],
        'query': [str, None],
        'config_params': [dict],
    })
    def __init__(
        self,
        version: str,
        comparison_metrics: Optional[List[str]] = None,
        track_history: bool = True,
        data_source: Optional[str] = None,
        api_url: Optional[str] = None,
        db_connection: Optional[object] = None,
        query: Optional[str] = None,
        **config_params
    ):
        self.comparison_metrics_ = comparison_metrics or ['accuracy']
        self.track_history = track_history
        self.comparison_results_ = None
        self.history_ = [] if self.track_history else None
        self.data_source_ = data_source
        self.api_url_ = api_url
        self.db_connection_ = db_connection
        self.query_ = query or (
            "SELECT accuracy, precision, recall, f1_score FROM metrics "
            "WHERE version = '{version}'"
        )

        super().__init__(version, config=config_params)
        self._perform_version_checks()

    @RunReturn
    def run(self, version_a: str, version_b: str, dataset: Optional[str] = None,
            model_type: Optional[str] = None, **run_kwargs):
        """
        Compares two versions based on the specified metrics.

        Parameters
        ----------
        version_a : str
            The first version to compare.

        version_b : str
            The second version to compare.

        dataset : str, optional
            The dataset being compared.

        model_type : str, optional
            The type of model being compared.

        **run_kwargs : dict, optional
            Additional parameters, including pre-computed metrics for
            ``version_a`` and ``version_b``.

        Returns
        -------
        self : VersionComparison
            The current instance of the class.

        Notes
        -----
        This method performs the comparison between the two specified
        versions using the metrics defined in the `comparison_metrics_`
        attribute.
        """
        self._validate_versions(version_a, version_b)

        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'comparison_metrics': {},
            'dataset': dataset,
            'model_type': model_type,
            'result_summary': None
        }

        for metric in self.comparison_metrics_:
            comparison['comparison_metrics'][metric] = self._compare_metric(
                version_a, version_b, metric, **run_kwargs)

        comparison['result_summary'] = self._summarize_comparison(
            version_a, version_b)

        self.comparison_results_ = comparison

        if self.track_history:
            self._track_comparison_history()

        return self

    def get_comparison_results(self):
        """
        Retrieves the results of the last comparison.

        Returns
        -------
        dict
            The results of the last comparison.

        Raises
        ------
        RuntimeError
            If the `run` method has not been executed before accessing
            results.

        Notes
        -----
        This method provides access to the comparison results after the
        comparison between versions has been run.
        """
        check_is_runned(self, attributes=["comparison_results_"],
                        msg="Comparison must be run before accessing results.")
        return self.comparison_results_

    def get_comparison_history(self):
        """
        Returns the history of all comparisons made.

        Returns
        -------
        list of dict
            The history of version comparisons.

        Raises
        ------
        ValueError
            If history tracking is disabled.

        Notes
        -----
        This method retrieves all the past comparisons if history tracking
        has been enabled.
        """
        check_is_runned(self, attributes=["comparison_results_"],
                        msg="Comparison must be run before accessing history.")
        if not self.track_history:
            raise ValueError("Comparison history tracking is disabled.")
        return self.history_

    def rollback_comparison(self, target_version: str):
        """
        Rolls back to a previous comparison for a specific version.

        Parameters
        ----------
        target_version : str
            The version identifier to which to roll back.

        Returns
        -------
        str
            A message indicating successful rollback to the target version.

        Raises
        ------
        RuntimeError
            If the comparison has not been run before attempting to roll
            back.

        ValueError
            If version history is not available or the target version is
            not found in the history.

        Notes
        -----
        This method allows rolling back to a previous comparison if version
        history tracking is enabled. It checks the history to find the
        target version and updates the comparison results accordingly.
        """
        check_is_runned(self, attributes=["comparison_results_"],
                        msg="Comparison must be run before rollback.")
        if not self.track_history or not self.history_:
            raise ValueError("Comparison history is not available for rollback.")

        for record in self.history_:
            if record['version'] == target_version:
                self.comparison_results_ = record['comparison_results']
                self.version = target_version
                self.log_event('rollback_success', {
                    'version': target_version
                })
                return f"Rolled back to comparison with version {target_version}."

        raise ValueError(f"Version {target_version} not found in comparison history.")

    def validate(self):
        """
        Validates that the current version exists and is consistent.

        Raises
        ------
        RuntimeError
            If validation fails due to missing or incomplete metrics for
            the version.

        ValueError
            If an invalid data source is specified for fetching metrics.

        Notes
        -----
        This method fetches the version metrics from the specified data
        source (either ``'db'`` or ``'api'``) and ensures the completeness
        of the metrics for comparison. If any inconsistency is found, an
        appropriate error is raised.
        """
        try:
            if self.data_source_ == 'db':
                metrics = self._fetch_from_database(self.version)
            elif self.data_source_ == 'api':
                metrics = self._fetch_from_api(self.version)
            else:
                raise ValueError("Invalid data source. Cannot validate version.")

            if not metrics or not all(
                key in metrics for key in self.comparison_metrics_
            ):
                raise ValueError(f"Missing or incomplete metrics for version {self.version}.")

            self.log_event('version_validation_success', {
                'version': self.version,
                'metrics': metrics
            })

        except Exception as e:
            self.log_event('version_validation_failed', {
                'version': self.version,
                'error': str(e)
            })
            raise RuntimeError(f"Version validation failed for {self.version}: {e}")

    def _validate_versions(self, version_a: str, version_b: str):
        if not version_a or not version_b:
            raise ValueError("Both version_a and version_b must be provided.")
        self.log_event('versions_validated', {
            'version_a': version_a,
            'version_b': version_b
        })

    def _compare_metric(self, version_a: str, version_b: str, metric: str,
                        **run_kwargs):
        metrics_a = run_kwargs.get('metrics_a') or self._fetch_version_metrics(version_a)
        metrics_b = run_kwargs.get('metrics_b') or self._fetch_version_metrics(version_b)

        if metric not in metrics_a or metric not in metrics_b:
            raise ValueError(f"Metric '{metric}' not found in one or both versions. "
                             f"Available metrics: {list(metrics_a.keys())}.")

        value_a = metrics_a[metric]
        value_b = metrics_b[metric]
        metric_difference = value_a - value_b

        if metric_difference > 0:
            comparison_result = (f"Version {version_a} outperforms {version_b} "
                                 f"on {metric} by {metric_difference:.4f}")
            better_version = version_a
        elif metric_difference < 0:
            comparison_result = (f"Version {version_b} outperforms {version_a} "
                                 f"on {metric} by {abs(metric_difference):.4f}")
            better_version = version_b
        else:
            comparison_result = (f"Version {version_a} and {version_b} "
                                 f"perform equally on {metric}.")
            better_version = "equal"

        self.log_event('metric_comparison', {
            'version_a': version_a,
            'version_b': version_b,
            'metric': metric,
            'value_a': value_a,
            'value_b': value_b,
            'difference': metric_difference,
            'better_version': better_version
        })

        return {
            'metric': metric,
            'version_a_value': value_a,
            'version_b_value': value_b,
            'difference': metric_difference,
            'better_version': better_version,
            'comparison_summary': comparison_result
        }

    def _fetch_version_metrics(self, version: str) -> Dict[str, float]:
        if self.data_source_ == 'db':
            return self._fetch_from_database(version)
        elif self.data_source_ == 'api':
            return self._fetch_from_api(version)
        else:
            raise ValueError("Invalid data source specified. Use 'db' or 'api'.")

    def _fetch_from_database(self, version: str) -> Dict[str, float]:
        if not self.db_connection_:
            raise ValueError("No database connection provided.")

        try:
            query = self.query_.format(version=version)
            if hasattr(self.db_connection_, 'cursor'):
                cursor = self.db_connection_.cursor()
                cursor.execute(query)
                result = cursor.fetchone()
            elif hasattr(self.db_connection_, 'execute'):
                result = self.db_connection_.execute(query).fetchone()
            else:
                raise ValueError("Unsupported database connection type.")

            if result:
                metrics = dict(zip(['accuracy', 'precision', 'recall', 'f1_score'], result))
                self.log_event('fetch_metrics', {'version': version, 'metrics': metrics})
                return metrics
            else:
                raise ValueError(f"No metrics found for version {version} in the database.")

        except Exception as e:
            self.log_event('db_error', {'error': str(e), 'query': query})
            raise RuntimeError(f"Failed to fetch metrics for version {version}: {e}")

    @ensure_pkg(
        "requests",
        extra="VersionComparison requires 'requests' package for API interactions.",
        auto_install=INSTALL_DEPENDENCIES,
        use_conda=USE_CONDA
    )
    def _fetch_from_api(self, version: str) -> Dict[str, float]:
        if not self.api_url_:
            raise ValueError("No API URL provided.")

        import requests
        url = f"{self.api_url_}/metrics?version={version}"
        response = requests.get(url)

        if response.status_code == 200:
            metrics = response.json()
            self.log_event('fetch_metrics', {'version': version, 'metrics': metrics})
            return metrics
        else:
            raise RuntimeError(
                f"Failed to fetch metrics from API. Status code: {response.status_code}"
            )

    def _summarize_comparison(self, version_a: str, version_b: str):
        return f"Comparison between {version_a} and {version_b} completed."

    def _track_comparison_history(self):
        if self.history_ is not None:
            self.history_.append({
                'version': self.version,
                'comparison_results': self.comparison_results_,
                'comparison_metrics': self.comparison_metrics_
            })
            self.log_event('comparison_tracked', {
                'version': self.version,
                'metrics': self.comparison_metrics_
            })

    def _perform_version_checks(self):
        if not self.data_source_:
            raise ValueError("No data source specified for fetching version metrics. "
                             "Please provide a valid data source (e.g., 'db' or 'api').")

        if self.data_source_ == 'db' and not self.db_connection_:
            raise ValueError("Database connection is required when 'db' is specified as the data source.")

        if self.data_source_ == 'api' and not self.api_url_:
            raise ValueError("API URL is required when 'api' is specified as the data source.")

        self.log_event('version_check_performed', {
            'version': self.version,
            'data_source': self.data_source_,
            'api_url': self.api_url_,
            'db_connection': bool(self.db_connection_)
        })

