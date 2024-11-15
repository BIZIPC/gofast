# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock, patch

from gofast.mlops.versioning import (
    ModelVersionControl,
    DatasetVersioning,
    PipelineVersioning,
    VersionComparison
)

import numpy as np


def test_dataset_versioning():
    """
    Test the DatasetVersioning class.
    """
    # Initialize with valid parameters
    dataset_version = DatasetVersioning(version='v1.0', dataset_name='test_dataset')
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)

    # Fit the dataset versioning
    dataset_version.fit(X, y)

    # Retrieve metadata
    metadata = dataset_version.get_metadata()
    assert metadata['version'] == 'v1.0'
    assert metadata['dataset_name'] == 'test_dataset'
    assert metadata['n_samples'] == 100
    assert metadata['n_features'] == 10

    # Get version history
    history = dataset_version.get_version_history()
    assert len(history) == 1
    assert history[0]['version'] == 'v1.0'

    # Validate
    dataset_version.validate()

    # Rollback should raise ValueError because only one version exists
    with pytest.raises(ValueError):
        dataset_version.rollback('v0.9')

    # Fit a new version and test rollback
    dataset_version_new = DatasetVersioning(version='v1.1', dataset_name='test_dataset')
    dataset_version_new.fit(X, y)
    dataset_version_new.history_ = dataset_version.history_ + dataset_version_new.history_

    # Rollback to previous version
    dataset_version_new.rollback('v1.0')
    assert dataset_version_new.version == 'v1.0'


def test_pipeline_versioning():
    """
    Test the PipelineVersioning class.
    """
    # Initialize with valid parameters
    pipeline_version = PipelineVersioning(version='v1.0', pipeline_name='test_pipeline')

    # Run the pipeline versioning
    config = {
        'model_version': 'v1.0',
        'dataset_version': 'v1.0',
        'params': {'learning_rate': 0.01}
    }
    pipeline_version.run(config=config)

    # Retrieve pipeline tag
    tag = pipeline_version.get_pipeline_tag()
    assert tag['version'] == 'v1.0'
    assert tag['pipeline_name'] == 'test_pipeline'
    assert tag['config'] == config

    # Compare versions
    comparison = pipeline_version.compare_versions('v1.1')
    assert 'Comparing current version' in comparison

    # Validate
    pipeline_version.validate()

    # Rollback should raise ValueError because no history is available
    with pytest.raises(ValueError):
        pipeline_version.rollback('v0.9')

    # Add to history and rollback
    pipeline_version.history_ = [pipeline_version.pipeline_tag_]
    pipeline_version.rollback('v1.0')
    assert pipeline_version.version == 'v1.0'


def test_version_comparison():
    """
    Test the VersionComparison class.
    """
    # Mock the database connection and methods
    db_connection_mock = MagicMock()
    db_connection_mock.cursor.return_value.execute.return_value.fetchone.return_value = (
        0.9, 0.8, 0.7, 0.6
    )

    # Initialize with valid parameters
    version_comp = VersionComparison(
        version='v1.0',
        comparison_metrics=['accuracy', 'precision'],
        data_source='db',
        db_connection=db_connection_mock
    )

    # Mock the _fetch_from_database method to return predefined metrics
    version_comp._fetch_from_database = MagicMock(return_value={
        'accuracy': 0.9,
        'precision': 0.8,
        'recall': 0.7,
        'f1_score': 0.6
    })

    # Run the comparison
    version_comp.run(version_a='v1.0', version_b='v2.0')

    # Retrieve comparison results
    results = version_comp.get_comparison_results()
    assert results['version_a'] == 'v1.0'
    assert results['version_b'] == 'v2.0'
    assert 'comparison_metrics' in results

    # Validate
    version_comp.validate()

    # Rollback should raise ValueError because no history is available
    with pytest.raises(ValueError):
        version_comp.rollback_comparison('v0.9')

    # Add to history and rollback
    version_comp.history_ = [{'version': 'v1.0', 'comparison_results': results}]
    version_comp.rollback_comparison('v1.0')
    assert version_comp.version == 'v1.0'


def test_model_version_control():
    """
    Test the ModelVersionControl class.
    """
    # Mock subprocess.run to prevent actual git or dvc commands
    with patch('subprocess.run') as subprocess_run_mock:
        subprocess_run_mock.return_value.returncode = 0
        subprocess_run_mock.return_value.stdout = ''
        subprocess_run_mock.return_value.stderr = ''

        # Initialize with valid parameters
        model_version = ModelVersionControl(
            version='v1.0',
            repo_url='https://github.com/example/model_repo.git'
        )

        # Mock _version_exists_in_repo and _checkout_version methods
        model_version._version_exists_in_repo = MagicMock(return_value=True)
        model_version._checkout_version = MagicMock(return_value={})

        # Run the version control
        model_version.run(commit_message='Initial commit', tag='v1.0')

        # Compare versions
        comparison = model_version.compare_versions('v1.1')
        assert 'Comparing current version' in comparison

        # Track metrics
        model_version.track_metrics({'accuracy': 0.95, 'loss': 0.05})
        assert model_version.version_info_['metrics']['accuracy'] == 0.95

        # Validate
        model_version.validate()

        # Rollback
        rollback_message = model_version.rollback('v1.0')
        assert 'Rolled back to version v1.0 successfully.' == rollback_message

if __name__=='__main__': 
    pytest.main( [__file__])