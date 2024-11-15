# -*- coding: utf-8 -*-
# test_utils.py

import os
import json
import shutil
import logging
import tempfile
import random
import pytest
import numpy as np

from gofast.mlops.utils import (
    ConfigManager,
    ExperimentTracker,
    setup_logging,
    save_model,
    load_model,
    Timer,
    set_random_seed,
    EarlyStopping,
    calculate_metrics,
    DataVersioning,
    ParameterGrid,
    TrainTestSplitter,
    CrossValidator,
    PipelineBuilder,
    MetadataManager,
    save_pipeline,
    load_pipeline,
    log_model_summary,
    get_model_metadata,
)

# Disable logging during tests to keep output clean
logging.disable(logging.CRITICAL)


def test_config_manager_json():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.json')
        config_data = {'learning_rate': 0.001, 'batch_size': 32}

        # Save config
        config_manager = ConfigManager(config_file)
        config_manager.config = config_data
        config_manager.save_config()

        # Load config
        loaded_config = config_manager.load_config()
        assert loaded_config == config_data

        # Update config
        config_manager.update_config({'epochs': 10})
        assert config_manager.config['epochs'] == 10

def test_config_manager_yaml():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, 'config.yaml')
        config_data = {'learning_rate': 0.001, 'batch_size': 32}

        # Save config
        config_manager = ConfigManager(config_file, config_format='yaml')
        config_manager.config = config_data
        config_manager.save_config()

        # Load config
        loaded_config = config_manager.load_config()
        assert loaded_config == config_data

        # Update config
        config_manager.update_config({'epochs': 10})
        assert config_manager.config['epochs'] == 10

def test_experiment_tracker():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        experiment_name = 'test_experiment'
        tracker = ExperimentTracker(experiment_name, base_dir=tmpdir)

        # Log params
        params = {'learning_rate': 0.001, 'batch_size': 32}
        tracker.log_params(params)
        with open(tracker.params_file, 'r') as f:
            logged_params = json.load(f)
        assert logged_params == params

        # Log metrics
        metrics = {'accuracy': 0.95}
        tracker.log_metrics(metrics)
        with open(tracker.metrics_file, 'r') as f:
            logged_metrics = json.load(f)
        assert logged_metrics == metrics

        # Save artifact
        artifact_content = b"test artifact"
        artifact_path = os.path.join(tmpdir, 'artifact.txt')
        with open(artifact_path, 'wb') as f:
            f.write(artifact_content)
        tracker.save_artifact(artifact_path)
        saved_artifact_path = os.path.join(tracker.artifacts_dir, 'artifact.txt')
        with open(saved_artifact_path, 'rb') as f:
            saved_artifact_content = f.read()
        assert saved_artifact_content == artifact_content

def test_setup_logging():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, 'test.log')
        setup_logging(log_file=log_file, level=logging.INFO)

        logger = logging.getLogger('test_logger')
        logger.info('Test message')

        # Check if log file exists
        assert os.path.exists(log_file)
        with open(log_file, 'r') as f:
            logs = f.read()
        assert 'Test message' in logs

def test_save_and_load_model():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        model = {'param1': 1, 'param2': 2}
        model_path = os.path.join(tmpdir, 'model.pkl')

        # Save model
        save_model(model, model_path)
        assert os.path.exists(model_path)

        # Load model
        loaded_model = load_model(model_path)
        assert loaded_model == model

def test_timer():
    import time

    with Timer('sleep_test'):
        time.sleep(1)

    # No assertions needed, as we just want to ensure no exceptions occur

def test_set_random_seed():
    set_random_seed(123)
    random_value = random.randint(0, 100)
    np_random_value = np.random.randint(0, 100)
    set_random_seed(123)
    random_value_2 = random.randint(0, 100)
    np_random_value_2 = np.random.randint(0, 100)
    assert random_value == random_value_2
    assert np_random_value == np_random_value_2

def test_early_stopping():
    early_stopper = EarlyStopping(patience=2, delta=0.01, monitor='loss')
    losses = [0.5, 0.4, 0.4, 0.4, 0.3]
    for loss in losses:
        early_stopper(loss)
        if early_stopper.early_stop:
            break
    assert early_stopper.early_stop is True
    assert early_stopper.counter == 2

def test_calculate_metrics():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    metrics = calculate_metrics(y_true, y_pred)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics

def test_data_versioning():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write('Test data')

        metadata_file = os.path.join(tmpdir, 'metadata.json')
        data_versioning = DataVersioning(data_dir, metadata_file=metadata_file)
        data_versioning.generate_checksums()

        # No changes
        changes = data_versioning.check_for_changes()
        assert changes is False

        # Modify the file
        with open(file_path, 'w') as f:
            f.write('Modified data')

        # Changes detected
        changes = data_versioning.check_for_changes()
        assert changes is True

def test_parameter_grid():
    param_grid = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'optimizer': ['adam', 'sgd']
    }
    grid = ParameterGrid(param_grid)
    assert len(grid) == 8  # 2 * 2 * 2 combinations
    for params in grid:
        assert 'learning_rate' in params
        assert 'batch_size' in params
        assert 'optimizer' in params

def test_train_test_splitter():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    splitter = TrainTestSplitter(test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test = splitter.split(X, y)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25

def test_cross_validator():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    model = RandomForestClassifier(random_state=42)
    cross_validator = CrossValidator(model, cv=5)
    results = cross_validator.evaluate(X, y)
    assert 'mean_score' in results
    assert 'std_score' in results
    assert 'scores' in results
    assert len(results['scores']) == 5

def test_pipeline_builder():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    pipeline_builder = PipelineBuilder([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    pipeline = pipeline_builder.build()
    assert hasattr(pipeline, 'fit')
    assert hasattr(pipeline, 'predict')

def test_metadata_manager():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_file = os.path.join(tmpdir, 'metadata.json')
        metadata_manager = MetadataManager(metadata_file)
        metadata = {'model_name': 'test_model', 'version': 1}
        metadata_manager.save_metadata(metadata)
        loaded_metadata = metadata_manager.load_metadata()
        assert loaded_metadata == metadata

        # Update metadata
        metadata_manager.update_metadata({'version': 2})
        assert metadata_manager.get_metadata()['version'] == 2

def test_save_and_load_pipeline():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = os.path.join(tmpdir, 'pipeline.pkl')

        # Save pipeline
        save_pipeline(pipeline, pipeline_path)
        assert os.path.exists(pipeline_path)

        # Load pipeline
        loaded_pipeline = load_pipeline(pipeline_path)
        assert hasattr(loaded_pipeline, 'fit')
        assert hasattr(loaded_pipeline, 'predict')

def test_log_model_summary():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    # This function logs the model summary, which may not be visible during testing
    # Ensure no exceptions are raised
    log_model_summary(model)

def test_get_model_metadata():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    metadata = get_model_metadata(model)
    assert 'class_name' in metadata
    assert 'module' in metadata
    assert 'params' in metadata
    assert metadata['class_name'] == 'LogisticRegression'

# Re-enable logging after tests
logging.disable(logging.NOTSET)

if __name__=='__main__': 
    pytest.main([__file__])
