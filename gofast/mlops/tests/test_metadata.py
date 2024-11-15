# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 20:46:03 2024

@author: Daniel
"""

# test_metadata.py

import os
import json
import pytest
import shutil
from datetime import datetime, timedelta

# Assume these are imported from the gofast.mlops.metadata module
from gofast.mlops.metadata import (
    MetadataManager,
    LineageTracker,
    AuditLogger,
    ReproducibilityEnsurer,
    PerformanceTracker,
    log_metadata,
    retrieve,
    compare,
    audit,
    sync_with_cloud,
    validate_schema,
    track_experiment,
    prune_old,
)

# Mock implementations or fixtures can be added here if necessary

# Constants for testing
TEST_METADATA_TYPE = 'test_model'
TEST_METADATA = {'accuracy': 0.95, 'loss': 0.05}
TEST_METADATA_UPDATED = {'accuracy': 0.96, 'loss': 0.04}
TEST_USER = 'test_user'
TEST_CHANGE_DESCRIPTION = 'Updated accuracy and loss metrics.'
TEST_BUCKET_NAME = 'test-bucket'
TEST_MONGO_URI = 'mongodb://localhost:27017/test_db'
TEST_EXPERIMENT_ID = 'exp_test_001'
TEST_SCHEMA = {
    'type': 'object',
    'properties': {
        'accuracy': {'type': 'number'},
        'loss': {'type': 'number'},
    },
    'required': ['accuracy', 'loss']
}


# Pytest fixtures for setup and teardown
@pytest.fixture(scope='module')
def setup_local_storage():
    # Setup actions before tests run
    os.makedirs('test_metadata', exist_ok=True)
    yield
    # Teardown actions after tests run
    shutil.rmtree('test_metadata', ignore_errors=True)

@pytest.fixture
def clear_test_files():
    # Remove test files before each test
    yield
    for file in os.listdir():
        if file.startswith('test_model') or file.startswith('experiment_'):
            os.remove(file)

# Test cases for MetadataManager
def test_metadata_manager_save_and_load(clear_test_files):
    # Provide the required 'metadata_store' argument
    manager = MetadataManager(metadata_store='test_metadata_store.json')
    manager.save_metadata(TEST_METADATA_TYPE, TEST_METADATA)
    loaded_metadata = manager.load_metadata(TEST_METADATA_TYPE)
    assert loaded_metadata == TEST_METADATA, "Loaded metadata does not match saved metadata."


def test_metadata_manager_update(clear_test_files):
    # Provide the required 'metadata_store' argument
    manager = MetadataManager(metadata_store='test_metadata_store.json')
    manager.save_metadata(TEST_METADATA_TYPE, TEST_METADATA)
    manager.update_metadata(TEST_METADATA_TYPE, TEST_METADATA_UPDATED)
    loaded_metadata = manager.load_metadata(TEST_METADATA_TYPE)
    assert loaded_metadata == TEST_METADATA_UPDATED, "Metadata update failed."

# Test cases for LineageTracker
def test_lineage_tracker_record_and_retrieve(clear_test_files):
    tracker = LineageTracker(lineage_store='test_lineage.json')
    # Use existing method to log lineage information
    tracker.log_data_ingestion(
        data_version='v1.0.0',
        source='test_source.csv',
        dependencies=['dep1.csv'],
        tags=['test']
    )
    # Access the lineage information directly
    lineage_info = tracker.lineage[-1]  # Get the last logged entry
    assert lineage_info['stage'] == 'data_ingestion', "Lineage entry stage mismatch."
    assert lineage_info['data_version'] == 'v1.0.0', "Lineage data_version mismatch."
    assert lineage_info['source'] == 'test_source.csv', "Lineage source mismatch."

    tracker.record_lineage(TEST_METADATA_TYPE, {'parent_model': 'model_v1'})
    lineage_info = tracker.retrieve_lineage(TEST_METADATA_TYPE)
    assert lineage_info == {'parent_model': 'model_v1'}, "Lineage information mismatch."

# Test cases for AuditLogger
def test_audit_logger_log_and_retrieve(clear_test_files):
    logger = AuditLogger(retry_policy= {"retries": 3, "backoff": 2})
    logger.log_change(TEST_METADATA_TYPE, TEST_USER, TEST_CHANGE_DESCRIPTION)
    logs = logger.retrieve_logs(TEST_METADATA_TYPE)
    assert len(logs) == 1, "Audit log entry was not recorded."
    assert logs[0]['user'] == TEST_USER, "Audit log user mismatch."


# Updated Test Case for LineageTracker
def test_lineage_tracker_log_and_persist(clear_test_files):
    tracker = LineageTracker(lineage_store='test_lineage.json')
    tracker.log_data_ingestion(
        data_version='v1.0.0',
        source='test_source.csv',
        dependencies=['dep1.csv'],
        tags=['test']
    )
    tracker.log_model_training(
        model_version='v1.0.0',
        hyperparameters={'learning_rate': 0.01},
        environment={'python_version': '3.8'},
        tags=['test']
    )
    tracker.log_deployment(
        model_version='v1.0.0',
        deployment_time='2024-10-10 12:00:00',
        environment={'cloud_provider': 'AWS'},
        access_permissions={'deployed_by': TEST_USER},
        tags=['test']
    )
    # Check that lineage entries have been logged
    assert len(tracker.lineage) == 3, "Lineage entries not properly logged."

    # Verify that the lineage has been persisted to the specified store
    assert os.path.exists('test_lineage.json'), "Lineage file was not created."

    # Load the persisted lineage and verify contents
    with open('test_lineage.json', 'r') as f:
        lineage_data = json.load(f)

    assert len(lineage_data) == 3, "Persisted lineage data is incorrect."
    # Verify specific entries
    data_ingestion_entry = lineage_data[0]
    assert data_ingestion_entry['stage'] == 'data_ingestion', "First lineage entry is not data ingestion."
    assert data_ingestion_entry['data_version'] == 'v1.0.0', "Data version mismatch in lineage."

# Updated Test Case for AuditLogger
def test_audit_logger_log_and_retrieve2(clear_test_files):
    logger = AuditLogger(storage_path='test_audit_logs.json',
                         retry_policy={"retries": 3, "backoff": 2})
    logger.log_decision(
        decision='metadata_update',
        user=TEST_USER,
        timestamp='2024-10-10 12:00:00',
        rationale=TEST_CHANGE_DESCRIPTION,
        severity='high',
        tags=['test']
    )
    logs = logger.get_audit_log()
    assert len(logs) == 1, "Audit log entry was not recorded."
    assert logs[0]['user'] == TEST_USER, "Audit log user mismatch."
    assert logs[0]['decision'] == 'metadata_update', "Audit log decision mismatch."

    # Verify that the logs are persisted if storage_path is specified
    assert os.path.exists('test_audit_logs.json'), "Audit log file was not created."

    # Load the persisted logs and verify contents
    with open('test_audit_logs.json', 'r') as f:
        log_data = json.load(f)

    assert len(log_data) == 1, "Persisted audit log data is incorrect."
    assert log_data[0]['user'] == TEST_USER, "Persisted audit log user mismatch."
    assert log_data[0]['decision'] == 'metadata_update', "Persisted audit log decision mismatch."

# Test cases for ReproducibilityEnsurer
def test_reproducibility_ensurer_verify_hash(clear_test_files):
    ensurer = ReproducibilityEnsurer()
    ensurer.save_metadata_with_hash(TEST_METADATA_TYPE, TEST_METADATA)
    is_valid = ensurer.verify_metadata_hash(TEST_METADATA_TYPE)
    assert is_valid, "Metadata hash verification failed."

def test_performance_tracker_log_and_compare(clear_test_files):
    # Provide the required 'alert_threshold' argument
    tracker = PerformanceTracker(alert_threshold=0.05, )
    tracker.log_performance(TEST_METADATA_TYPE, TEST_METADATA)
    performance_data = tracker.retrieve_performance(TEST_METADATA_TYPE)
    assert performance_data == TEST_METADATA, "Performance data mismatch."

    # Compare performance using the updated metadata
    comparison = tracker.compare_performance(TEST_METADATA_TYPE, TEST_METADATA_UPDATED)
    expected_change = TEST_METADATA_UPDATED['accuracy'] - TEST_METADATA['accuracy']
    assert comparison['accuracy']['change'] == expected_change, "Performance comparison failed."

# Test cases for log_metadata
def test_log_metadata(clear_test_files):
    result = log_metadata(
        metadata_type=TEST_METADATA_TYPE,
        metadata=TEST_METADATA,
        storage_backend='local'
    )
    assert result == f"Metadata of type '{TEST_METADATA_TYPE}' logged successfully.", "log_metadata failed."

# Test cases for retrieve
def test_retrieve(clear_test_files):
    log_metadata(
        metadata_type=TEST_METADATA_TYPE,
        metadata=TEST_METADATA,
        storage_backend='local'
    )
    retrieved_metadata = retrieve(
        metadata_type=TEST_METADATA_TYPE,
        storage_backend='local'
    )
    assert retrieved_metadata == TEST_METADATA, "retrieve function failed to get the correct metadata."

# Test cases for compare
def test_compare(clear_test_files):
    metadata_old = {'accuracy': 0.90, 'loss': 0.10}
    metadata_new = {'accuracy': 0.95, 'loss': 0.05}
    differences = compare(metadata_old, metadata_new)
    assert differences['accuracy'] == 0.05, "Accuracy difference calculation failed."
    assert differences['loss'] == -0.05, "Loss difference calculation failed."

# Test cases for audit
def test_audit(clear_test_files):
    message = audit(
        metadata_type=TEST_METADATA_TYPE,
        user=TEST_USER,
        change_description=TEST_CHANGE_DESCRIPTION,
        storage_backend='local'
    )
    assert message == f"Audit log successfully recorded for {TEST_METADATA_TYPE} by {TEST_USER}."

# Test cases for sync_with_cloud
@pytest.mark.skip(reason="Requires cloud setup and credentials.")
def test_sync_with_cloud(clear_test_files):
    metadata = TEST_METADATA
    message = sync_with_cloud(
        metadata=metadata,
        cloud_provider='aws',
        retries=1,
        batch_size=1,
        bucket_name=TEST_BUCKET_NAME,
        aws_credentials={
            'aws_access_key_id': 'fake_access_key',
            'aws_secret_access_key': 'fake_secret_key'
        }
    )
    assert message == "Metadata synced successfully with AWS.", "sync_with_cloud failed."

# Test cases for validate_schema
def test_validate_schema(clear_test_files):
    is_valid = validate_schema(
        metadata=TEST_METADATA,
        schema=TEST_SCHEMA,
        auto_correct=False
    )
    assert is_valid, "validate_schema failed on valid metadata."

def test_validate_schema_with_autocorrect(clear_test_files):
    invalid_metadata = {'accuracy': '0.95', 'loss': 0.05}
    is_valid = validate_schema(
        metadata=invalid_metadata,
        schema=TEST_SCHEMA,
        auto_correct=True,
        correction_log='test_corrections.log'
    )
    assert is_valid, "validate_schema failed to auto-correct metadata."
    assert os.path.exists('test_corrections.log'), "Correction log was not created."

# Test cases for track_experiment
def test_track_experiment(clear_test_files):
    configuration = {'optimizer': 'adam'}
    hyperparameters = {'learning_rate': 0.001}
    performance_metrics = {'accuracy': 0.95}
    message = track_experiment(
        experiment_id=TEST_EXPERIMENT_ID,
        configuration=configuration,
        hyperparameters=hyperparameters,
        performance_metrics=performance_metrics,
        storage_backend='local'
    )
    assert message == f"Experiment metadata tracked successfully for {TEST_EXPERIMENT_ID}."
    assert os.path.exists(f"experiment_{TEST_EXPERIMENT_ID}.json"), "Experiment metadata file was not created."

# Test cases for prune_old
def test_prune_old(clear_test_files):
    # Create old metadata file
    old_file = f"{TEST_METADATA_TYPE}_old_v1.json"
    with open(old_file, 'w') as f:
        json.dump(TEST_METADATA, f)
    # Modify the file's last modified time to be older than retention_days
    old_time = datetime.now() - timedelta(days=31)
    os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
    # Prune old files
    message = prune_old(
        metadata_type=TEST_METADATA_TYPE,
        retention_days=30,
        storage_backend='local'
    )
    assert message == "Old metadata pruned successfully based on a retention policy of 30 days."
    assert not os.path.exists(old_file), "Old metadata file was not pruned."

if __name__=='__main__': 
    pytest.main([__file__])