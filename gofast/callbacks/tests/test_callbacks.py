# -*- coding: utf-8 -*-
import os 
import pytest
import numpy as np
from unittest.mock import MagicMock

from gofast.callbacks.data import BaseData, DataAugmentation, DataOps, DataLogging
from gofast.callbacks.model import EarlyStopping, LearningRateScheduler, ModelCheckpoint

class MockModel:
    def __init__(self):
        self.train_data = {
            'features': np.array([[1, 2], [3, 4]]),
            'labels': np.array([0, 1])
        }
        self.current_batch_data = {
            'features': np.array([[1, 2]]),
            'labels': np.array([0])
        }
        self.stop_training = False
        self.weights = None
        self.learning_rate = 0.01

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, lr):
        self.learning_rate = lr

def test_BaseData():
    model = MockModel()
    base_data_callback = BaseData(
        model=model,
        data_transformations=[lambda data: data],
        batch_transformations=[lambda data: data],
        epoch_operations=[lambda model, epoch: None],
        batch_operations=[lambda model, batch: None],
        verbose=1
    )
    base_data_callback.on_epoch_start(epoch=0, logs={})
    assert 0 in base_data_callback.data_statistics
    base_data_callback.on_batch_start(batch=0, logs={})
    base_data_callback.on_batch_end(batch=0, logs={})
    base_data_callback.on_epoch_end(epoch=0, logs={})
    # Check that data_statistics is populated
    assert 'epoch_start' in base_data_callback.data_statistics[0]
    # assert 'epoch_end' in base_data_callback.data_statistics[0]

def test_DataAugmentation():
    model = MockModel()

    def augmentation_function(data):
        data['features'] += 1
        return data

    data_augmentation_callback = DataAugmentation(
        model=model,
        augmentation_functions=[augmentation_function],
        verbose=1
    )

    # Simulate batch processing
    data_before = model.current_batch_data['features'].copy()
    data_augmentation_callback.on_batch_start(batch=0, logs={})
    assert np.array_equal(
        model.current_batch_data['features'],
        data_before + 1
    )

def test_DataOps():
    model = MockModel()

    def operation(data):
        data['features'] *= 2
        return data

    data_ops_callback = DataOps(
        model=model,
        data_operations=[operation],
        verbose=1
    )

    data_before = model.train_data['features'].copy()
    data_ops_callback.on_epoch_start(epoch=0, logs={})
    assert np.array_equal(
        model.train_data['features'],
        data_before * 2
    )

def test_DataLogging(tmpdir):
    import os
    import json
    model = MockModel()
    log_file = os.path.join(tmpdir, 'data_log.json')

    def statistics_function(data):
        return {'mean_feature_0': np.mean(data['features'][:, 0])}

    data_logging_callback = DataLogging(
        model=model,
        log_file=log_file,
        statistics_functions=[statistics_function],
        log_on_epoch_end=True,
        log_on_batch_end=False,
        verbose=1
    )

    data_logging_callback.on_epoch_end(epoch=0, logs={})
    assert os.path.exists(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 1
    data_info = json.loads(lines[0])
    assert data_info['epoch'] == 1  # Epochs are 1-indexed in logging
    assert 'data_statistics' in data_info
    assert 'mean_feature_0' in data_info['data_statistics']

def test_EarlyStopping():
    model = MockModel()
    model.stop_training = False
    model.get_weights = MagicMock(return_value='best_weights')
    model.set_weights = MagicMock()

    logs = {'val_loss': 0.5}
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=1,
        restore_best_weights=True
    )
    early_stopping_callback.set_model(model)
    early_stopping_callback.on_train_begin()

    # First epoch, best value is set
    early_stopping_callback.on_epoch_end(epoch=0, logs=logs)
    assert early_stopping_callback.best == 0.5
    # Second epoch, no improvement
    logs['val_loss'] = 0.5
    early_stopping_callback.on_epoch_end(epoch=1, logs=logs)
    assert early_stopping_callback.wait == 1
    # Third epoch, no improvement
    early_stopping_callback.on_epoch_end(epoch=2, logs=logs)
    assert model.stop_training is True
    model.set_weights.assert_called_with('best_weights')

def test_LearningRateScheduler():
    model = MockModel()
    initial_lr = model.learning_rate

    def schedule(epoch, lr):
        return lr * 0.1

    lr_scheduler = LearningRateScheduler(schedule=schedule, verbose=1)
    lr_scheduler.set_model(model)

    logs = {}
    for epoch in range(3):
        lr_scheduler.on_epoch_end(epoch, logs=logs)
        expected_lr = initial_lr * (0.1 ** (epoch + 1))
        assert model.learning_rate == pytest.approx(expected_lr)

class MockModel2:
    def __init__(self):
        self.stop_training = False

    def save(self, filepath):
        pass  # Mock save method

def test_ModelCheckpoint(tmpdir):
    model = MockModel2()
    model.save = MagicMock()
    filepath = os.path.join(tmpdir, 'best_model.h5')

    # Create the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    checkpoint.set_model(model)

    logs = {'val_loss': 0.5}

    # First epoch, should save the model
    checkpoint.on_epoch_end(epoch=0, logs=logs)
    model.save.assert_called_with(filepath)
    assert checkpoint.best == 0.5

    # Reset mock
    model.save.reset_mock()

    # Second epoch, no improvement
    logs['val_loss'] = 0.6
    checkpoint.on_epoch_end(epoch=1, logs=logs)
    model.save.assert_not_called()
    assert checkpoint.best == 0.5

    # Third epoch, improvement
    logs['val_loss'] = 0.4
    checkpoint.on_epoch_end(epoch=2, logs=logs)
    model.save.assert_called_with(filepath)
    assert checkpoint.best == 0.4

if __name__=="__main__": 
    pytest.main([__file__])