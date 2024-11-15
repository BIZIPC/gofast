# -*- coding: utf-8 -*-
# test_inference.py

import pytest
import os
import json
import logging
import tempfile
import threading
import time

from unittest.mock import MagicMock, patch

# Import the classes from the gofast.mlops.inference module
from gofast.mlops.inference import (
    BatchInference,
    StreamingInference,
    MultiModelServing,
    InferenceParallelizer,
    InferenceCacheManager
)

# Disable logging during tests to keep the output clean
logging.disable(logging.CRITICAL)


def test_batch_inference():
    """Test the BatchInference class with default settings."""

    # Define a simple model with a predict method
    class SimpleModel:
        def predict(self, batch):
            # Simulate processing by summing values
            return [sum(item.values()) for item in batch]

    model = SimpleModel()

    # Prepare data
    data = [{'feature1': x, 'feature2': x * 2} for x in range(10)]

    # Initialize BatchInference
    batch_inference = BatchInference(
        model=model,
        batch_size=3,
        max_workers=2,
        gpu_enabled=False,
        enable_padding=True
    )

    # Run batch inference
    results = batch_inference.run(data)

    # Verify results
    expected_results = [item['feature1'] + item['feature2'] for item in data]
    assert results == expected_results


def test_batch_inference_with_preprocessing():
    """Test BatchInference with preprocessing and postprocessing functions."""

    # Define a simple model
    class SimpleModel:
        def predict(self, batch):
            return [item * 2 for item in batch]

    model = SimpleModel()

    # Prepare data
    data = [{'value': x} for x in range(5)]

    # Define preprocessing and postprocessing functions
    def preprocess_fn(item):
        return item['value']

    def postprocess_fn(prediction):
        return prediction + 1

    # Initialize BatchInference
    batch_inference = BatchInference(
        model=model,
        batch_size=2,
        max_workers=1,
        gpu_enabled=False,
        enable_padding=False
    )

    # Run batch inference
    results = batch_inference.run(
        data,
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn
    )

    # Verify results
    expected_results = [(item['value'] * 2) + 1 for item in data]
    assert results == expected_results


def test_batch_inference_error_handling():
    """Test BatchInference with error handling enabled and disabled."""

    # Define a model that raises an exception
    class ErrorModel:
        def predict(self, batch):
            raise ValueError("Model error")

    model = ErrorModel()
    data = [{'feature1': x} for x in range(5)]

    # Test with error handling enabled
    batch_inference = BatchInference(
        model=model,
        batch_size=2,
        max_workers=1,
    )
    results = batch_inference.run(data)
    assert results == []

    # Test with error handling disabled
    batch_inference.handle_errors = False
    with pytest.raises(ValueError):
        batch_inference.run(data)


def test_batch_inference_gpu():
    """Test BatchInference with GPU acceleration."""

    # Check if PyTorch and CUDA are available
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available, skipping GPU test.")
    except ImportError:
        pytest.skip("PyTorch is not installed, skipping GPU test.")

    import torch.nn as nn

    # Define a simple PyTorch model
    class SimpleTorchModel(nn.Module):
        def __init__(self):
            super(SimpleTorchModel, self).__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleTorchModel()

    # Prepare data
    data = [{'feature1': x, 'feature2': x * 2} for x in range(5)]

    # Initialize BatchInference with GPU enabled
    batch_inference = BatchInference(
        model=model,
        batch_size=2,
        max_workers=1,
        gpu_enabled=True
    )

    # Run batch inference
    results = batch_inference.run(data)

    # Verify results length
    assert len(results) == len(data)


def test_multi_model_serving_predict():
    """Test MultiModelServing with specified and auto-selected models."""

    # Define two simple models
    class ModelV1:
        def predict(self, data):
            return "Prediction from Model V1"

    class ModelV2:
        def predict(self, data):
            return "Prediction from Model V2"

    models = {'model_v1': ModelV1(), 'model_v2': ModelV2()}
    traffic_split = {'model_v1': 0.7, 'model_v2': 0.3}
    fallback_models = {'model_v1': ['model_v2']}

    # Initialize MultiModelServing
    multi_model_serving = MultiModelServing(
        models=models,
        traffic_split=traffic_split,
        fallback_models=fallback_models
    )

    data = {'input': [1, 2, 3]}

    # Predict without specifying model
    prediction = multi_model_serving.run(data)
    assert prediction in ["Prediction from Model V1", "Prediction from Model V2"]

    # Predict specifying model
    prediction = multi_model_serving.run(data, model_name='model_v1')
    assert prediction == "Prediction from Model V1"

    # Predict specifying a non-existent model
    with pytest.raises(ValueError):
        multi_model_serving.run(data, model_name='model_x')


def test_multi_model_serving_fallback():
    """Test MultiModelServing with error handling and fallback models."""

    # Define models where one raises an exception
    class ModelV1:
        def predict(self, data):
            raise ValueError("Model V1 failed")

    class ModelV2:
        def predict(self, data):
            return "Prediction from Model V2"

    models = {'model_v1': ModelV1(), 'model_v2': ModelV2()}
    fallback_models = {'model_v1': ['model_v2']}

    # Initialize MultiModelServing
    multi_model_serving = MultiModelServing(
        models=models,
        fallback_models=fallback_models,
        error_handling=True,
        retry_attempts=1
    )

    data = {'input': [1, 2, 3]}

    # Predict with a model that fails and should fallback
    prediction = multi_model_serving.run(data, model_name='model_v1')
    assert prediction == "Prediction from Model V2"


def test_multi_model_serving_latency():
    """Test MultiModelServing with latency-based model selection."""

    # Define models with artificial latency
    class ModelFast:
        def predict(self, data):
            time.sleep(0.1)
            return "Fast Model Prediction"

    class ModelSlow:
        def predict(self, data):
            time.sleep(0.5)
            return "Slow Model Prediction"

    models = {'fast_model': ModelFast(), 'slow_model': ModelSlow()}
    performance_metrics = {
        'fast_model': {'latency': 0.1},
        'slow_model': {'latency': 0.5}
    }

    # Initialize MultiModelServing with latency threshold
    multi_model_serving = MultiModelServing(
        models=models,
        performance_metrics=performance_metrics,
        latency_threshold=0.2
    )

    data = {'input': [1, 2, 3]}

    # Predict without specifying model
    prediction = multi_model_serving.run(data)
    assert prediction == "Fast Model Prediction"


def test_inference_parallelizer_threads():
    """Test InferenceParallelizer using threading."""

    # Define a simple model
    class SimpleModel:
        def predict(self, batch):
            return [sum(item.values()) for item in batch]

    model = SimpleModel()
    data = [{'input1': x, 'input2': x * 2} for x in range(10)]

    # Initialize InferenceParallelizer with threading
    parallelizer = InferenceParallelizer(
        model=model,
        parallel_type='threads',
        max_workers=2,
        batch_size=2
    )

    # Run parallel inference
    results = parallelizer.run(data)

    # Verify results
    expected_results = [item['input1'] + item['input2'] for item in data]
    assert results == expected_results


def test_inference_parallelizer_processes():
    """Test InferenceParallelizer using multiprocessing."""

    # Define a simple model
    class SimpleModel:
        def predict(self, batch):
            return [sum(item.values()) for item in batch]

    model = SimpleModel()
    data = [{'input1': x, 'input2': x * 2} for x in range(10)]

    # Initialize InferenceParallelizer with multiprocessing
    parallelizer = InferenceParallelizer(
        model=model,
        parallel_type='processes',
        max_workers=2,
        batch_size=2
    )

    # Run parallel inference
    results = parallelizer.run(data)

    # Verify results
    expected_results = [item['input1'] + item['input2'] for item in data]
    assert results == expected_results


def test_inference_parallelizer_error_handling():
    """Test InferenceParallelizer with error handling enabled."""

    # Define a model that raises an exception
    class ErrorModel:
        def predict(self, batch):
            raise ValueError("Model error")

    model = ErrorModel()
    data = [{'input1': x} for x in range(5)]

    # Initialize InferenceParallelizer
    parallelizer = InferenceParallelizer(
        model=model,
        handle_errors=True
    )

    # Run parallel inference
    results = parallelizer.run(data)

    # Verify that results are empty due to error handling
    assert results == []


def test_inference_cache_manager():
    """Test InferenceCacheManager with default settings."""

    # Define a simple model
    class SimpleModel:
        def __init__(self):
            self.call_count = 0

        def predict(self, data):
            self.call_count += 1
            return sum(data.values())

    model = SimpleModel()
    cache_manager = InferenceCacheManager(
        model=model,
        cache_size=100,
        eviction_policy='LRU'
    )

    data = {'input1': 1.0, 'input2': 2.0}

    # First prediction, should increase call_count
    result1 = cache_manager.run(data)
    assert result1 == 3.0
    assert model.call_count == 1

    # Second prediction with same data, should use cache
    result2 = cache_manager.run(data)
    assert result2 == 3.0
    assert model.call_count == 1  # call_count unchanged


def test_inference_cache_manager_persistent():
    """Test InferenceCacheManager with persistent caching."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, 'cache.pkl')

        # Define a simple model
        class SimpleModel:
            def __init__(self):
                self.call_count = 0

            def predict(self, data):
                self.call_count += 1
                return sum(data.values())

        model = SimpleModel()
        cache_manager = InferenceCacheManager(
            model=model,
            cache_size=100,
            eviction_policy='LRU',
            persistent_cache_path=cache_file
        )

        data = {'input1': 1.0, 'input2': 2.0}

        # First prediction
        result1 = cache_manager.run(data)
        assert result1 == 3.0
        assert model.call_count == 1

        # Save cache to disk
        cache_manager._save_persistent_cache()

        # Initialize a new cache manager and load cache
        model2 = SimpleModel()
        cache_manager2 = InferenceCacheManager(
            model=model2,
            cache_size=100,
            eviction_policy='LRU',
            persistent_cache_path=cache_file
        )
        cache_manager2._load_persistent_cache()

        # Predict with same data, should use cached result
        result2 = cache_manager2.run(data)
        assert result2 == 3.0
        assert model2.call_count == 0  # call_count unchanged


def test_inference_cache_manager_custom_hash():
    """Test InferenceCacheManager with a custom hash function."""

    # Define a simple model
    class SimpleModel:
        def __init__(self):
            self.call_count = 0

        def predict(self, data):
            self.call_count += 1
            return sum(data.values())

    model = SimpleModel()

    # Define a custom hash function
    def custom_hash_fn(data):
        return hash(sum(data.values()))

    cache_manager = InferenceCacheManager(
        model=model,
        cache_size=100,
        eviction_policy='LRU',
        custom_hash_fn=custom_hash_fn
    )

    data1 = {'input1': 1.0, 'input2': 2.0}
    data2 = {'input1': 2.0, 'input2': 1.0}

    # First prediction
    result1 = cache_manager.run(data1)
    assert result1 == 3.0
    assert model.call_count == 1

    # Second prediction with different data but same sum
    result2 = cache_manager.predict(data2)
    assert result2 == 3.0
    assert model.call_count == 1  # call_count unchanged


# Re-enable logging after tests
logging.disable(logging.NOTSET)

if __name__=='__main__': 
    pytest.main([__file__])

if __name__=='__main__': 
    pytest.main([__file__])