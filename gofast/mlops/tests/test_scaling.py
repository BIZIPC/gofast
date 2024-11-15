# -*- coding: utf-8 -*-

import time
import pytest
from unittest.mock import Mock
from gofast.mlops.scaling import (
    ScalingManager, DataPipelineScaler, ElasticScaler,
    partition_data_pipeline, get_system_workload, elastic_scale_logic
)

def test_scaling_manager_run():
    model = Mock()
    data = Mock()
    scaling_manager = ScalingManager(framework='tensorflow')
    scaling_manager.run(model, data)
    assert scaling_manager._is_runned
    assert scaling_manager.model_ is not None

def test_data_pipeline_scaler_run():
    def data_pipeline(partition):
        return [x * 2 for x in partition]

    data = list(range(100))
    scaler = DataPipelineScaler(num_partitions=4)
    results = scaler.run(data_pipeline, data)
    assert scaler._is_runned
    assert len(results) == 4

def test_elastic_scaler_run():
    def scale_up(metrics):
        print("Scaling up triggered.")

    def scale_down(metrics):
        print("Scaling down triggered.")

    scaler = ElasticScaler(
        scale_up_callback=scale_up,
        scale_down_callback=scale_down,
        monitoring_interval=1.0,
        min_scale_up_duration=2.0,
        min_scale_down_duration=2.0
    )
    # Run the scaler in a separate thread to allow stopping
    import threading
    thread = threading.Thread(target=scaler.run)
    thread.start()
    time.sleep(3)
    scaler.stop_monitoring()
    thread.join()
    assert scaler._is_runned

def test_partition_data_pipeline():
    def data_pipeline_fn(index):
        return [index] * 5

    partitions = partition_data_pipeline(
        data_pipeline_fn, num_partitions=3, partition_strategy='even'
    )
    assert len(partitions) == 3
    assert partitions[0] == [0, 0, 0, 0, 0]

def test_get_system_workload():
    workload = get_system_workload()
    assert 'cpu_usage' in workload
    assert 'memory_usage' in workload

def test_elastic_scale_logic():
    def scale_up(workload):
        print("Scaling up resources.")

    def scale_down(workload):
        print("Scaling down resources.")

    workload = {'cpu_usage': 85.0, 'memory_usage': 70.0}
    scale_up_thresholds = {'cpu_usage': 80.0, 'memory_usage': 80.0}
    scale_down_thresholds = {'cpu_usage': 30.0, 'memory_usage': 30.0}

    elastic_scale_logic(
        workload, scale_up_thresholds, scale_down_thresholds,
        scale_up_callback=scale_up, scale_down_callback=scale_down
    )
    # Since it's the first call, scaling might not happen due to duration checks
    # Subsequent calls should be tested with adjusted state


if __name__=='__main__': 
    pytest.main( [__file__])