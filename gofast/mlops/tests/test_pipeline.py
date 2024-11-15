# -*- coding: utf-8 -*-
import pytest
import logging
import time
from datetime import datetime

from gofast.mlops.pipeline import (
    PipelineStep,
    PipelineManager,
    PipelineOptimizer,
    ResourceMonitor,
    ResourceManager,
    PrefectOrchestrator, 
    create_pipeline,
    reconfigure_pipeline_on_the_fly,
    execute_step_conditionally,
    run_parallel_subpipelines,
    split_data_for_multitask_pipeline,
    rollback_to_previous_state,
    smart_retry_with_backoff,
    AirflowOrchestrator,
    Pipeline, 
)

def test_pipeline_manager():
    """
    Test the PipelineManager class.
    """

    # Define simple functions for pipeline steps
    def increment(data):
        return data + 1

    def double(data):
        return data * 2

    # Create pipeline steps
    step1 = PipelineStep(name='increment', func=increment)
    step2 = PipelineStep(name='double', func=double, dependencies=['increment'])

    # Initialize PipelineManager and add steps
    manager = PipelineManager()
    manager.add_step(step1)
    manager.add_step(step2)

    # Run the pipeline
    result = manager.run(initial_data=1)

    # Assert the final result
    assert result == 4  # (1 + 1) * 2 = 4

    # Get and check metadata
    metadata = manager.get_metadata()
    assert metadata['increment']['status'] == 'success'
    assert metadata['double']['status'] == 'success'
    assert metadata['increment']['output'] == 2
    assert metadata['double']['output'] == 4

def test_resource_manager():
    """
    Test the ResourceManager class.
    """

    # Initialize ResourceManager and run
    manager = ResourceManager()
    manager.run()

    # Get initial system resources
    resources = manager.get_system_resources()
    initial_cpu = resources['available_cpu_cores']
    initial_memory = resources['available_memory_gb']

    # Allocate resources
    cpu_allocated = manager.allocate_cpu(1)
    memory_allocated = manager.allocate_memory(1024 * 1024 * 1024)  # 1 GB

    # Assert allocations were successful
    assert cpu_allocated is True
    assert memory_allocated is True

    # Get resources after allocation
    resources_after = manager.get_system_resources()
    assert resources_after['available_cpu_cores'] == initial_cpu - 1
    assert resources_after['available_memory_gb'] <= initial_memory - 0.9  # Allow some margin

    # Release resources
    manager.release_resources(cpu_cores=1, memory=1024 * 1024 * 1024)

    # Get resources after release
    resources_final = manager.get_system_resources()
    assert resources_final['available_cpu_cores'] == initial_cpu
    assert resources_final['available_memory_gb'] >= initial_memory - 0.1  # Allow some margin

def test_resource_monitor():
    """
    Test the ResourceMonitor class.
    """

    # Initialize ResourceMonitor and run
    monitor = ResourceMonitor()
    monitor.run()

    # Start monitoring
    monitor.start_monitoring()

    # Simulate workload and record usage
    for _ in range(3):
        monitor.record_usage()
        time.sleep(1)

    # Stop monitoring
    monitor.stop_monitoring()

    # Assert that usage data was recorded
    assert len(monitor.cpu_usage_) >= 3
    assert len(monitor.memory_usage_) >= 3

def test_pipeline_optimizer():
    """
    Test the PipelineOptimizer class.
    """
    # Define a dummy step function
    def train_model(**kwargs):
        learning_rate = kwargs.get('learning_rate', 0.1)
        n_estimators = kwargs.get('n_estimators', 100)
        accuracy = 0.5 + 0.1 * learning_rate + 0.001 * n_estimators
        return {'accuracy': accuracy}

    # Create PipelineManager and add the training step
    manager = PipelineManager()
    step = PipelineStep(name='TrainModel', func=train_model)
    manager.add_step(step)

    # Initialize PipelineOptimizer and run
    optimizer = PipelineOptimizer(manager)
    optimizer.run()

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300]
    }

    # Tune hyperparameters
    best_params = optimizer.tune_hyperparameters(
        'TrainModel', param_grid, n_trials=3, eval_metric='accuracy'
    )

    # Assert that best parameters were found
    assert 'learning_rate' in best_params
    assert 'n_estimators' in best_params

def test_prefect_orchestrator():
    """
    Test the PrefectOrchestrator class.
    """
    try:
        import prefect  # noqa
    except ImportError:
        pytest.skip("Prefect is not installed")

    # Define a simple pipeline function
    def increment(data):
        return data + 1

    # Create PipelineManager and add a step
    manager = PipelineManager()
    step = PipelineStep(name='increment', func=increment)
    manager.add_step(step)

    # Initialize PrefectOrchestrator and run
    orchestrator = PrefectOrchestrator(manager)
    orchestrator.run(flow_name='test_flow')

    # Assert that the flow was created
    assert orchestrator.flow_ is not None

    # Schedule the pipeline
    orchestrator.schedule_pipeline('0 0 * * *')  # Daily at midnight

    # Monitor the pipeline (logs message)
    orchestrator.monitor_pipeline()

def test_airflow_orchestrator():
    """
    Test the AirflowOrchestrator class.
    """
    try:
        import airflow  # noqa
    except ImportError:
        pytest.skip("Airflow is not installed")

    # Define a simple pipeline function
    def increment(data):
        return data + 1

    # Create PipelineManager and add a step
    manager = PipelineManager()
    step = PipelineStep(name='increment', func=increment)
    manager.add_step(step)

    # Initialize AirflowOrchestrator and run
    orchestrator = AirflowOrchestrator(manager)
    orchestrator.run(
        dag_id='test_dag',
        start_date=datetime(2023, 1, 1),
        schedule_interval='@daily'
    )

    # Assert that the DAG was created
    assert orchestrator.dag_ is not None

    # Schedule the pipeline
    orchestrator.schedule_pipeline('@hourly')

    # Monitor the pipeline (logs message)
    orchestrator.monitor_pipeline()

# Configure logging for the tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample functions for pipeline steps
def sample_load_data(data):
    logger.info("Loading data...")
    return [1, 2, 3, 4, 5]

def sample_process_data(data):
    logger.info("Processing data...")
    return [x * 2 for x in data]

def sample_fallback_step(data):
    logger.info("Executing fallback step...")
    return "Fallback executed."

# Tests for create_pipeline
def test_create_pipeline(pipeline_steps):
    pipeline = create_pipeline(steps=pipeline_steps, parallel=False)
    assert isinstance(pipeline, PipelineManager)

# Tests for reconfigure_pipeline_on_the_fly
def test_reconfigure_pipeline_on_the_fly(pipeline_manager):
    def new_train_model(data, epochs=3):
        return f"Model re-trained with {epochs} epochs."

    reconfigure_pipeline_on_the_fly(pipeline_manager, "TrainModel", new_train_model, {"epochs": 3})
    step = pipeline_manager.get_step("TrainModel")
    assert step.params["epochs"] == 3

# Tests for execute_step_conditionally
def test_execute_step_conditionally(pipeline_manager):
    conditional_step = PipelineStep(name="ConditionalStep", func=sample_fallback_step)
    pipeline_manager.add_step(conditional_step)
    execute_step_conditionally(
        pipeline_manager,
        "ConditionalStep",
        condition_func=lambda output: True,
        fallback_step=None,
    )
    status = pipeline_manager.step_metadata["ConditionalStep"]["status"]
    assert status == "success"

# Tests for run_parallel_subpipelines
def test_run_parallel_subpipelines():
    sub_pipeline_steps_1 = [
        PipelineStep(name="SubLoadData1", func=sample_load_data),
        PipelineStep(name="SubProcessData1", func=sample_process_data, dependencies=["SubLoadData1"]),
    ]
    sub_pipeline_steps_2 = [
        PipelineStep(name="SubLoadData2", func=sample_load_data),
        PipelineStep(name="SubProcessData2", func=sample_process_data, dependencies=["SubLoadData2"]),
    ]
    pipeline_manager = PipelineManager()
    run_parallel_subpipelines(pipeline_manager, [sub_pipeline_steps_1, sub_pipeline_steps_2])

# Tests for split_data_for_multitask_pipeline
def test_split_data_for_multitask_pipeline(pipeline_manager):
    data = list(range(100))
    split_ratios = [0.5, 0.5]
    tasks = ["TrainModel", "ValidateModel"]
    split_data_for_multitask_pipeline(data, split_ratios, tasks, pipeline_manager)
    assert len(pipeline_manager.get_step("TrainModel").params["data"]) == 50

# Tests for rollback_to_previous_state
def test_rollback_to_previous_state(pipeline_manager):
    pipeline_manager.step_metadata["ProcessData"]["status"] = "failed"
    rollback_to_previous_state(pipeline_manager, "LoadData")
    assert pipeline_manager.step_metadata["ProcessData"]["status"] == "rolled_back"

# Tests for smart_retry_with_backoff
def test_smart_retry_with_backoff(pipeline_manager):
    error_step = PipelineStep(name="ErrorStep", func=lambda data: ValueError("Test Error"))
    pipeline_manager.add_step(error_step)
    with pytest.raises(ValueError):
        smart_retry_with_backoff(pipeline_manager, "ErrorStep", max_retries=2)

# Sample function for pipeline steps
def sample_task_func(data):
    return f"Processed {data}"

# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])
