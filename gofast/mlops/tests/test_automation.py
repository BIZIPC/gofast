# -*- coding: utf-8 -*-
import pytest
import logging
import random
import time
from unittest.mock import patch, MagicMock
from gofast.mlops.automation import (
    AutomationManager,
    RetrainingScheduler,
    AirflowAutomation,
    KubeflowAutomation,
    KafkaAutomation,
    RabbitMQAutomation,
)
from gofast.mlops._config import INSTALL_DEPENDENCIES
from datetime import datetime

# Enable dependency installation during tests
INSTALL_DEPENDENCIES = True  # noqa

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions for tasks
def sample_task(task_name):
    logger.info(f"Executing task: {task_name}")

def retrain_model(model):
    logger.info(f"Retraining model: {model.__class__.__name__}")

def evaluate_model(model):
    return random.uniform(0.6, 1.0)  # Simulated performance score

def process_kafka_data(data):
    logger.info(f"Processing Kafka data: {data}")

def process_rabbitmq_data(data):
    logger.info(f"Processing RabbitMQ data: {data}")

# Sample model class
class SampleModel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<Model {self.name}>"

# Fixtures
@pytest.fixture
def automation_manager():
    manager = AutomationManager(max_workers=2, state_persistence_file="test_state.pkl")
    yield manager
    manager.shutdown()
    import os
    if os.path.exists("test_state.pkl"):
        os.remove("test_state.pkl")

@pytest.fixture
def retraining_scheduler():
    scheduler = RetrainingScheduler(max_workers=2)
    yield scheduler
    scheduler.shutdown()

@pytest.fixture
def airflow_manager():
    return AirflowAutomation(dag_id="test_dag", start_date=datetime(2024, 1, 1))

@pytest.fixture
def kubeflow_manager():
    return KubeflowAutomation(host="http://localhost:8080")

@pytest.fixture
def kafka_manager():
    return KafkaAutomation(kafka_servers=["localhost:9092"], topic="test_topic")

@pytest.fixture
def rabbitmq_manager():
    return RabbitMQAutomation(host="localhost", queue="test_queue")

# Tests for AutomationManager
def test_add_and_schedule_task(automation_manager):
    automation_manager.add_task("task1", sample_task, interval=1, args=("Task1",), retries=2)
    assert "task1" in automation_manager.tasks
    automation_manager.schedule_task("task1")
    assert automation_manager.tasks["task1"]["running"] == True
    time.sleep(2)
    automation_manager.cancel_task("task1")
    assert automation_manager.tasks["task1"]["running"] == False

def test_task_retries(automation_manager):
    def flaky_task():
        if not hasattr(flaky_task, "counter"):
            flaky_task.counter = 0
        flaky_task.counter += 1
        if flaky_task.counter < 3:
            raise ValueError("Intentional Failure")
        logger.info("Flaky task succeeded.")

    automation_manager.add_task("flaky_task", flaky_task, interval=1, retries=3)
    automation_manager.schedule_task("flaky_task")
    time.sleep(5)
    assert automation_manager.tasks["flaky_task"]["failures"] == 0
    automation_manager.cancel_task("flaky_task")

def test_state_persistence(automation_manager):
    automation_manager.add_task("task_persist", sample_task, interval=1, args=("PersistTask",), retries=1)
    automation_manager.schedule_task("task_persist")
    time.sleep(2)
    automation_manager.persist_state()
    new_manager = AutomationManager(max_workers=2, state_persistence_file="test_state.pkl")
    assert "task_persist" in new_manager.tasks
    assert new_manager.tasks["task_persist"]["running"] == True
    new_manager.cancel_task("task_persist")
    new_manager.shutdown()

# Tests for RetrainingScheduler
def test_schedule_retraining(retraining_scheduler):
    model = SampleModel("TestModel")
    retraining_scheduler.schedule_retraining(model, retrain_model, interval=1)
    assert f"retrain_{model.__class__.__name__}" in retraining_scheduler.tasks
    time.sleep(2)
    retraining_scheduler.cancel_task(f"retrain_{model.__class__.__name__}")
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["running"] == False

def test_trigger_retraining_on_decay(retraining_scheduler):
    model = SampleModel("DecayModel")
    with patch('gofast.mlops.automation.retrain_model') as mock_retrain:
        retraining_scheduler.trigger_retraining_on_decay(model, evaluate_model, decay_threshold=0.7)
        score = evaluate_model(model)
        if score < 0.7:
            mock_retrain.assert_called_once_with(model)
        else:
            mock_retrain.assert_not_called()

def test_monitor_model(retraining_scheduler):
    model = SampleModel("MonitorModel")
    with patch.object(retraining_scheduler, 'trigger_retraining_on_decay') as mock_trigger:
        retraining_scheduler.monitor_model(model, evaluate_model, decay_threshold=0.7, check_interval=1)
        time.sleep(2)
        assert mock_trigger.call_count >= 1
        retraining_scheduler.cancel_task(f"monitor_{model.__class__.__name__}")

def test_adjust_retraining_schedule(retraining_scheduler):
    model = SampleModel("AdjustModel")
    retraining_scheduler.schedule_retraining(model, retrain_model, interval=2)
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["interval"] == 2
    retraining_scheduler.adjust_retraining_schedule(model, new_interval=1)
    assert retraining_scheduler.tasks[f"retrain_{model.__class__.__name__}"]["interval"] == 1
    retraining_scheduler.cancel_task(f"retrain_{model.__class__.__name__}")

# Tests for AirflowAutomation
@patch("gofast.mlops.automation.PythonOperator")
@patch("gofast.mlops.automation.DAG")
def test_airflow_add_task(mock_dag, mock_python_operator, airflow_manager):
    mock_dag.return_value = MagicMock()
    mock_python_operator.return_value = MagicMock()

    task = airflow_manager.add_task_to_airflow("sample_task", sample_task, data="sample_data")
    mock_python_operator.assert_called_once_with(
        task_id="sample_task",
        python_callable=sample_task,
        op_kwargs={"data": "sample_data"},
        dag=airflow_manager.dag,
    )

@patch("gofast.mlops.automation.PythonOperator")
@patch("gofast.mlops.automation.DAG")
def test_airflow_schedule_task(mock_dag, mock_python_operator, airflow_manager):
    mock_dag.return_value = MagicMock()
    mock_python_operator.return_value = MagicMock()
    airflow_manager.add_task_to_airflow("sample_task", sample_task, data="sample_data")
    airflow_manager.schedule_airflow_task("sample_task")
    task = airflow_manager.dag.get_task("sample_task")
    task.execute.assert_called_once()

# KubeflowAutomation Tests
@patch("gofast.mlops.automation.Client")
@patch("gofast.mlops.automation.dsl.ContainerOp")
def test_kubeflow_create_pipeline(mock_container_op, mock_client, kubeflow_manager):
    mock_client.return_value = MagicMock()
    mock_container_op.return_value = MagicMock()
    kubeflow_manager.create_kubeflow_pipeline("test_pipeline", "sample_task", sample_task, data="sample_data")
    mock_container_op.assert_called_once()

# KafkaAutomation Tests
@patch("gofast.mlops.automation.KafkaConsumer")
def test_kafka_process_message(mock_kafka_consumer, kafka_manager):
    mock_kafka_consumer.return_value = MagicMock()
    kafka_manager.process_kafka_message(process_kafka_data)
    message = MagicMock()
    message.value = b"test_message"
    kafka_manager.consumer.__iter__.return_value = [message]
    kafka_manager.process_kafka_message(process_kafka_data)
    assert process_kafka_data.called_once_with(b"test_message")

# RabbitMQAutomation Tests
@patch("gofast.mlops.automation.pika.BlockingConnection")
def test_rabbitmq_process_message(mock_pika_connection, rabbitmq_manager):
    mock_channel = MagicMock()
    mock_pika_connection.return_value.channel.return_value = mock_channel
    rabbitmq_manager.process_rabbitmq_message(process_rabbitmq_data)
    method_frame = MagicMock()
    header_frame = MagicMock()
    body = b"test_message"
    rabbitmq_manager.channel.basic_consume.call_args[1]["on_message_callback"](
        mock_channel, method_frame, header_frame, body
    )
    assert process_rabbitmq_data.called_once_with(b"test_message")

# Run the test suite
if __name__ == "__main__":
    pytest.main([__file__])
