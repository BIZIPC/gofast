# -*- coding: utf-8 -*-
# test_deployment.py

import pytest
import logging
import sys
from gofast.mlops.deployment import (
    ModelExporter,
    APIDeployment,
    CloudDeployment,
    ABTesting
)

from gofast.mlops._config import INSTALL_DEPENDENCIES 

INSTALL_DEPENDENCIES =True # noqa # install dependencies during the test

# Disable logging during tests to keep output clean
logging.disable(sys.maxsize)

# Mock model for testing ModelExporter
class MockModel:
    def __init__(self):
        pass

    def predict(self, x):
        return x

    def state_dict(self):
        return {}

    def eval(self):
        pass

    def __call__(self, x):
        return x

# Test ModelExporter class
def test_model_exporter():
    model = MockModel()
    exporter = ModelExporter(model, model_name='test_model', versioning=True)

    # Test exporting to ONNX
    try:
        exporter.export_to_onnx('test_model.onnx', input_shape=(1, 3, 224, 224))
    except Exception as e:
        pytest.fail(f"Export to ONNX failed: {e}")

    # Test exporting to Torch
    try:
        exporter.export_to_torch('test_model.pt')
    except Exception as e:
        pytest.fail(f"Export to Torch failed: {e}")

    # Test compress_model method with quantization
    try:
        exporter.compress_model(method='quantization')
    except Exception as e:
        pytest.fail(f"Model quantization failed: {e}")

    # Test version control
    try:
        exporter.version_control('test_model_v')
    except Exception as e:
        pytest.fail(f"Version control failed: {e}")

# Mock model for testing APIDeployment
class MockAPIDeploymentModel:
    def predict(self, input_data):
        return {"prediction": "mock_prediction"}

# Test APIDeployment with FastAPI
def test_api_deployment_fastapi():
    model = MockAPIDeploymentModel()
    api_deployment = APIDeployment(model, 'test_model', api_type='FastAPI')

    api_deployment.create_api()
    api_deployment.health_check()
    api_deployment.version_control(model_version=1)

    # Use FastAPI's TestClient to test endpoints
    from fastapi.testclient import TestClient
    client = TestClient(api_deployment.app)

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

    # Test model_version endpoint
    response = client.get("/model_version")
    assert response.status_code == 200
    assert response.json()["version"] == 1

    # Test predict endpoint
    response = client.post("/predict", json={"input_data": {"key": "value"}})
    assert response.status_code == 200
    assert response.json()["result"]["prediction"] == "mock_prediction"

# Test APIDeployment with Flask
def test_api_deployment_flask():
    model = MockAPIDeploymentModel()
    api_deployment = APIDeployment(model, 'test_model', api_type='Flask')

    api_deployment.create_api()
    api_deployment.health_check()
    api_deployment.version_control(model_version=1)

    # Use Flask's test client
    with api_deployment.app.test_client() as client:
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "healthy"

        # Test model_version endpoint
        response = client.get("/model_version")
        assert response.status_code == 200
        assert response.get_json()["version"] == 1

        # Test predict endpoint
        response = client.post("/predict", json={"input_data": {"key": "value"}})
        assert response.status_code == 200
        assert response.get_json()["result"]["prediction"] == "mock_prediction"

# Mock configurations for CloudDeployment
def test_cloud_deployment_aws(mocker):
    model = MockModel()
    deployer = CloudDeployment(model, platform='aws', model_name='test_model')

    # Mock boto3 client
    mock_sagemaker_client = mocker.Mock()
    mocker.patch('boto3.client', return_value=mock_sagemaker_client)

    config = {
        'model_data': 's3://my-bucket/model.tar.gz',
        'role_arn': 'arn:aws:iam::123456789012:role/SageMakerRole'
    }

    # Mock responses for create_model, create_endpoint_config, create_endpoint
    mock_sagemaker_client.create_model.return_value = {}
    mock_sagemaker_client.create_endpoint_config.return_value = {}
    mock_sagemaker_client.create_endpoint.return_value = {}

    try:
        deployer.deploy_to_aws(config)
    except Exception as e:
        pytest.fail(f"Deployment to AWS failed: {e}")

    # Assert that the methods were called
    mock_sagemaker_client.create_model.assert_called()
    mock_sagemaker_client.create_endpoint_config.assert_called()
    mock_sagemaker_client.create_endpoint.assert_called()

def test_cloud_deployment_gcp(mocker):
    model = MockModel()
    deployer = CloudDeployment(model, platform='gcp', model_name='test_model')

    # Mock google.cloud.aiplatform module
    mock_gcp_ai = mocker.Mock()
    mocker.patch('google.cloud.aiplatform', mock_gcp_ai)

    config = {
        'project': 'my-gcp-project',
        'model_path': 'gs://my-bucket/model/'
    }

    # Mock methods
    mock_gcp_ai.init.return_value = None
    mock_model = mocker.Mock()
    mock_gcp_ai.Model.upload.return_value = mock_model
    mock_model.deploy.return_value = mocker.Mock(resource_name='test_endpoint')

    try:
        deployer.deploy_to_gcp(config)
    except Exception as e:
        pytest.fail(f"Deployment to GCP failed: {e}")

    # Assert that methods were called
    mock_gcp_ai.init.assert_called_with(project='my-gcp-project', location='us-central1')
    mock_gcp_ai.Model.upload.assert_called()
    mock_model.deploy.assert_called()

def test_cloud_deployment_azure(mocker):
    model = MockModel()
    deployer = CloudDeployment(model, platform='azure', model_name='test_model')

    # Mock azureml.core and related classes
    mock_Workspace = mocker.Mock()
    mock_Model = mocker.Mock()
    mock_AksCompute = mocker.Mock()
    mocker.patch('azureml.core.Workspace', mock_Workspace)
    mocker.patch('azureml.core.Model', mock_Model)
    mocker.patch('azureml.core.compute.AksCompute', mock_AksCompute)

    config = {
        'workspace_config': 'config.json',
        'model_path': 'model/',
        'aks_cluster_name': 'my-aks-cluster',
        'inference_config': mocker.Mock(),
        'deployment_config': mocker.Mock()
    }

    # Mock methods
    mock_workspace = mocker.Mock()
    mock_Workspace.from_config.return_value = mock_workspace
    mock_model = mocker.Mock()
    mock_Model.register.return_value = mock_model
    mock_deployment_target = mocker.Mock()
    mock_AksCompute.return_value = mock_deployment_target
    mock_service = mocker.Mock()
    mock_Model.deploy.return_value = mock_service
    mock_service.wait_for_deployment.return_value = None
    mock_service.scoring_uri = 'http://test_uri'

    try:
        deployer.deploy_to_azure(config)
    except Exception as e:
        pytest.fail(f"Deployment to Azure failed: {e}")

    # Assert that methods were called
    mock_Workspace.from_config.assert_called_with(path='config.json')
    mock_Model.register.assert_called()
    mock_AksCompute.assert_called_with(workspace=mock_workspace, name='my-aks-cluster')
    mock_Model.deploy.assert_called()

# Test ABTesting class
def test_ab_testing():
    # Mock models with predict method
    class MockModelV1:
        def predict(self, request):
            return "ModelV1 Prediction"

    class MockModelV2:
        def predict(self, request):
            return "ModelV2 Prediction"

    model_v1 = MockModelV1()
    model_v2 = MockModelV2()

    ab_test = ABTesting(
        model_v1, model_v2, split_ratio=0.5,
        min_split_ratio=0.1, max_split_ratio=0.9,
        performance_threshold=0.05, traffic_increment=0.1,
        graceful_degradation=True
    )

    # Test routing traffic
    request = {}
    for _ in range(10):
        response = ab_test.route_traffic(request)
        assert response in ["ModelV1 Prediction", "ModelV2 Prediction"]

    # Test evaluate_performance
    performance_metrics = {"model_v1": 0.9, "model_v2": 0.8}
    ab_test.evaluate_performance(performance_metrics)
    assert ab_test.split_ratio == 0.6  # Increased towards model_v1

    performance_metrics = {"model_v1": 0.7, "model_v2": 0.85}
    ab_test.evaluate_performance(performance_metrics)
    assert ab_test.split_ratio == 0.5  # Decreased towards model_v2

    # Test graceful degradation
    performance_metrics = {"model_v1": 0.4, "model_v2": 0.4}
    ab_test.evaluate_performance(performance_metrics)
    assert ab_test.split_ratio == 0.5  # Reset due to underperformance

    # Test rollback
    ab_test.split_ratio = 0.85
    ab_test.rollback()
    assert ab_test.split_ratio == 1.0  # All traffic to model_v1

    ab_test.split_ratio = 0.15
    ab_test.rollback()
    assert ab_test.split_ratio == 0.0  # All traffic to model_v2

# Re-enable logging after tests
logging.disable(logging.NOTSET)

if __name__=='__main__': 
    pytest.main([__file__])