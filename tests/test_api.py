import pytest
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "gpu_available" in data
    assert "detector_model_loaded" in data

def test_demo_samples_list():
    response = client.get("/api/v1/demo/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)

def test_inference_synthetic_upload():
    # Generate dummy sonar image
    synthetic_sonar = np.random.normal(120, 20, (300, 300)).astype(np.uint8)
    synthetic_sonar[100:140, 100:140] = 250
    synthetic_sonar[100:140, 140:190] = 10
    _, buffer = cv2.imencode(".png", synthetic_sonar)

    response = client.post(
        "/api/v1/inference",
        files={"file": ("test_sonar_ping.png", io.BytesIO(buffer.tobytes()), "image/png")},
        data={"survey_name": "Automated Test Survey"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "survey_id" in data
    assert "image_id" in data
    assert "candidates" in data
    assert "quality_metrics" in data
    assert "inference_latency_ms" in data

