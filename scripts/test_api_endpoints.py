import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app

def test_api():
    client = TestClient(app)

    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    health_data = res.json()
    print("Health Check Passed:", health_data["status"], "| GPU:", health_data["hardware"]["device"])

    # 2. Demo samples
    res = client.get("/api/demo-samples")
    assert res.status_code == 200, f"Demo samples failed: {res.status_code}"
    samples = res.json()
    print(f"Demo Samples: {len(samples)} samples available.")
    assert len(samples) >= 4

    # 3. Root dashboard HTML
    res = client.get("/")
    assert res.status_code == 200
    assert "USIP" in res.text
    print("Dashboard HTML Root Endpoint Passed.")

    # 4. Analyze endpoint with sample_id
    res = client.post("/api/analyze", data={"sample_id": "shipwreck_1"})
    assert res.status_code == 200, f"Analyze failed: {res.status_code}"
    analysis = res.json()
    print("Analysis Endpoint Passed!")
    print(f"  Target count: {len(analysis['detections'])}")
    print(f"  Processed image url: {analysis['processed_image_url']}")

    # 5. Export JSON
    analysis_id = analysis["summary"]["analysis_id"]
    res = client.get(f"/api/export/json?analysis_id={analysis_id}")
    assert res.status_code == 200
    assert "USIP_Mission_Report" in res.headers.get("content-disposition", "")
    print("Export JSON Endpoint Passed.")

    # 6. Export CSV
    res = client.get(f"/api/export/csv?analysis_id={analysis_id}")
    assert res.status_code == 200
    assert "Target_ID,Category" in res.text
    print("Export CSV Endpoint Passed.")

    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
