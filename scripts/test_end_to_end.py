import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.main import app

def run_verification():
    print("=== USIP Full System Verification Suite ===")
    client = TestClient(app)

    # 1. Health Check
    print("[1/6] Testing Health Endpoint...")
    res = client.get("/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f"      Status: {health_data['status']}, GPU: {health_data['gpu_available']} ({health_data['gpu_device']})")

    # 2. Dashboard UI Serving
    print("[2/6] Testing Dashboard Serving at '/'...")
    res = client.get("/")
    assert res.status_code == 200, f"Dashboard serving failed: {res.status_code}"
    assert "<title>USIP — Underwater Sonar Intelligence Platform</title>" in res.text
    print("      Dashboard HTML loaded successfully.")

    # 3. 1-Click Judge Demo Presets
    print("[3/6] Testing 1-Click Judge Demo Execution...")
    res = client.post("/api/analyze", data={"sample_id": "sample_shipwreck_1.png"})
    assert res.status_code == 200, f"Demo inference failed: {res.text}"
    analysis = res.json()
    print(f"      Analyzed image: {analysis['filename']} ({analysis['image_width']}x{analysis['image_height']})")
    print(f"      Total detections: {len(analysis['detections'])}, Latency: {analysis['summary']['latency_ms']} ms")
    survey_id = analysis["summary"]["survey_id"]

    # 4. Human-in-the-Loop Analyst Review
    print("[4/6] Testing Analyst Review Queue Submission...")
    if len(analysis["detections"]) > 0:
        cand_id = analysis["detections"][0]["id"]
        res = client.post(
            "/api/v1/reviews",
            json={
                "candidate_id": cand_id,
                "analyst_id": "LEAD-ANALYST-UDAY",
                "review_status": "confirmed",
                "notes": "Acoustic highlight shadow structure verified."
            }
        )
        assert res.status_code == 200
        print(f"      Submitted review for candidate {cand_id}: Confirmed.")

    # 5. Mission Intelligence Report Exports
    print("[5/6] Testing JSON and CSV Report Exports...")
    res_json = client.get(f"/api/v1/reports/{survey_id}/json")
    assert res_json.status_code == 200
    print("      JSON Report generated successfully.")

    res_csv = client.get(f"/api/v1/reports/{survey_id}/csv")
    assert res_csv.status_code == 200
    assert "Candidate ID" in res_csv.text
    print("      CSV Target Log generated successfully.")

    # 6. Direct Upload Inference
    print("[6/6] Testing Direct File Upload Inference...")
    sample_file = Path("demo_samples/sample_pipe_1.jpg")
    if sample_file.exists():
        with open(sample_file, "rb") as f:
            res_upload = client.post(
                "/api/v1/inference",
                files={"file": (sample_file.name, f.read(), "image/jpeg")},
                data={"survey_name": "AUV Inspection Run B"}
            )
            assert res_upload.status_code == 200
            print(f"      Direct file upload processed: {sample_file.name}")

    print("\n>>> ALL SYSTEM VERIFICATION CHECKS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    run_verification()

