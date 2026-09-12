import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.preprocessing.sonar_proc import load_and_preprocess_sonar
from backend.inference.detector import detect_known_targets
from backend.inference.anomaly_engine import discover_anomalies
from backend.main import run_analysis_pipeline

def test_pipeline():
    print("Testing USIP End-to-End Pipeline on real sample...")
    demo_img = Path("demo_samples/sample_shipwreck_1.png")
    if not demo_img.exists():
        print("Demo sample does not exist!")
        return False

    resp = run_analysis_pipeline(demo_img, demo_img.name)
    print(f"Analysis successful!")
    print(f"Image dimensions: {resp.image_width}x{resp.image_height}")
    print(f"Total detections: {resp.summary['total_detections']}")
    print(f"Known targets: {resp.summary['known_targets_count']}")
    print(f"Anomaly candidates: {resp.summary['anomaly_candidates_count']}")
    print(f"Highest priority: {resp.summary['highest_priority']}")
    for d in resp.detections:
        print(f"  [{d.priority}] {d.id} | {d.class_name} | Conf: {d.confidence*100:.0f}% | Shadow: {d.acoustic_metrics.shadow_evidence_level} | Seabed: {d.acoustic_metrics.seafloor_similarity}")
    return True

if __name__ == "__main__":
    assert test_pipeline()
