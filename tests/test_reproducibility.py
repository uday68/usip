import pytest
import hashlib
from pathlib import Path
from configs.settings import config
from scripts.reproduce_evaluation import verify_file_md5, EXPECTED_WEIGHTS_MD5
from ml.evaluation.audit_dataset import audit_dataset_integrity
from ml.evaluation.error_analysis import compute_box_iou

def test_model_weights_cryptographic_hash():
    """Verify that detector weights strictly match the verified cryptographic reference signature."""
    weights_path = Path(config.model_path)
    assert weights_path.exists(), f"Model weights missing at {weights_path}"
    actual_md5 = verify_file_md5(weights_path)
    assert actual_md5 == EXPECTED_WEIGHTS_MD5, (
        f"Weights hash mismatch! Expected {EXPECTED_WEIGHTS_MD5}, got {actual_md5}"
    )

def test_box_iou_computation():
    """Verify standard IoU calculations on known bounding boxes."""
    # Perfect overlap
    boxA = (0.0, 0.0, 1.0, 1.0)
    boxB = (0.0, 0.0, 1.0, 1.0)
    assert compute_box_iou(boxA, boxB) == 1.0

    # No overlap
    boxC = (2.0, 2.0, 3.0, 3.0)
    assert compute_box_iou(boxA, boxC) == 0.0

    # Half overlap: boxA area=1.0, boxD area=1.0, intersection area=0.5, union=1.5 -> IoU = 1/3
    boxD = (0.5, 0.0, 1.5, 1.0)
    assert pytest.approx(compute_box_iou(boxA, boxD), 0.001) == 0.5 / 1.5

def test_dataset_split_zero_leakage():
    """Verify that dataset splits have zero MD5 content collisions."""
    if not config.dataset_root.exists():
        pytest.skip(f"Dataset root {config.dataset_root} not present on this machine.")

    audit = audit_dataset_integrity(config.dataset_root)
    assert audit["leakage"]["leakage_detected"] is False, "Cross-split data leakage detected!"
    assert audit["corrupt_labels_count"] == 0, f"Found {audit['corrupt_labels_count']} corrupt label files!"
    assert audit["out_of_bounds_count"] == 0, f"Found {audit['out_of_bounds_count']} out of bounds boxes!"
