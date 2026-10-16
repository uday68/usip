import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import numpy as np
from ultralytics import YOLO
from configs.settings import config
from ml.utils import set_deterministic_seeds

def run_evaluation_verification(
    weights_path: Path = config.model_path,
    dataset_yaml: Path = config.dataset_root / "dataset.yaml",
    split: str = "test",
    imgsz: int = config.img_size,
    batch_size: int = config.batch_size,
    device: str = config.device,
    save_dir: Path = Path("reports/evaluation")
):
    """
    Independent, deterministic evaluation harness verifying model predictions on held-out test data.
    Directly computes mAP@50, mAP@50-95, Precision, Recall, and per-class metrics without modifying weights.
    """
    set_deterministic_seeds(config.seed)

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained detector weights not found at: {weights_path}")
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML configuration not found at: {dataset_yaml}")

    # Fallback to CPU if CUDA is requested but unavailable
    if device != "cpu" and not torch.cuda.is_available():
        print("[EvalSuite] CUDA not available on this machine. Falling back to CPU.")
        device = "cpu"
    else:
        print(f"[EvalSuite] Evaluating on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print(f"[EvalSuite] Loading model weights from: {weights_path}")
    model = YOLO(str(weights_path))

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EvalSuite] Running validation pass on '{split}' split ({dataset_yaml})...")

    results = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        plots=True,
        project="reports",
        name="evaluation_verification",
        exist_ok=True
    )

    class_names = ["Shipwreck", "Pipeline", "Debris"]
    per_class_results = {}
    for i, name in enumerate(class_names):
        try:
            res = results.box.class_result(i)
            # class_result returns: (precision, recall, map50, map)
            per_class_results[name] = {
                "precision": float(res[0]),
                "recall": float(res[1]),
                "mAP50": float(res[2]),
                "mAP50_95": float(res[3])
            }
        except Exception as e:
            per_class_results[name] = {"error": str(e)}

    verification_data = {
        "weights_path": str(weights_path),
        "dataset_yaml": str(dataset_yaml),
        "split_evaluated": split,
        "image_size": imgsz,
        "batch_size": batch_size,
        "device": device,
        "overall_metrics": {
            "mAP50": float(results.box.map50),
            "mAP50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "fitness": float(results.fitness) if hasattr(results, "fitness") else None
        },
        "per_class": per_class_results
    }

    out_file = save_dir / "verified_test_metrics.json"
    with open(out_file, "w") as f:
        json.dump(verification_data, f, indent=4)

    print("\n" + "="*60)
    print("=== USIP VERIFIED HELD-OUT TEST METRICS ===")
    print("="*60)
    print(f"Overall mAP@50    : {verification_data['overall_metrics']['mAP50']:.4f}")
    print(f"Overall mAP@50-95 : {verification_data['overall_metrics']['mAP50_95']:.4f}")
    print(f"Overall Precision : {verification_data['overall_metrics']['precision']:.4f}")
    print(f"Overall Recall    : {verification_data['overall_metrics']['recall']:.4f}")
    print("-"*60)
    print("Per-Class mAP@50 Breakdown:")
    for cls_name, vals in per_class_results.items():
        if "mAP50" in vals:
            print(f"  - {cls_name:<12}: mAP50 = {vals['mAP50']:.4f} | Recall = {vals['recall']:.4f} | Precision = {vals['precision']:.4f}")
    print("="*60)
    print(f"Detailed verified metrics written to: {out_file}\n")

    return verification_data

if __name__ == "__main__":
    run_evaluation_verification()
