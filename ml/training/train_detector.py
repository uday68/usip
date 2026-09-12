import os
import shutil
import json
from pathlib import Path
import torch
from ultralytics import YOLO

def train_usip_detector(
    epochs: int = 10, 
    batch_size: int = 4, 
    imgsz: int = 640,
    device: str = "0",
    workers: int = 2
):
    print("=== USIP Known-Target Detector Training Pipeline ===")
    
    yaml_path = Path("D:/USIP_DATA/unified_sonar_dataset/dataset.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Unified dataset YAML not found at {yaml_path}. Run ml.data.compiler first!")

    # Check CUDA
    if device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize model
    model = YOLO("yolov8n.pt")

    # Training arguments calibrated for 4GB VRAM safety & acoustic physics
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=workers,
        project="experiments",
        name="usip_sonar_detector",
        exist_ok=True,
        amp=True,          # FP16 Automatic Mixed Precision for VRAM efficiency
        cache=False,       # Prevents RAM exhaustion
        hsv_h=0.015,       # minimal hue shift
        hsv_s=0.0,         # zero saturation jitter (pure acoustic intensity)
        hsv_v=0.2,         # moderate gain variation
        fliplr=0.5,        # port/starboard swap valid
        flipud=0.0,        # along-track reversal disabled
        mosaic=0.5,        # moderate mosaic
        close_mosaic=2,    # disable mosaic last 2 epochs for realistic acoustic shadow edges
        save=True,
        plots=True,
        verbose=True
    )

    # Save best weights to ml/detection/weights/best.pt
    exp_dir = Path("experiments/usip_sonar_detector")
    best_weights = exp_dir / "weights" / "best.pt"
    target_weights_dir = Path("ml/detection/weights")
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    target_weights = target_weights_dir / "best.pt"

    if best_weights.exists():
        shutil.copy(best_weights, target_weights)
        print(f"Successfully copied best weights to {target_weights}")
    else:
        last_weights = exp_dir / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy(last_weights, target_weights)
            print(f"Copied last weights to {target_weights}")

    # Evaluate on held-out test split
    print("\n=== Evaluating Model on Held-Out Test Sites & Sequences ===")
    test_metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        plots=True,
        project="reports",
        name="evaluation"
    )

    # Compile structured metrics report
    report_dir = Path("reports/evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_summary = {
        "dataset": "USIP Unified Sonar Dataset",
        "model_architecture": "YOLOv8n (Decoupled Head, Anchor-Free)",
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "device": str(device),
        "classes": ["Shipwreck", "Pipeline", "Debris"],
        "metrics": {
            "mAP50": float(test_metrics.box.map50),
            "mAP50_95": float(test_metrics.box.map),
            "precision": float(test_metrics.box.mp),
            "recall": float(test_metrics.box.mr),
        },
        "per_class_mAP50": {
            cls_name: float(test_metrics.box.class_result(i)[2])
            for i, cls_name in enumerate(["Shipwreck", "Pipeline", "Debris"])
        }
    }

    metrics_file = report_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"\nEvaluation metrics successfully written to {metrics_file}")
    print(f"Test mAP@50: {metrics_summary['metrics']['mAP50']:.4f}")
    print(f"Test Precision: {metrics_summary['metrics']['precision']:.4f}")
    print(f"Test Recall: {metrics_summary['metrics']['recall']:.4f}")

    return metrics_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    train_usip_detector(
        epochs=args.epochs, 
        batch_size=args.batch, 
        device=args.device,
        workers=args.workers
    )

