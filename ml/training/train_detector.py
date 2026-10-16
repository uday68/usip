import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shutil
import json
import torch
from ultralytics import YOLO
from configs.settings import config
from ml.utils import set_deterministic_seeds
from ml.evaluation.eval_suite import run_evaluation_verification

def train_usip_detector(
    epochs: int = 25, 
    batch_size: int = config.batch_size, 
    imgsz: int = config.img_size,
    device: str = config.device,
    workers: int = 2,
    seed: int = config.seed
):
    print("=== USIP Known-Target Detector Training Pipeline ===")
    set_deterministic_seeds(seed)

    yaml_path = config.dataset_root / "dataset.yaml"
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

    # Initialize model from yolov8n base
    model = YOLO("yolov8n.pt")

    # Training arguments calibrated for 4GB VRAM safety & acoustic physics
    # Note: mosaic is reduced to 0.15 to preserve linear pipeline geometry across swath
    # close_mosaic is set to 8 so final epochs train strictly on authentic sonar geometry
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
        cos_lr=True,       # Cosine learning rate schedule
        warmup_epochs=3.0, # Smooth initial convergence
        lr0=0.01,
        lrf=0.01,
        hsv_h=0.015,       # Minimal hue shift
        hsv_s=0.0,         # Zero saturation jitter (pure acoustic intensity)
        hsv_v=0.2,         # Moderate gain variation
        fliplr=0.5,        # Port/starboard swap physically valid
        flipud=0.0,        # Along-track reversal disabled (violates towfish heading)
        mosaic=0.15,       # Low mosaic preserves continuous linear pipelines
        close_mosaic=8,    # Authentic shadow edges during convergence
        seed=seed,
        deterministic=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Save best weights to ml/detection/weights/best.pt
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect/experiments/usip_sonar_detector")
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        fallback_paths = [
            Path("runs/detect/experiments/usip_sonar_detector/weights/best.pt"),
            Path("experiments/usip_sonar_detector/weights/best.pt")
        ]
        for fb in fallback_paths:
            if fb.exists():
                best_weights = fb
                break

    target_weights_dir = Path("ml/detection/weights")
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    target_weights = target_weights_dir / "best.pt"

    if best_weights.exists():
        shutil.copy(best_weights, target_weights)
        print(f"Successfully copied best weights from {best_weights} to {target_weights}")
    else:
        last_weights = save_dir / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy(last_weights, target_weights)
            print(f"Copied last weights from {last_weights} to {target_weights}")

    # Evaluate on held-out test split using standard independent evaluation harness
    print("\n=== Running Independent Verification on Held-Out Test Set ===")
    verification_metrics = run_evaluation_verification(
        weights_path=target_weights,
        dataset_yaml=yaml_path,
        split="test",
        imgsz=imgsz,
        batch_size=batch_size,
        device=device,
        save_dir=Path("reports/evaluation")
    )

    return verification_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train USIP YOLOv8 Sonar Detector")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=config.batch_size)
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.seed)
    args = parser.parse_args()

    train_usip_detector(
        epochs=args.epochs, 
        batch_size=args.batch, 
        device=args.device,
        workers=args.workers,
        seed=args.seed
    )

