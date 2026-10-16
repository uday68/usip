import os
import sys
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import config

def compute_file_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset_integrity(dataset_root: Path = config.dataset_root) -> Dict[str, Any]:
    """
    Rigorously verifies:
    1. Sample counts and split completeness
    2. Zero data leakage across train, val, and test splits (via MD5 content hashing)
    3. YOLO label format validity and coordinate bounds ([0, 1])
    4. Per-class instance distributions
    """
    print(f"\n[AuditDataset] Starting Forensic Dataset Audit at: {dataset_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    splits = ["train", "val", "test"]
    split_hashes: Dict[str, Dict[str, str]] = {s: {} for s in splits}
    split_images: Dict[str, List[Path]] = {s: [] for s in splits}
    split_instances: Dict[str, Counter] = {s: Counter() for s in splits}
    corrupt_labels: List[str] = []
    out_of_bounds_boxes: List[str] = []

    class_names = {0: "Shipwreck", 1: "Pipeline", 2: "Debris"}

    for s in splits:
        img_dir = dataset_root / "images" / s
        lbl_dir = dataset_root / "labels" / s

        if not img_dir.exists() or not lbl_dir.exists():
            raise FileNotFoundError(f"Missing images or labels directory for split '{s}' in {dataset_root}")

        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]])
        split_images[s] = imgs
        print(f"[AuditDataset] Scanning {len(imgs)} files in '{s}' split...", flush=True)

        for i, img_path in enumerate(imgs):
            if (i + 1) % 500 == 0:
                print(f"  - [{s}] processed {i + 1}/{len(imgs)}...", flush=True)

            # Hash content to detect exact image duplication across splits
            h = compute_file_md5(img_path)
            split_hashes[s][img_path.name] = h

            # Check corresponding label file
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                content = lbl_path.read_text(encoding="utf-8").strip()
                if content:
                    for line_idx, line in enumerate(content.splitlines()):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            corrupt_labels.append(f"{lbl_path}:{line_idx} invalid parts count ({len(parts)})")
                            continue
                        try:
                            cls_id = int(parts[0])
                            xc, yc, w, h_box = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        except ValueError as e:
                            corrupt_labels.append(f"{lbl_path}:{line_idx} parse error: {e}")
                            continue

                        if cls_id not in class_names:
                            corrupt_labels.append(f"{lbl_path}:{line_idx} unknown class ID {cls_id}")

                        # Check bounding box validity in [0, 1]
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h_box <= 1.0):
                            out_of_bounds_boxes.append(f"{lbl_path}:{line_idx} box out of [0, 1] bounds: {parts}")

                        split_instances[s][class_names.get(cls_id, f"Class-{cls_id}")] += 1

    # Check for MD5 collisions between splits (data leakage detection)
    leakage_train_val = set(split_hashes["train"].values()) & set(split_hashes["val"].values())
    leakage_train_test = set(split_hashes["train"].values()) & set(split_hashes["test"].values())
    leakage_val_test = set(split_hashes["val"].values()) & set(split_hashes["test"].values())

    leakage_detected = bool(leakage_train_val or leakage_train_test or leakage_val_test)

    audit_summary = {
        "dataset_root": str(dataset_root),
        "split_counts": {s: len(split_images[s]) for s in splits},
        "per_class_instances": {s: dict(split_instances[s]) for s in splits},
        "corrupt_labels_count": len(corrupt_labels),
        "out_of_bounds_count": len(out_of_bounds_boxes),
        "leakage": {
            "leakage_detected": leakage_detected,
            "train_val_overlap": len(leakage_train_val),
            "train_test_overlap": len(leakage_train_test),
            "val_test_overlap": len(leakage_val_test)
        }
    }

    print("="*60)
    print("=== USIP DATASET INTEGRITY & LEAKAGE AUDIT REPORT ===")
    print("="*60)
    for s in splits:
        total_imgs = len(split_images[s])
        total_inst = sum(split_instances[s].values())
        print(f"Split [{s.upper():<5}]: {total_imgs:>5} images | {total_inst:>5} target instances")
        for cls_name in ["Shipwreck", "Pipeline", "Debris"]:
            print(f"    - {cls_name:<10}: {split_instances[s][cls_name]:>4} instances")
    print("-"*60)
    print(f"Label Format Errors     : {len(corrupt_labels)}")
    print(f"Out of Bounds Boxes     : {len(out_of_bounds_boxes)}")
    print(f"Data Leakage (Train/Val): {len(leakage_train_val)} overlapping image hashes")
    print(f"Data Leakage (Train/Test): {len(leakage_train_test)} overlapping image hashes")
    print(f"Data Leakage (Val/Test) : {len(leakage_val_test)} overlapping image hashes")
    print("="*60)
    if not leakage_detected and len(corrupt_labels) == 0 and len(out_of_bounds_boxes) == 0:
        print("[AuditDataset] VERDICT: PASSED. Dataset is clean, valid, and strictly leakage-free.\n")
    else:
        print("[AuditDataset] VERDICT: WARNING. Inconsistencies detected.\n")

    return audit_summary

if __name__ == "__main__":
    audit_dataset_integrity()
