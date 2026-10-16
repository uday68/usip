import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import cv2
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import config
from ml.utils import set_deterministic_seeds

def compute_box_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea

def run_per_class_error_analysis(
    weights_path: Path = config.model_path,
    dataset_root: Path = config.dataset_root,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50
):
    print("\n" + "="*60)
    print("=== USIP DETAILED PER-CLASS ERROR ANALYSIS ===")
    print("="*60)
    set_deterministic_seeds(config.seed)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[ErrorAnalysis] Loading detector from: {weights_path} on {device}")
    model = YOLO(str(weights_path))

    test_img_dir = dataset_root / "images" / "test"
    test_lbl_dir = dataset_root / "labels" / "test"

    class_names = {0: "Shipwreck", 1: "Pipeline", 2: "Debris"}

    test_imgs = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    print(f"[ErrorAnalysis] Analyzing {len(test_imgs)} test images at conf={conf_threshold}, IoU={iou_threshold}...")

    stats = {
        cls_name: {
            "TP": 0, "FP": 0, "FN": 0,
            "gt_count": 0, "pred_count": 0,
            "gt_aspect_ratios": [],
            "pred_aspect_ratios": [],
            "fp_reasons": defaultdict(int) # background vs cross-class
        } for cls_name in class_names.values()
    }

    # Track pipeline-specific error breakdown
    pipeline_errors = {
        "missed_images": [],
        "low_iou_matches": [],  # detected but IoU < 0.5
        "false_alarms": []
    }

    for idx, img_path in enumerate(test_imgs):
        # Read GT
        gt_boxes = [] # (cls_id, x1, y1, x2, y2)
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            content = lbl_path.read_text(encoding="utf-8").strip()
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(1.0, xc + w / 2.0)
                        y2 = min(1.0, yc + h / 2.0)
                        gt_boxes.append((cid, x1, y1, x2, y2, w, h))
                        cls_name = class_names[cid]
                        stats[cls_name]["gt_count"] += 1
                        stats[cls_name]["gt_aspect_ratios"].append(w / (h + 1e-6))

        # Run inference
        results = model.predict(str(img_path), conf=conf_threshold, device=device, verbose=False)
        pred_boxes = [] # (cls_id, conf, x1, y1, x2, y2)
        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                c = float(box.conf[0])
                bx, by, bw, bh = box.xywhn[0].tolist()
                x1 = max(0.0, bx - bw / 2.0)
                y1 = max(0.0, by - bh / 2.0)
                x2 = min(1.0, bx + bw / 2.0)
                y2 = min(1.0, by + bh / 2.0)
                pred_boxes.append((cid, c, x1, y1, x2, y2, bw, bh))
                if cid in class_names:
                    cls_name = class_names[cid]
                    stats[cls_name]["pred_count"] += 1
                    stats[cls_name]["pred_aspect_ratios"].append(bw / (bh + 1e-6))

        # Match GT and Preds per class
        for cid, cls_name in class_names.items():
            cls_gt = [b for b in gt_boxes if b[0] == cid]
            cls_pred = [b for b in pred_boxes if b[0] == cid]

            matched_gt = set()
            for p in cls_pred:
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g in enumerate(cls_gt):
                    if g_idx in matched_gt:
                        continue
                    iou = compute_box_iou(p[2:6], g[1:5])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_threshold:
                    stats[cls_name]["TP"] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    stats[cls_name]["FP"] += 1
                    if best_iou > 0.1:
                        stats[cls_name]["fp_reasons"]["low_iou_overlap"] += 1
                        if cid == 1:
                            pipeline_errors["low_iou_matches"].append({
                                "image": img_path.name,
                                "iou": round(best_iou, 3),
                                "pred_box": p[2:6],
                                "gt_box": cls_gt[best_gt_idx][1:5] if best_gt_idx >= 0 else None
                            })
                    else:
                        stats[cls_name]["fp_reasons"]["background_false_alarm"] += 1
                        if cid == 1:
                            pipeline_errors["false_alarms"].append({
                                "image": img_path.name,
                                "confidence": round(p[1], 3)
                            })

            fn_count = len(cls_gt) - len(matched_gt)
            stats[cls_name]["FN"] += fn_count
            if cid == 1 and fn_count > 0:
                pipeline_errors["missed_images"].append({
                    "image": img_path.name,
                    "missed_count": fn_count
                })

    # Summary Display
    print("\n" + "-"*60)
    print("DETECTION ERROR METRIC SUMMARY (Conf >= 0.25, IoU >= 0.50):")
    print("-"*60)
    for cls_name, d in stats.items():
        tp = d["TP"]
        fp = d["FP"]
        fn = d["FN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_gt_ar = float(np.mean(d["gt_aspect_ratios"])) if d["gt_aspect_ratios"] else 0.0
        mean_pred_ar = float(np.mean(d["pred_aspect_ratios"])) if d["pred_aspect_ratios"] else 0.0

        print(f"Class: {cls_name:<10}")
        print(f"  Ground Truth: {d['gt_count']:>4} | Predictions: {d['pred_count']:>4}")
        print(f"  TP: {tp:>4} | FP: {fp:>4} | FN: {fn:>4}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  Mean Aspect Ratio (W/H) -> GT: {mean_gt_ar:.2f} | Pred: {mean_pred_ar:.2f}")
        print(f"  FP Breakdown: {dict(d['fp_reasons'])}")
        print()

    print("="*60)
    print(f"PIPELINE DEEP-DIVE ERROR INSIGHTS:")
    print(f"  Total GT Pipeline instances in test split : {stats['Pipeline']['gt_count']}")
    print(f"  Missed Pipelines (FN)                      : {stats['Pipeline']['FN']}")
    print(f"  False Alarm Pipelines (FP)                : {stats['Pipeline']['FP']}")
    print(f"  Low IoU Overlap (boundary drift)           : {stats['Pipeline']['fp_reasons'].get('low_iou_overlap', 0)}")
    print(f"  Pure Background False Alarms               : {stats['Pipeline']['fp_reasons'].get('background_false_alarm', 0)}")
    print("="*60 + "\n")

    out_file = Path("reports/evaluation/error_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        # Convert non-serializable objects
        clean_stats = {
            k: {
                "TP": v["TP"], "FP": v["FP"], "FN": v["FN"],
                "gt_count": v["gt_count"], "pred_count": v["pred_count"],
                "mean_gt_aspect_ratio": float(np.mean(v["gt_aspect_ratios"])) if v["gt_aspect_ratios"] else None,
                "mean_pred_aspect_ratio": float(np.mean(v["pred_aspect_ratios"])) if v["pred_aspect_ratios"] else None,
                "fp_reasons": dict(v["fp_reasons"])
            } for k, v in stats.items()
        }
        json.dump({"per_class": clean_stats, "pipeline_errors": pipeline_errors}, f, indent=4)
    print(f"[ErrorAnalysis] Full diagnostic report written to: {out_file}")

    return stats

if __name__ == "__main__":
    run_per_class_error_analysis()
