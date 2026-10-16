"""
USIP Comprehensive Per-Class Error Analysis & Diagnostic Engine
============================================================
Performs rigorous object-detection error attribution adhering to VOC/COCO evaluation standards:
1. Confidence-sorted greedy bipartite matching across ALL classes.
2. Multi-class Confusion Matrix across classes + Background.
3. Fine-grained Error Categorization:
   - True Positive (TP): IoU >= threshold, class match
   - Localization Error (Loc): 0.10 <= IoU < threshold, class match
   - Cross-Class Confusion (Cls): IoU >= 0.10, class mismatch
   - Duplicate Detection (Dup): IoU >= threshold on already-matched GT
   - Background False Alarm (Bkg): IoU < 0.10 against all GT
   - False Negative (FN / Miss): GT instance not matched by any prediction
4. IoU Distribution Profiling & Sensitivity Analysis across IoU thresholds.
5. In-depth geometric aspect-ratio and size breakdown.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import config
from ml.utils import set_deterministic_seeds

CLASS_NAMES = {0: "Shipwreck", 1: "Pipeline", 2: "Debris"}

def compute_box_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection-over-Union between two boxes in [x1, y1, x2, y2] format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / union

def run_per_class_error_analysis(
    weights_path: Path = config.model_path,
    dataset_root: Path = config.dataset_root,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("=== USIP RIGOROUS PER-CLASS ERROR ANALYSIS & CONFUSION ENGINE ===")
    print("=" * 70)
    set_deterministic_seeds(config.seed)

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[ErrorAnalysis] Model: {weights_path} on {device}")
    model = YOLO(str(weights_path))

    test_img_dir = dataset_root / "images" / "test"
    test_lbl_dir = dataset_root / "labels" / "test"

    test_imgs = sorted([p for p in test_img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    total_imgs = len(test_imgs)
    print(f"[ErrorAnalysis] Analyzing {total_imgs} test images (conf >= {conf_threshold}, IoU >= {iou_threshold})...")

    # Metrics containers
    classes = list(CLASS_NAMES.values())
    all_classes_with_bkg = classes + ["Background"]
    
    # 4x4 Confusion Matrix: [GT_class][Pred_class]
    confusion_matrix = {
        gt_cls: {pred_cls: 0 for pred_cls in all_classes_with_bkg}
        for gt_cls in all_classes_with_bkg
    }

    stats = {
        cls_name: {
            "TP": 0, "FP": 0, "FN": 0,
            "gt_count": 0, "pred_count": 0,
            "loc_errors": 0,          # 0.1 <= IoU < threshold, same class
            "cls_confusion": 0,       # IoU >= 0.1, wrong class
            "background_fp": 0,       # IoU < 0.1
            "duplicate_fp": 0,        # duplicate detection on same GT
            "gt_aspect_ratios": [],
            "pred_aspect_ratios": [],
            "gt_areas": [],
            "pred_areas": [],
            "all_matched_ious": []
        } for cls_name in classes
    }

    # Pipeline threshold sensitivity curve (IoU: 0.10 -> 0.60)
    pipeline_sensitivity = {thresh: {"TP": 0, "FP": 0, "FN": 0} for thresh in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]}
    
    # Pipeline specific sample tracker
    pipeline_errors = {
        "missed_targets": [],
        "low_iou_detections": [],
        "cross_class_confusions": [],
        "pure_background_alarms": []
    }

    for idx, img_path in enumerate(test_imgs):
        # 1. Parse Ground Truth
        gt_boxes = []  # list of (gt_id, cls_id, [x1, y1, x2, y2], w, h)
        lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            content = lbl_path.read_text(encoding="utf-8").strip()
            if content:
                for g_i, line in enumerate(content.splitlines()):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = max(0.0, xc - w / 2.0)
                        y1 = max(0.0, yc - h / 2.0)
                        x2 = min(1.0, xc + w / 2.0)
                        y2 = min(1.0, yc + h / 2.0)
                        gt_boxes.append((g_i, cid, (x1, y1, x2, y2), w, h))
                        
                        cls_name = CLASS_NAMES[cid]
                        stats[cls_name]["gt_count"] += 1
                        stats[cls_name]["gt_aspect_ratios"].append(w / (h + 1e-6))
                        stats[cls_name]["gt_areas"].append(w * h)

        # 2. Run Inference
        results = model.predict(str(img_path), conf=conf_threshold, device=device, verbose=False)
        pred_boxes = []  # list of (conf, cls_id, [x1, y1, x2, y2], w, h)
        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                c = float(box.conf[0])
                bx, by, bw, bh = box.xywhn[0].tolist()
                x1 = max(0.0, bx - bw / 2.0)
                y1 = max(0.0, by - bh / 2.0)
                x2 = min(1.0, bx + bw / 2.0)
                y2 = min(1.0, by + bh / 2.0)
                pred_boxes.append((c, cid, (x1, y1, x2, y2), bw, bh))
                if cid in CLASS_NAMES:
                    cls_name = CLASS_NAMES[cid]
                    stats[cls_name]["pred_count"] += 1
                    stats[cls_name]["pred_aspect_ratios"].append(bw / (bh + 1e-6))
                    stats[cls_name]["pred_areas"].append(bw * bh)

        # CRITICAL FIX 1: Sort predictions by confidence descending (VOC/COCO standard)
        pred_boxes.sort(key=lambda x: x[0], reverse=True)

        # 3. Greedy Bipartite Matching across ALL classes
        matched_gt_ids = set()

        for (conf, pred_cid, pred_coords, bw, bh) in pred_boxes:
            pred_cls_name = CLASS_NAMES.get(pred_cid, f"Class-{pred_cid}")
            best_iou = 0.0
            best_gt = None

            # Find GT with maximum IoU across ALL ground truth boxes
            for gt in gt_boxes:
                iou = compute_box_iou(pred_coords, gt[2])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None and best_iou >= iou_threshold:
                gt_id, gt_cid, gt_coords, gw, gh = best_gt
                gt_cls_name = CLASS_NAMES[gt_cid]

                if pred_cid == gt_cid:
                    if gt_id not in matched_gt_ids:
                        # True Positive
                        stats[pred_cls_name]["TP"] += 1
                        stats[pred_cls_name]["all_matched_ious"].append(best_iou)
                        confusion_matrix[gt_cls_name][pred_cls_name] += 1
                        matched_gt_ids.add(gt_id)
                    else:
                        # Duplicate False Positive
                        stats[pred_cls_name]["FP"] += 1
                        stats[pred_cls_name]["duplicate_fp"] += 1
                else:
                    # Cross-Class Confusion
                    stats[pred_cls_name]["FP"] += 1
                    stats[pred_cls_name]["cls_confusion"] += 1
                    confusion_matrix[gt_cls_name][pred_cls_name] += 1
                    if pred_cid == 1 or gt_cid == 1:
                        pipeline_errors["cross_class_confusions"].append({
                            "image": img_path.name,
                            "predicted": pred_cls_name,
                            "ground_truth": gt_cls_name,
                            "iou": round(best_iou, 3),
                            "conf": round(conf, 3)
                        })

            elif best_gt is not None and best_iou >= 0.10:
                # Low IoU overlap
                gt_id, gt_cid, gt_coords, gw, gh = best_gt
                gt_cls_name = CLASS_NAMES[gt_cid]
                stats[pred_cls_name]["FP"] += 1

                if pred_cid == gt_cid:
                    # Localization error
                    stats[pred_cls_name]["loc_errors"] += 1
                    stats[pred_cls_name]["all_matched_ious"].append(best_iou)
                    if pred_cid == 1:
                        pipeline_errors["low_iou_detections"].append({
                            "image": img_path.name,
                            "iou": round(best_iou, 3),
                            "conf": round(conf, 3),
                            "pred_box": list(pred_coords),
                            "gt_box": list(gt_coords)
                        })
                else:
                    # Cross-class confusion with moderate overlap
                    stats[pred_cls_name]["cls_confusion"] += 1
                    confusion_matrix[gt_cls_name][pred_cls_name] += 1
            else:
                # Pure background false alarm
                stats[pred_cls_name]["FP"] += 1
                stats[pred_cls_name]["background_fp"] += 1
                confusion_matrix["Background"][pred_cls_name] += 1
                if pred_cid == 1:
                    pipeline_errors["pure_background_alarms"].append({
                        "image": img_path.name,
                        "conf": round(conf, 3)
                    })

        # 4. Unmatched Ground Truth -> False Negatives
        for gt in gt_boxes:
            gt_id, gt_cid, gt_coords, gw, gh = gt
            gt_cls_name = CLASS_NAMES[gt_cid]
            if gt_id not in matched_gt_ids:
                stats[gt_cls_name]["FN"] += 1
                confusion_matrix[gt_cls_name]["Background"] += 1
                if gt_cid == 1:
                    pipeline_errors["missed_targets"].append({
                        "image": img_path.name,
                        "gt_box": list(gt_coords)
                    })

        # 5. Pipeline Sensitivity Evaluation across variable IoU thresholds
        gt_pipe = [g for g in gt_boxes if g[1] == 1]
        pred_pipe = [p for p in pred_boxes if p[1] == 1]
        for t_val in pipeline_sensitivity.keys():
            t_matched = set()
            for p in pred_pipe:
                p_iou = max([compute_box_iou(p[2], g[2]) for g in gt_pipe], default=0.0)
                if p_iou >= t_val:
                    pipeline_sensitivity[t_val]["TP"] += 1
                else:
                    pipeline_sensitivity[t_val]["FP"] += 1
            pipeline_sensitivity[t_val]["FN"] += max(0, len(gt_pipe) - pipeline_sensitivity[t_val]["TP"])

    # -------------------------------------------------------------
    # Report Presentation & Formatting
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DETECTION PERFORMANCE & ERROR BREAKDOWN (Conf >= 0.25, IoU >= 0.50)")
    print("-" * 70)
    for cls_name in classes:
        d = stats[cls_name]
        tp, fp, fn = d["TP"], d["FP"], d["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        ious = d["all_matched_ious"]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        median_iou = float(np.median(ious)) if ious else 0.0

        print(f"Class: {cls_name:<10}")
        print(f"  GT Instances : {d['gt_count']:>4} | Model Predictions: {d['pred_count']:>4}")
        print(f"  TP: {tp:>4} | FP: {fp:>4} | FN: {fn:>4} -> Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"  Mean Overlap IoU on detections: {mean_iou:.3f} (Median: {median_iou:.3f})")
        print(f"  FP Categorization:")
        print(f"    - Pure Background False Alarms  : {d['background_fp']:>3} ({d['background_fp']/(fp+1e-5)*100:.1f}%)")
        print(f"    - Localization Offset (Low IoU) : {d['loc_errors']:>3} ({d['loc_errors']/(fp+1e-5)*100:.1f}%)")
        print(f"    - Cross-Class Confusion         : {d['cls_confusion']:>3} ({d['cls_confusion']/(fp+1e-5)*100:.1f}%)")
        print(f"    - Duplicate Pred on Same Target : {d['duplicate_fp']:>3}")
        print(f"  Aspect Ratio (W/H) -> GT Mean: {np.mean(d['gt_aspect_ratios']):.2f} | Pred Mean: {np.mean(d['pred_aspect_ratios']):.2f}")
        print()

    # Confusion Matrix Print
    print("=" * 70)
    print("MULTI-CLASS CONFUSION MATRIX (Rows: Ground Truth, Columns: Predictions):")
    print("=" * 70)
    header = f"{'GT \\ Pred':<14} | " + " | ".join(f"{c:<10}" for c in all_classes_with_bkg)
    print(header)
    print("-" * len(header))
    for gt_c in all_classes_with_bkg:
        row_vals = " | ".join(f"{confusion_matrix[gt_c][pred_c]:>10}" for pred_c in all_classes_with_bkg)
        print(f"{gt_c:<14} | {row_vals}")
    print("=" * 70)

    # Pipeline Sensitivity Curve Print
    print("\n" + "=" * 70)
    print("PIPELINE OPERATIONAL SENSITIVITY PROFILE (Varying IoU Threshold):")
    print("=" * 70)
    print(f"{'IoU Threshold':<14} | {'TP':>6} | {'FP':>6} | {'FN':>6} | {'Precision':>10} | {'Recall':>10}")
    print("-" * 62)
    for t_val, res in sorted(pipeline_sensitivity.items()):
        p_tp, p_fp, p_fn = res["TP"], res["FP"], res["FN"]
        p_prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) > 0 else 0.0
        p_rec = p_tp / (p_tp + p_fn) if (p_tp + p_fn) > 0 else 0.0
        print(f"{t_val:<14.2f} | {p_tp:>6} | {p_fp:>6} | {p_fn:>6} | {p_prec:>9.4f}  | {p_rec:>9.4f}")
    print("=" * 70)
    print(f"Total Ground-Truth Pipeline instances in test split : {stats['Pipeline']['gt_count']}")
    print(f"Low IoU Boundary Drift Detections                  : {stats['Pipeline']['loc_errors']}")
    print(f"Pure Background False Alarms (False Targets)        : {stats['Pipeline']['background_fp']}")
    print("=" * 70 + "\n")

    # Serialize detailed diagnostic report
    out_file = Path("reports/evaluation/error_analysis.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "per_class": {
            cls_name: {
                "TP": stats[cls_name]["TP"],
                "FP": stats[cls_name]["FP"],
                "FN": stats[cls_name]["FN"],
                "precision": round(stats[cls_name]["TP"] / max(stats[cls_name]["TP"] + stats[cls_name]["FP"], 1), 4),
                "recall": round(stats[cls_name]["TP"] / max(stats[cls_name]["TP"] + stats[cls_name]["FN"], 1), 4),
                "gt_count": stats[cls_name]["gt_count"],
                "pred_count": stats[cls_name]["pred_count"],
                "mean_matched_iou": round(float(np.mean(stats[cls_name]["all_matched_ious"])), 4) if stats[cls_name]["all_matched_ious"] else None,
                "median_matched_iou": round(float(np.median(stats[cls_name]["all_matched_ious"])), 4) if stats[cls_name]["all_matched_ious"] else None,
                "error_breakdown": {
                    "background_fp": stats[cls_name]["background_fp"],
                    "localization_offset_fp": stats[cls_name]["loc_errors"],
                    "cross_class_confusion_fp": stats[cls_name]["cls_confusion"],
                    "duplicate_fp": stats[cls_name]["duplicate_fp"]
                },
                "mean_gt_aspect_ratio": round(float(np.mean(stats[cls_name]["gt_aspect_ratios"])), 3) if stats[cls_name]["gt_aspect_ratios"] else None,
                "mean_pred_aspect_ratio": round(float(np.mean(stats[cls_name]["pred_aspect_ratios"])), 3) if stats[cls_name]["pred_aspect_ratios"] else None,
            } for cls_name in classes
        },
        "confusion_matrix": confusion_matrix,
        "pipeline_sensitivity": {
            str(k): {
                "TP": v["TP"], "FP": v["FP"], "FN": v["FN"],
                "precision": round(v["TP"] / max(v["TP"] + v["FP"], 1), 4),
                "recall": round(v["TP"] / max(v["TP"] + v["FN"], 1), 4)
            } for k, v in pipeline_sensitivity.items()
        },
        "pipeline_errors": pipeline_errors
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"[ErrorAnalysis] Rigorous diagnostic report written to: {out_file}")

    return report_data

if __name__ == "__main__":
    run_per_class_error_analysis()
