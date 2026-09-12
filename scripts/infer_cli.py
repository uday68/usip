import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from ml.preprocessing.pipeline import SonarPreprocessingPipeline
from ml.validation.validator import validate_acoustic_signature, compute_priority_rating
from ml.validation.fusion import fuse_and_rank_candidates
from ml.anomaly.anomaly_engine import AnomalyDiscoveryEngine
from ml.metadata.parser import MetadataIngestionEngine
from ultralytics import YOLO

def run_cli_inference(image_path: Path, output_dir: Path, conf_thresh: float = 0.25):
    print(f"=== USIP Sonar Intelligence CLI: Analyzing {image_path.name} ===")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read image
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not decode image: {image_path}")
    h, w = raw.shape[:2]

    # 2. Preprocessing
    pipeline = SonarPreprocessingPipeline()
    enhanced, qc = pipeline.process(raw)

    # 3. Metadata Ingestion
    meta_engine = MetadataIngestionEngine()
    metadata = meta_engine.parse(image_path.name)

    # 4. Supervised Target Detection
    weights = Path("ml/detection/weights/best.pt")
    if not weights.exists():
        weights = Path("yolov8n.pt")
    model = YOLO(str(weights))

    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    results = model(enhanced_rgb, conf=conf_thresh, verbose=False)

    supervised_cands = []
    for r in results:
        for box in r.boxes:
            bx, by, bw, bh = box.xywh[0].tolist()
            x0 = max(0, bx - bw / 2.0)
            y0 = max(0, by - bh / 2.0)
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            cls_name = model.names.get(cls_idx, f"Class-{cls_idx}")

            acoustic = validate_acoustic_signature(enhanced, int(x0), int(y0), int(bw), int(bh))
            priority, p_score = compute_priority_rating(
                conf, 0.0, acoustic.shadow_contrast_ratio, acoustic.context_similarity, is_anomaly=False
            )

            supervised_cands.append({
                "class": cls_name,
                "bbox": [x0, y0, bw, bh],
                "model_confidence": conf,
                "shadow_evidence": acoustic.shadow_contrast_ratio,
                "shadow_evidence_level": acoustic.shadow_evidence_level,
                "context_score": acoustic.context_similarity,
                "final_priority": priority,
                "priority_score": p_score,
                "latitude": metadata.latitude,
                "longitude": metadata.longitude,
                "metadata_available": metadata.metadata_available,
                "geolocation_note": metadata.status_note,
                "explanation": acoustic.physics_explanation,
                "source_image": image_path.name
            })

    # 5. Anomaly Discovery
    anomaly_engine = AnomalyDiscoveryEngine()
    anomaly_cands = []
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, -15)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        if 300 < area < (h * w * 0.4):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            crop = enhanced[by:by+bh, bx:bx+bw]
            if crop.size > 0:
                anom_res = anomaly_engine.score_patch(crop)
                if anom_res["is_anomaly"] or anom_res["anomaly_score"] > 0.65:
                    acoustic = validate_acoustic_signature(enhanced, bx, by, bw, bh)
                    priority, p_score = compute_priority_rating(
                        0.0, anom_res["anomaly_score"], acoustic.shadow_contrast_ratio, acoustic.context_similarity, is_anomaly=True
                    )
                    anomaly_cands.append({
                        "bbox": [bx, by, bw, bh],
                        "anomaly_score": anom_res["anomaly_score"],
                        "shadow_evidence": acoustic.shadow_contrast_ratio,
                        "shadow_evidence_level": acoustic.shadow_evidence_level,
                        "context_score": acoustic.context_similarity,
                        "final_priority": priority,
                        "priority_score": p_score,
                        "latitude": metadata.latitude,
                        "longitude": metadata.longitude,
                        "metadata_available": metadata.metadata_available,
                        "geolocation_note": metadata.status_note,
                        "explanation": f"{anom_res['explanation']} {acoustic.physics_explanation}",
                        "source_image": image_path.name
                    })

    # 6. Fusion & Ranking
    fused = fuse_and_rank_candidates(supervised_cands, anomaly_cands)

    # 7. Render Visualization Overlay
    vis = enhanced_rgb.copy()
    for c in fused:
        bx, by, bw, bh = [int(v) for v in c.bbox]
        color = (16, 185, 129) if c.source == "detector" else (11, 158, 245)
        # Bbox
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
        # Label
        tag = f"{c.candidate_id}: {c.target_class} [{c.final_priority}]"
        cv2.putText(vis, tag, (bx, max(15, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save output visualization
    out_vis_path = output_dir / f"pred_{image_path.stem}.png"
    cv2.imwrite(str(out_vis_path), vis)

    # Save JSON report
    out_json_path = output_dir / f"candidates_{image_path.stem}.json"
    report = {
        "source_image": image_path.name,
        "image_dimensions": [w, h],
        "quality_metrics": qc,
        "metadata": {
            "latitude": metadata.latitude,
            "longitude": metadata.longitude,
            "available": metadata.metadata_available,
            "status_note": metadata.status_note
        },
        "total_candidates": len(fused),
        "candidates": [c.model_dump() for c in fused]
    }
    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Results saved:")
    print(f"  Visualization: {out_vis_path}")
    print(f"  Candidates JSON: {out_json_path}")
    print(f"  Found {len(fused)} ranked candidates ({sum(1 for c in fused if c.final_priority in ['CRITICAL', 'HIGH'])} High/Critical)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="USIP Standalone Sonar Inference CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input SSS image")
    parser.add_argument("--output", type=str, default="reports/sample_predictions", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    run_cli_inference(Path(args.input), Path(args.output), args.conf)

