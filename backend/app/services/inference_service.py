import os
import time
import uuid
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from ultralytics import YOLO

from ml.preprocessing.pipeline import SonarPreprocessingPipeline
from ml.preprocessing.tiling import SonarTiler
from ml.validation.validator import validate_acoustic_signature, compute_priority_rating
from ml.validation.fusion import fuse_and_rank_candidates, UnifiedCandidate
from ml.anomaly.anomaly_engine import AnomalyDiscoveryEngine
from ml.metadata.parser import MetadataIngestionEngine
from backend.app.models.entities import SurveyRecord, ImageRecord, TelemetryRecord, CandidateRecord
from configs.settings import config

STATIC_RESULTS_DIR = Path("backend/static/results")
STATIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class InferenceService:
    def __init__(self):
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.pipeline = SonarPreprocessingPipeline()
        self.tiler = SonarTiler(tile_size=640, overlap_ratio=0.15)
        self.anomaly_engine = AnomalyDiscoveryEngine()
        self.metadata_engine = MetadataIngestionEngine()

        # Initialize detector model
        self.model = None
        self._load_detector()

        # If anomaly engine not yet fitted, fit on SSL sample
        if not self.anomaly_engine.is_fitted:
            ssl_sample_dir = Path("D:/USIP_SSL_SAMPLE")
            if ssl_sample_dir.exists():
                self.anomaly_engine.fit_from_normal_seabed(ssl_sample_dir)

    def _load_detector(self):
        weights_path = Path(config.model_path)
        if weights_path.exists():
            print(f"Loading trained USIP detector from {weights_path}...")
            self.model = YOLO(str(weights_path))
        else:
            # Check experiment checkpoint
            exp_weights = Path("experiments/usip_sonar_detector/weights/best.pt")
            if exp_weights.exists():
                print(f"Loading checkpoint detector from {exp_weights}...")
                self.model = YOLO(str(exp_weights))
            else:
                print("Trained weights not found yet. Using base model until training completes...")
                self.model = YOLO("yolov8n.pt")

    def run_inference(
        self, 
        image_bytes: bytes, 
        filename: str, 
        db: Session,
        survey_name: str = "Tactical SSS Survey"
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Reload model if new trained weights appeared
        if not Path(config.model_path).exists() and Path("experiments/usip_sonar_detector/weights/best.pt").exists():
            self._load_detector()

        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if raw_img is None:
            raise ValueError("Failed to decode uploaded sonar image file.")

        h_orig, w_orig = raw_img.shape[:2]

        # 1. Acoustic Preprocessing Pipeline
        enhanced_gray, qc_report = self.pipeline.process(raw_img)

        # 2. Metadata Ingestion
        metadata = self.metadata_engine.parse(filename)

        # Save images to static results directory for UI display
        session_id = uuid.uuid4().hex[:8]
        raw_out_filename = f"{session_id}_raw_{filename}"
        proc_out_filename = f"{session_id}_processed_{filename}"
        
        raw_save_path = STATIC_RESULTS_DIR / raw_out_filename
        proc_save_path = STATIC_RESULTS_DIR / proc_out_filename
        
        # Save raw and enhanced
        cv2.imwrite(str(raw_save_path), raw_img)
        cv2.imwrite(str(proc_save_path), enhanced_gray)

        # 3. Supervised Target Detection
        supervised_candidates = []
        
        # If image is very wide or long, run tiled inference
        if h_orig > 1000 or w_orig > 1000:
            tiles = self.tiler.generate_tiles(enhanced_gray)
            for t_data in tiles:
                tile_crop = t_data["tile_image"]
                crop_coords = t_data["crop_coords"]
                # Convert to 3-channel for YOLO input
                tile_rgb = cv2.cvtColor(tile_crop, cv2.COLOR_GRAY2RGB)
                
                det_results = self.model(tile_rgb, conf=0.30, device=self.device, verbose=False)
                for r in det_results:
                    for box in r.boxes:
                        bx_l, by_l, bw_l, bh_l = box.xywh[0].tolist()
                        top_left_x = bx_l - bw_l / 2.0
                        top_left_y = by_l - bh_l / 2.0
                        
                        # Map back to global coordinates
                        gx, gy, gw, gh = self.tiler.map_bbox_to_global(
                            (top_left_x, top_left_y, bw_l, bh_l), crop_coords
                        )
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        class_name = self.model.names.get(cls_idx, f"Target-{cls_idx}")

                        # Physics validation
                        acoustic = validate_acoustic_signature(enhanced_gray, int(gx), int(gy), int(gw), int(gh))
                        p_level, p_score = compute_priority_rating(
                            conf, 0.0, acoustic.shadow_contrast_ratio, acoustic.context_similarity, is_anomaly=False
                        )

                        supervised_candidates.append({
                            "class": class_name,
                            "bbox": [gx, gy, gw, gh],
                            "model_confidence": conf,
                            "shadow_evidence": acoustic.shadow_contrast_ratio,
                            "shadow_evidence_level": acoustic.shadow_evidence_level,
                            "context_score": acoustic.context_similarity,
                            "final_priority": p_level,
                            "priority_score": p_score,
                            "latitude": metadata.latitude,
                            "longitude": metadata.longitude,
                            "metadata_available": metadata.metadata_available,
                            "geolocation_note": metadata.status_note,
                            "explanation": acoustic.physics_explanation,
                            "source_image": filename
                        })
        else:
            # Direct inference
            proc_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            det_results = self.model(proc_rgb, conf=0.30, device=self.device, verbose=False)
            for r in det_results:
                for box in r.boxes:
                    bx, by, bw, bh = box.xywh[0].tolist()
                    top_left_x = max(0, bx - bw / 2.0)
                    top_left_y = max(0, by - bh / 2.0)
                    conf = float(box.conf[0])
                    cls_idx = int(box.cls[0])
                    class_name = self.model.names.get(cls_idx, f"Target-{cls_idx}")

                    acoustic = validate_acoustic_signature(enhanced_gray, int(top_left_x), int(top_left_y), int(bw), int(bh))
                    p_level, p_score = compute_priority_rating(
                        conf, 0.0, acoustic.shadow_contrast_ratio, acoustic.context_similarity, is_anomaly=False
                    )

                    supervised_candidates.append({
                        "class": class_name,
                        "bbox": [top_left_x, top_left_y, bw, bh],
                        "model_confidence": conf,
                        "shadow_evidence": acoustic.shadow_contrast_ratio,
                        "shadow_evidence_level": acoustic.shadow_evidence_level,
                        "context_score": acoustic.context_similarity,
                        "final_priority": p_level,
                        "priority_score": p_score,
                        "latitude": metadata.latitude,
                        "longitude": metadata.longitude,
                        "metadata_available": metadata.metadata_available,
                        "geolocation_note": metadata.status_note,
                        "explanation": acoustic.physics_explanation,
                        "source_image": filename
                    })

        # 4. Unsupervised Anomaly Discovery (scan salient regions)
        anomaly_candidates = []
        # Find high backscatter peaks distinct from background
        thresh = cv2.adaptiveThreshold(
            enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, -15
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours[:15]:
            area = cv2.contourArea(cnt)
            if 300 < area < (h_orig * w_orig * 0.4):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                crop = enhanced_gray[by:by+bh, bx:bx+bw]
                if crop.size > 0:
                    anom_res = self.anomaly_engine.score_patch(crop)
                    if anom_res["is_anomaly"] or anom_res["anomaly_score"] > 0.65:
                        acoustic = validate_acoustic_signature(enhanced_gray, bx, by, bw, bh)
                        p_level, p_score = compute_priority_rating(
                            0.0, anom_res["anomaly_score"], acoustic.shadow_contrast_ratio, acoustic.context_similarity, is_anomaly=True
                        )
                        anomaly_candidates.append({
                            "bbox": [bx, by, bw, bh],
                            "anomaly_score": anom_res["anomaly_score"],
                            "shadow_evidence": acoustic.shadow_contrast_ratio,
                            "shadow_evidence_level": acoustic.shadow_evidence_level,
                            "context_score": acoustic.context_similarity,
                            "final_priority": p_level,
                            "priority_score": p_score,
                            "latitude": metadata.latitude,
                            "longitude": metadata.longitude,
                            "metadata_available": metadata.metadata_available,
                            "geolocation_note": metadata.status_note,
                            "explanation": f"{anom_res['explanation']} {acoustic.physics_explanation}",
                            "source_image": filename
                        })

        # 5. Candidate Fusion Layer
        fused_candidates: List[UnifiedCandidate] = fuse_and_rank_candidates(
            supervised_candidates, anomaly_candidates
        )

        # 6. Database Persistence
        survey_rec = SurveyRecord(
            survey_name=survey_name,
            vessel_name="AUV Explorer-1",
            status="ANALYSIS_COMPLETE",
            notes=f"Processed sonar file: {filename}"
        )
        db.add(survey_rec)
        db.flush()

        img_rec = ImageRecord(
            survey_id=survey_rec.id,
            filename=filename,
            file_path=str(raw_save_path),
            processed_path=str(proc_save_path),
            width=w_orig,
            height=h_orig,
            channels=1 if len(raw_img.shape) == 2 else raw_img.shape[2],
            estimated_snr_db=qc_report.get("estimated_snr_db", 20.0),
            dynamic_range=qc_report.get("dynamic_range", 150.0)
        )
        db.add(img_rec)
        db.flush()

        telem_rec = TelemetryRecord(
            image_id=img_rec.id,
            timestamp=metadata.timestamp,
            latitude=metadata.latitude,
            longitude=metadata.longitude,
            x_local_m=metadata.x_local_m,
            y_local_m=metadata.y_local_m,
            depth_m=metadata.depth_m,
            altitude_m=metadata.altitude_m,
            heading_deg=metadata.heading_deg,
            metadata_available=metadata.metadata_available,
            status_note=metadata.status_note
        )
        db.add(telem_rec)

        cand_out_list = []
        for c in fused_candidates:
            c_rec = CandidateRecord(
                image_id=img_rec.id,
                candidate_id=c.candidate_id,
                source=c.source,
                target_class=c.target_class,
                bbox_x=c.bbox[0],
                bbox_y=c.bbox[1],
                bbox_w=c.bbox[2],
                bbox_h=c.bbox[3],
                model_confidence=c.model_confidence,
                anomaly_score=c.anomaly_score,
                shadow_evidence=c.shadow_evidence,
                shadow_evidence_level=c.shadow_evidence_level,
                context_score=c.context_score,
                final_priority=c.final_priority,
                priority_score=c.priority_score,
                latitude=c.latitude,
                longitude=c.longitude,
                metadata_available=c.metadata_available,
                geolocation_note=c.geolocation_note,
                explanation=c.explanation
            )
            db.add(c_rec)
            db.flush()

            cand_out_list.append({
                "id": c_rec.id,
                "candidate_id": c.candidate_id,
                "source": c.source,
                "target_class": c.target_class,
                "bbox": c.bbox,
                "model_confidence": c.model_confidence,
                "anomaly_score": c.anomaly_score,
                "shadow_evidence": c.shadow_evidence,
                "shadow_evidence_level": c.shadow_evidence_level,
                "context_score": c.context_score,
                "final_priority": c.final_priority,
                "priority_score": c.priority_score,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "metadata_available": c.metadata_available,
                "geolocation_note": c.geolocation_note,
                "explanation": c.explanation
            })

        db.commit()

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "survey_id": survey_rec.id,
            "image_id": img_rec.id,
            "filename": filename,
            "image_width": w_orig,
            "image_height": h_orig,
            "raw_image_url": f"/static/results/{raw_out_filename}",
            "processed_image_url": f"/static/results/{proc_out_filename}",
            "quality_metrics": qc_report,
            "telemetry": {
                "latitude": metadata.latitude,
                "longitude": metadata.longitude,
                "depth_m": metadata.depth_m,
                "altitude_m": metadata.altitude_m,
                "metadata_available": metadata.metadata_available,
                "status_note": metadata.status_note
            },
            "total_candidates": len(cand_out_list),
            "high_priority_count": sum(1 for c in cand_out_list if c["final_priority"] in ["CRITICAL", "HIGH"]),
            "known_targets_count": sum(1 for c in cand_out_list if c["source"] == "detector"),
            "unknown_anomalies_count": sum(1 for c in cand_out_list if c["source"] == "anomaly"),
            "candidates": cand_out_list,
            "inference_latency_ms": round(latency_ms, 2),
            "device_used": f"CUDA ({torch.cuda.get_device_name(0)})" if self.device != "cpu" else "CPU"
        }

# Global singleton
inference_service = InferenceService()

