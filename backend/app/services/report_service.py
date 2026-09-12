import io
import csv
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from backend.app.models.entities import SurveyRecord, ImageRecord, CandidateRecord, ReviewRecord

class ReportService:
    @staticmethod
    def generate_json_report(survey_id: str, db: Session) -> Dict[str, Any]:
        survey = db.query(SurveyRecord).filter(SurveyRecord.id == survey_id).first()
        if not survey:
            return {"error": "Survey not found"}

        report = {
            "mission_intelligence_report": {
                "survey_id": survey.id,
                "survey_name": survey.survey_name,
                "vessel_name": survey.vessel_name,
                "status": survey.status,
                "timestamp": str(survey.created_at),
                "images": []
            }
        }

        for img in survey.images:
            img_dict = {
                "image_id": img.id,
                "filename": img.filename,
                "dimensions": f"{img.width}x{img.height}",
                "estimated_snr_db": img.estimated_snr_db,
                "telemetry": {
                    "latitude": img.telemetry.latitude if img.telemetry else None,
                    "longitude": img.telemetry.longitude if img.telemetry else None,
                    "depth_m": img.telemetry.depth_m if img.telemetry else None,
                    "altitude_m": img.telemetry.altitude_m if img.telemetry else None,
                    "metadata_available": img.telemetry.metadata_available if img.telemetry else False,
                    "status_note": img.telemetry.status_note if img.telemetry else "No telemetry."
                },
                "candidates": []
            }

            for c in img.candidates:
                cand_dict = {
                    "candidate_id": c.candidate_id,
                    "source": c.source,
                    "target_class": c.target_class,
                    "bbox": [c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
                    "model_confidence": c.model_confidence,
                    "anomaly_score": c.anomaly_score,
                    "shadow_contrast_ratio": c.shadow_evidence,
                    "shadow_evidence_level": c.shadow_evidence_level,
                    "context_similarity": c.context_score,
                    "priority_level": c.final_priority,
                    "priority_score": c.priority_score,
                    "latitude": c.latitude,
                    "longitude": c.longitude,
                    "explanation": c.explanation,
                    "analyst_reviews": [
                        {"status": r.review_status, "notes": r.notes, "timestamp": str(r.created_at)}
                        for r in c.reviews
                    ]
                }
                img_dict["candidates"].append(cand_dict)

            report["mission_intelligence_report"]["images"].append(img_dict)

        return report

    @staticmethod
    def generate_csv_report(survey_id: str, db: Session) -> str:
        survey = db.query(SurveyRecord).filter(SurveyRecord.id == survey_id).first()
        if not survey:
            return "Error: Survey not found"

        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            "Survey ID", "Survey Name", "Image ID", "Filename",
            "Candidate ID", "Source", "Class",
            "BBox X", "BBox Y", "BBox Width", "BBox Height",
            "Model Confidence", "Anomaly Score", "Shadow Contrast Ratio",
            "Shadow Evidence Level", "Context Score", "Priority Rating", "Priority Score",
            "Latitude", "Longitude", "Metadata Available", "Latest Review Status", "Explanation"
        ]
        writer.writerow(headers)

        for img in survey.images:
            for c in img.candidates:
                latest_review = c.reviews[-1].review_status if c.reviews else "Pending Review"
                row = [
                    survey.id, survey.survey_name, img.id, img.filename,
                    c.candidate_id, c.source, c.target_class,
                    round(c.bbox_x, 1), round(c.bbox_y, 1), round(c.bbox_w, 1), round(c.bbox_h, 1),
                    c.model_confidence if c.model_confidence is not None else "N/A",
                    c.anomaly_score if c.anomaly_score is not None else "N/A",
                    c.shadow_evidence, c.shadow_evidence_level, c.context_score,
                    c.final_priority, c.priority_score,
                    c.latitude if c.latitude is not None else "N/A",
                    c.longitude if c.longitude is not None else "N/A",
                    c.metadata_available,
                    latest_review,
                    c.explanation
                ]
                writer.writerow(row)

        return output.getvalue()

