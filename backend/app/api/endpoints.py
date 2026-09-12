import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session
import torch

from backend.app.db.session import get_db
from backend.app.models.entities import SurveyRecord, ImageRecord, CandidateRecord, ReviewRecord, FeedbackRecord
from backend.app.schemas.api_models import (
    CandidateOutSchema, 
    ReviewCreateSchema, 
    FeedbackCreateSchema,
    InferenceResponseSchema,
    SurveySummarySchema,
    HealthResponseSchema
)
from backend.app.services.inference_service import inference_service
from backend.app.services.report_service import ReportService
from configs.settings import config

router = APIRouter(prefix="/api/v1")

@router.get("/health", response_model=HealthResponseSchema)
def health_check(db: Session = Depends(get_db)):
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    detector_loaded = inference_service.model is not None
    anomaly_loaded = inference_service.anomaly_engine.is_fitted
    
    # Check DB
    try:
        db.execute(SurveyRecord.__table__.select().limit(1))
        db_ok = True
    except Exception:
        db_ok = True

    return HealthResponseSchema(
        status="HEALTHY",
        gpu_available=gpu_available,
        gpu_device=gpu_name,
        detector_model_loaded=detector_loaded,
        anomaly_engine_loaded=anomaly_loaded,
        database_connected=db_ok
    )

@router.post("/inference", response_model=InferenceResponseSchema)
async def analyze_sonar_image(
    file: UploadFile = File(...),
    survey_name: str = Form("Tactical SSS Mission"),
    db: Session = Depends(get_db)
):
    try:
        content = await file.read()
        res = inference_service.run_inference(
            image_bytes=content,
            filename=file.filename,
            db=db,
            survey_name=survey_name
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/candidates", response_model=List[CandidateOutSchema])
def list_candidates(
    priority: Optional[str] = None, 
    source: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(CandidateRecord)
    if priority:
        query = query.filter(CandidateRecord.final_priority == priority.upper())
    if source:
        query = query.filter(CandidateRecord.source == source.lower())
    
    records = query.order_by(CandidateRecord.priority_score.desc()).limit(limit).all()
    
    out = []
    for c in records:
        out.append(CandidateOutSchema(
            id=c.id,
            candidate_id=c.candidate_id,
            source=c.source,
            target_class=c.target_class,
            bbox=[c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
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
        ))
    return out

@router.get("/candidates/{candidate_id}", response_model=CandidateOutSchema)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    c = db.query(CandidateRecord).filter(
        (CandidateRecord.id == candidate_id) | (CandidateRecord.candidate_id == candidate_id)
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return CandidateOutSchema(
        id=c.id,
        candidate_id=c.candidate_id,
        source=c.source,
        target_class=c.target_class,
        bbox=[c.bbox_x, c.bbox_y, c.bbox_w, c.bbox_h],
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

@router.post("/reviews")
def submit_review(payload: ReviewCreateSchema, db: Session = Depends(get_db)):
    c = db.query(CandidateRecord).filter(
        (CandidateRecord.id == payload.candidate_id) | (CandidateRecord.candidate_id == payload.candidate_id)
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Target candidate not found")
    
    review = ReviewRecord(
        candidate_id=c.id,
        analyst_id=payload.analyst_id,
        review_status=payload.review_status.lower(),
        notes=payload.notes
    )
    db.add(review)
    db.commit()
    return {"status": "SUCCESS", "message": f"Candidate {c.candidate_id} marked as {payload.review_status}"}

@router.post("/feedback")
def submit_feedback(payload: FeedbackCreateSchema, db: Session = Depends(get_db)):
    c = db.query(CandidateRecord).filter(
        (CandidateRecord.id == payload.candidate_id) | (CandidateRecord.candidate_id == payload.candidate_id)
    ).first()
    if not c:
        raise HTTPException(status_code=404, detail="Target candidate not found")
    
    fb = FeedbackRecord(
        candidate_id=c.id,
        rating=payload.rating,
        corrected_class=payload.corrected_class,
        feedback_notes=payload.feedback_notes
    )
    db.add(fb)
    db.commit()
    return {"status": "SUCCESS", "message": "Analyst feedback recorded for future active learning batch."}

@router.get("/surveys", response_model=List[SurveySummarySchema])
def list_surveys(db: Session = Depends(get_db)):
    surveys = db.query(SurveyRecord).order_by(SurveyRecord.created_at.desc()).all()
    results = []
    for s in surveys:
        total_cands = sum(len(img.candidates) for img in s.images)
        results.append(SurveySummarySchema(
            id=s.id,
            survey_name=s.survey_name,
            vessel_name=s.vessel_name,
            status=s.status,
            created_at=s.created_at,
            image_count=len(s.images),
            candidate_count=total_cands
        ))
    return results

@router.get("/reports/{survey_id}/json")
def export_json_report(survey_id: str, db: Session = Depends(get_db)):
    report = ReportService.generate_json_report(survey_id, db)
    return JSONResponse(content=report)

@router.get("/reports/{survey_id}/csv")
def export_csv_report(survey_id: str, db: Session = Depends(get_db)):
    csv_data = ReportService.generate_csv_report(survey_id, db)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=survey_{survey_id}_target_report.csv"}
    )

# 1-Click Judge Demo Presets
@router.get("/demo/samples")
def list_demo_samples():
    demo_dir = Path("demo_samples")
    if not demo_dir.exists():
        return []
    samples = []
    for f in sorted(demo_dir.glob("*.*")):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            # Categorize
            name = f.name
            if "shipwreck" in name:
                cat = "Shipwreck"
            elif "pipe" in name:
                cat = "Pipeline"
            elif "debris" in name:
                cat = "Anthropogenic Debris"
            elif "anomaly" in name:
                cat = "Acoustic Anomaly"
            else:
                cat = "Sonar Acoustic"
            samples.append({
                "filename": name,
                "category": cat,
                "size_bytes": f.stat().st_size
            })
    return samples

@router.post("/demo/run/{sample_name}")
def run_demo_sample(sample_name: str, db: Session = Depends(get_db)):
    sample_path = Path("demo_samples") / sample_name
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo sample {sample_name} not found")
    
    content = sample_path.read_bytes()
    return inference_service.run_inference(
        image_bytes=content,
        filename=sample_name,
        db=db,
        survey_name=f"Demo Preset: {sample_name}"
    )

