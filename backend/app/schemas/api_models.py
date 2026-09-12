from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime

class BoundingBoxSchema(BaseModel):
    x: float
    y: float
    width: float
    height: float

class CandidateOutSchema(BaseModel):
    id: str
    candidate_id: str
    source: str
    target_class: str
    bbox: List[float]
    model_confidence: Optional[float]
    anomaly_score: Optional[float]
    shadow_evidence: float
    shadow_evidence_level: str
    context_score: float
    final_priority: str
    priority_score: float
    latitude: Optional[float]
    longitude: Optional[float]
    metadata_available: bool
    geolocation_note: str
    explanation: str

    class Config:
        from_attributes = True

class ReviewCreateSchema(BaseModel):
    candidate_id: str
    analyst_id: str = "ANALYST-01"
    review_status: str = Field(description="'confirmed', 'rejected', or 'uncertain'")
    notes: Optional[str] = None

class FeedbackCreateSchema(BaseModel):
    candidate_id: str
    rating: int = 5
    corrected_class: Optional[str] = None
    feedback_notes: Optional[str] = None

class InferenceResponseSchema(BaseModel):
    survey_id: str
    image_id: str
    filename: str
    image_width: int
    image_height: int
    raw_image_url: str
    processed_image_url: str
    quality_metrics: Dict[str, Any]
    telemetry: Dict[str, Any]
    total_candidates: int
    high_priority_count: int
    known_targets_count: int
    unknown_anomalies_count: int
    candidates: List[CandidateOutSchema]
    inference_latency_ms: float
    device_used: str

class SurveySummarySchema(BaseModel):
    id: str
    survey_name: str
    vessel_name: str
    status: str
    created_at: datetime.datetime
    image_count: int
    candidate_count: int

    class Config:
        from_attributes = True

class HealthResponseSchema(BaseModel):
    status: str
    gpu_available: bool
    gpu_device: Optional[str]
    detector_model_loaded: bool
    anomaly_engine_loaded: bool
    database_connected: bool

