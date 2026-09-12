from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x: float = Field(..., description="Normalized or pixel X coordinate (top-left)")
    y: float = Field(..., description="Normalized or pixel Y coordinate (top-left)")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")

class AcousticMetrics(BaseModel):
    highlight_intensity: float = Field(..., description="Average highlight intensity (0-255)")
    shadow_intensity: float = Field(..., description="Average shadow intensity (0-255)")
    shadow_evidence_ratio: float = Field(..., description="Highlight-to-shadow contrast ratio")
    shadow_evidence_level: str = Field(..., description="'Strong', 'Moderate', 'Weak', or 'None'")
    seafloor_contrast: float = Field(..., description="Contrast against surrounding seabed")
    seafloor_similarity: str = Field(..., description="'Low similarity', 'Moderate', 'High similarity'")

class DetectionResult(BaseModel):
    id: str
    category: str = Field(..., description="'Known Target' or 'Anomaly Candidate'")
    class_name: str = Field(..., description="e.g., 'Shipwreck', 'Pipe / Cylinder', 'Debris / Object', 'Acoustic Anomaly'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox
    anomaly_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    nearest_cluster_distance: Optional[float] = Field(None)
    priority: str = Field(..., description="'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'")
    acoustic_metrics: AcousticMetrics
    explanation: str

class SonarMetadata(BaseModel):
    swath_width_m: float = 100.0
    frequency_khz: float = 450.0
    altitude_m: float = 12.5
    latitude: float = 18.9220
    longitude: float = 72.8347
    towfish_heading_deg: float = 142.0

class AnalysisResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    processed_image_url: str
    raw_image_url: str
    detections: List[DetectionResult]
    metadata: SonarMetadata
    summary: Dict[str, Any]

class DemoSampleInfo(BaseModel):
    id: str
    title: str
    category: str
    description: str
    filename: str
    thumbnail_url: str
