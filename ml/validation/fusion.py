from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

class UnifiedBoundingBox(BaseModel):
    x: float = Field(description="Top-left X pixel coordinate")
    y: float = Field(description="Top-left Y pixel coordinate")
    width: float = Field(description="Width in pixels")
    height: float = Field(description="Height in pixels")

class UnifiedCandidate(BaseModel):
    candidate_id: str = Field(description="Unique candidate identifier, e.g. CAND-01")
    source: str = Field(description="'detector' (supervised) or 'anomaly' (unsupervised)")
    target_class: str = Field(description="Class name, e.g. 'Shipwreck', 'Pipeline', 'Debris', or '[EXPERIMENTAL] Acoustic Anomaly'")
    bbox: List[float] = Field(description="[x, y, width, height] in pixel coordinates")
    model_confidence: Optional[float] = Field(None, description="Supervised detector confidence score [0.0, 1.0]")
    anomaly_score: Optional[float] = Field(None, description="Unsupervised acoustic novelty distance score [0.0, 1.0]")
    shadow_evidence: float = Field(description="Acoustic cast shadow contrast ratio [1.0, inf)")
    shadow_evidence_level: str = Field(description="'Strong', 'Moderate', 'Weak', or 'None'")
    context_score: float = Field(description="Seafloor context similarity score [0.0, 1.0]")
    final_priority: str = Field(description="'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'")
    priority_score: float = Field(description="Composite multi-factor priority score [0.0, 1.0]")
    latitude: Optional[float] = Field(None, description="WGS84 Latitude or null if unavailable")
    longitude: Optional[float] = Field(None, description="WGS84 Longitude or null if unavailable")
    metadata_available: bool = Field(False, description="True if valid navigation metadata was associated")
    geolocation_note: str = Field("Geolocation unavailable — no valid navigation metadata associated with candidate.", description="Explanation of spatial coordinate status")
    explanation: str = Field(description="Sonar physics and model explainability narrative")
    source_image: str = Field(description="Source image filename or survey ID")

def fuse_and_rank_candidates(
    supervised_candidates: List[Dict[str, Any]], 
    anomaly_candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.35
) -> List[UnifiedCandidate]:
    """
    Candidate Fusion Layer:
    Merges detections from the Supervised Detector and the Unsupervised Anomaly Engine.
    When an anomaly coincides with a known detector target, the evidence is fused into
    a single high-priority candidate. Unmatched anomalies are preserved with source='anomaly'.
    Ranks candidates by priority_score descending.
    """
    fused: List[UnifiedCandidate] = []
    matched_anomaly_indices = set()

    cand_counter = 0

    for det in supervised_candidates:
        cand_counter += 1
        d_box = det["bbox"]
        d_x, d_y, d_w, d_h = d_box[0], d_box[1], d_box[2], d_box[3]

        # Check overlap with anomaly candidates
        best_anomaly_score = det.get("anomaly_score", 0.12)
        anomaly_note = ""

        for a_idx, anom in enumerate(anomaly_candidates):
            a_box = anom["bbox"]
            a_x, a_y, a_w, a_h = a_box[0], a_box[1], a_box[2], a_box[3]

            # Intersection over Union (IoU)
            xi1 = max(d_x, a_x)
            yi1 = max(d_y, a_y)
            xi2 = min(d_x + d_w, a_x + a_w)
            yi2 = min(d_y + d_h, a_y + a_h)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = d_w * d_h
            box2_area = a_w * a_h
            union_area = box1_area + box2_area - inter_area
            iou = inter_area / max(1.0, union_area)

            if iou > iou_threshold:
                matched_anomaly_indices.add(a_idx)
                best_anomaly_score = max(best_anomaly_score, anom.get("anomaly_score", 0.0))
                anomaly_note = f" (Validated by Acoustic Anomaly Engine: score {best_anomaly_score:.2f})"

        cand = UnifiedCandidate(
            candidate_id=f"CAND-{cand_counter:02d}",
            source="detector",
            target_class=det["class"],
            bbox=[float(d_x), float(d_y), float(d_w), float(d_h)],
            model_confidence=round(det["model_confidence"], 4),
            anomaly_score=round(best_anomaly_score, 4),
            shadow_evidence=round(det.get("shadow_evidence", 1.2), 2),
            shadow_evidence_level=det.get("shadow_evidence_level", "Moderate"),
            context_score=round(det.get("context_score", 0.5), 3),
            final_priority=det["final_priority"],
            priority_score=round(det["priority_score"], 4),
            latitude=det.get("latitude"),
            longitude=det.get("longitude"),
            metadata_available=det.get("metadata_available", False),
            geolocation_note=det.get("geolocation_note", "Geolocation unavailable — no valid navigation metadata associated with candidate."),
            explanation=det.get("explanation", "") + anomaly_note,
            source_image=det.get("source_image", "unknown")
        )
        fused.append(cand)

    # Append distinct, non-overlapping anomaly discoveries
    for a_idx, anom in enumerate(anomaly_candidates):
        if a_idx in matched_anomaly_indices:
            continue
        cand_counter += 1
        a_box = anom["bbox"]
        cand = UnifiedCandidate(
            candidate_id=f"ANOM-{cand_counter:02d}",
            source="anomaly",
            target_class="[EXPERIMENTAL] Acoustic Anomaly",
            bbox=[float(a_box[0]), float(a_box[1]), float(a_box[2]), float(a_box[3])],
            model_confidence=None,
            anomaly_score=round(anom["anomaly_score"], 4),
            shadow_evidence=round(anom.get("shadow_evidence", 1.1), 2),
            shadow_evidence_level=anom.get("shadow_evidence_level", "Weak"),
            context_score=round(anom.get("context_score", 0.6), 3),
            final_priority=anom.get("final_priority", "MEDIUM"),
            priority_score=round(anom.get("priority_score", 0.55), 4),
            latitude=anom.get("latitude"),
            longitude=anom.get("longitude"),
            metadata_available=anom.get("metadata_available", False),
            geolocation_note=anom.get("geolocation_note", "Geolocation unavailable — no valid navigation metadata associated with candidate."),
            explanation=anom.get("explanation", "Experimental anomaly candidate requiring expert review."),
            source_image=anom.get("source_image", "unknown")
        )
        fused.append(cand)

    # Sort descending by priority_score
    fused.sort(key=lambda c: c.priority_score, reverse=True)
    return fused

