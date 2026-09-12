import cv2
import numpy as np
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

class AcousticValidationMetrics(BaseModel):
    highlight_mean: float = Field(description="Mean intensity of specular highlight region [0, 255]")
    shadow_mean: float = Field(description="Mean intensity of acoustic cast shadow [0, 255]")
    shadow_contrast_ratio: float = Field(description="Ratio of ambient seabed intensity to shadow intensity")
    shadow_evidence_level: str = Field(description="'Strong', 'Moderate', 'Weak', or 'None'")
    context_similarity: float = Field(description="Correlation of local background to surrounding seabed [0.0, 1.0]")
    estimated_aspect_ratio: float = Field(description="Bounding box width to height ratio")
    physics_explanation: str = Field(description="Physical explanation of sonar backscatter signature")

def validate_acoustic_signature(
    sonar_gray: np.ndarray, 
    bx: int, 
    by: int, 
    bw: int, 
    bh: int
) -> AcousticValidationMetrics:
    """
    Validates side-scan sonar physical evidence:
    1. Evaluates specular highlight backscatter within the candidate bounding box.
    2. Searches for down-range acoustic shadow (zone of acoustic occlusion behind elevated targets).
    3. Calculates highlight-to-shadow contrast ratio.
    4. Evaluates local seafloor context similarity to prevent false alarms from sediment ripples.
    """
    h_img, w_img = sonar_gray.shape[:2]
    
    # Clip coordinates
    x0 = max(0, min(w_img - 1, int(bx)))
    y0 = max(0, min(h_img - 1, int(by)))
    x1 = max(0, min(w_img, int(bx + bw)))
    y1 = max(0, min(h_img, int(by + bh)))

    if x1 <= x0 or y1 <= y0:
        return AcousticValidationMetrics(
            highlight_mean=128.0,
            shadow_mean=128.0,
            shadow_contrast_ratio=1.0,
            shadow_evidence_level="None",
            context_similarity=0.5,
            estimated_aspect_ratio=1.0,
            physics_explanation="Candidate box outside valid image bounds."
        )

    target_crop = sonar_gray[y0:y1, x0:x1]
    highlight_mean = float(np.mean(target_crop))

    # Down-range shadow search zone (typically extending across-track away from nadir/flight path)
    # Search immediately adjacent along X-axis (across swath)
    shadow_x0 = x1
    shadow_x1 = min(w_img, x1 + int(bw * 1.5))
    
    if shadow_x1 > shadow_x0 and (y1 - y0) > 0:
        shadow_crop = sonar_gray[y0:y1, shadow_x0:shadow_x1]
        shadow_mean = float(np.mean(shadow_crop))
    else:
        # Check opposite side if near image boundary
        alt_x0 = max(0, x0 - int(bw * 1.5))
        alt_x1 = x0
        if alt_x1 > alt_x0:
            shadow_crop = sonar_gray[y0:y1, alt_x0:alt_x1]
            shadow_mean = float(np.mean(shadow_crop))
        else:
            shadow_mean = highlight_mean * 0.8

    # Surrounding ambient seafloor background sample (above and below along-track)
    bg_y0 = max(0, y0 - int(bh * 0.8))
    bg_y1 = min(h_img, y1 + int(bh * 0.8))
    surrounding_strip = sonar_gray[bg_y0:bg_y1, max(0, x0-20):min(w_img, x1+20)]
    ambient_mean = float(np.mean(surrounding_strip))

    # Calculate contrast ratio: ambient seabed vs cast shadow
    # True acoustic shadows drop close to zero (acoustic occlusion)
    contrast_ratio = max(1.0, (ambient_mean + 1e-3) / (shadow_mean + 1e-3))

    # Classify shadow evidence
    if contrast_ratio >= 1.60 and shadow_mean < 75.0:
        evidence_level = "Strong"
        explanation = (
            f"Distinct elevated acoustic structure. Pronounced specular highlight (mean {highlight_mean:.1f}) "
            f"coupled with significant down-range acoustic cast shadow (contrast ratio {contrast_ratio:.2f}x). "
            f"Acoustic shadow geometry can support object-height estimation when acquisition geometry permits."
        )
    elif contrast_ratio >= 1.25:
        evidence_level = "Moderate"
        explanation = (
            f"Detectable acoustic backscatter variation with moderate down-range acoustic shadow "
            f"(contrast ratio {contrast_ratio:.2f}x). Physical elevation probable."
        )
    elif contrast_ratio >= 1.10:
        evidence_level = "Weak"
        explanation = (
            f"Faint acoustic backscatter contrast without well-defined cast shadow. "
            f"Target may be low-relief, partially buried, or reflective sediment anomaly."
        )
    else:
        evidence_level = "None"
        explanation = (
            "Minimal acoustic shadow detected. Target signature may represent flat seabed backscatter "
            "variation or non-elevated seabed feature."
        )

    # Context similarity: compare target variance to local background variance
    target_std = float(np.std(target_crop))
    ambient_std = float(np.std(surrounding_strip))
    context_similarity = min(1.0, max(0.0, 1.0 - abs(target_std - ambient_std) / max(1.0, ambient_std + target_std)))

    aspect_ratio = float(bw) / max(1.0, float(bh))

    return AcousticValidationMetrics(
        highlight_mean=round(highlight_mean, 2),
        shadow_mean=round(shadow_mean, 2),
        shadow_contrast_ratio=round(contrast_ratio, 2),
        shadow_evidence_level=evidence_level,
        context_similarity=round(context_similarity, 3),
        estimated_aspect_ratio=round(aspect_ratio, 2),
        physics_explanation=explanation
    )

def compute_priority_rating(
    model_conf: float, 
    anomaly_score: float, 
    shadow_contrast: float, 
    context_sim: float, 
    is_anomaly: bool = False
) -> Tuple[str, float]:
    """
    Computes rigorous multi-factor priority rating:
    Priority Score = 0.40 * ModelConf + 0.35 * ShadowEvidence + 0.25 * (1 - ContextSim)
    Maps to: CRITICAL (>= 0.82), HIGH (>= 0.65), MEDIUM (>= 0.45), LOW (< 0.45).
    """
    # Normalize shadow contrast: 1.0 -> 0.0, >= 2.0 -> 1.0
    shadow_norm = min(1.0, max(0.0, (shadow_contrast - 1.0) / 1.0))
    
    # Salience from context: distinctiveness from surrounding seabed
    salience = 1.0 - context_sim

    if is_anomaly:
        priority_score = (
            0.45 * anomaly_score +
            0.30 * shadow_norm +
            0.25 * salience
        )
    else:
        priority_score = (
            0.40 * model_conf +
            0.35 * shadow_norm +
            0.25 * salience
        )

    priority_score = min(0.99, max(0.10, priority_score))

    if priority_score >= 0.82:
        level = "CRITICAL"
    elif priority_score >= 0.65:
        level = "HIGH"
    elif priority_score >= 0.45:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, round(priority_score, 4)

