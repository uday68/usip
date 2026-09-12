import numpy as np
from backend.schemas.models import AcousticMetrics

def validate_acoustic_signature(gray_img: np.ndarray, x: int, y: int, w: int, h: int) -> AcousticMetrics:
    """
    Validates physical acoustic properties of a detected candidate:
    1. Highlight backscatter intensity
    2. Acoustic shadow darkness and presence
    3. Surrounding seafloor background contrast
    """
    img_h, img_w = gray_img.shape
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(img_w, int(x + w)), min(img_h, int(y + h))

    target_region = gray_img[y1:y2, x1:x2]
    if target_region.size == 0:
        return AcousticMetrics(
            highlight_intensity=128.0,
            shadow_intensity=128.0,
            shadow_evidence_ratio=1.0,
            shadow_evidence_level="None",
            seafloor_contrast=0.0,
            seafloor_similarity="High similarity"
        )

    # Highlight intensity (90th percentile of brightest reflections)
    highlight_val = float(np.percentile(target_region, 90))

    # Examine adjacent down-range region for acoustic shadow zone
    # In SSS, shadow extends horizontally away from the nadir/track
    sw_x1 = min(img_w - 1, x2)
    sw_x2 = min(img_w, int(x2 + max(20, w * 1.2)))
    shadow_region = gray_img[y1:y2, sw_x1:sw_x2] if sw_x2 > sw_x1 else None

    # Also check vertical/surrounding margins for ambient seabed backscatter
    pad = 30
    bg_y1 = max(0, y1 - pad)
    bg_y2 = min(img_h, y2 + pad)
    bg_x1 = max(0, x1 - pad)
    bg_x2 = min(img_w, x2 + pad)
    background_region = gray_img[bg_y1:bg_y2, bg_x1:bg_x2]
    bg_mean = float(np.mean(background_region)) if background_region.size > 0 else 100.0

    if shadow_region is not None and shadow_region.size > 0:
        shadow_val = float(np.percentile(shadow_region, 15))
    else:
        # Check lowest 10% in target border
        shadow_val = float(np.percentile(target_region, 10))

    # Highlight to Shadow Contrast Ratio
    ratio = (highlight_val - shadow_val) / max(1.0, highlight_val + shadow_val)

    if ratio > 0.45 and shadow_val < 65:
        shadow_level = "Strong"
    elif ratio > 0.25:
        shadow_level = "Moderate"
    elif ratio > 0.12:
        shadow_level = "Weak"
    else:
        shadow_level = "None"

    # Contrast against surrounding seabed
    seafloor_contrast = abs(highlight_val - bg_mean) / max(1.0, bg_mean)
    if seafloor_contrast > 0.6:
        seafloor_sim = "Low similarity" # Distinct from seabed
    elif seafloor_contrast > 0.3:
        seafloor_sim = "Moderate"
    else:
        seafloor_sim = "High similarity" # Blends into seabed

    return AcousticMetrics(
        highlight_intensity=round(highlight_val, 1),
        shadow_intensity=round(shadow_val, 1),
        shadow_evidence_ratio=round(ratio, 2),
        shadow_evidence_level=shadow_level,
        seafloor_contrast=round(seafloor_contrast, 2),
        seafloor_similarity=seafloor_sim
    )

def compute_priority(confidence: float, acoustic: AcousticMetrics, is_anomaly: bool = False, anomaly_score: float = 0.0) -> str:
    """
    Ranks target candidates based on multi-factor physics fusion.
    """
    score = 0.0

    if not is_anomaly:
        score += confidence * 0.5
        if acoustic.shadow_evidence_level == "Strong":
            score += 0.35
        elif acoustic.shadow_evidence_level == "Moderate":
            score += 0.20
        elif acoustic.shadow_evidence_level == "Weak":
            score += 0.05

        if acoustic.seafloor_similarity == "Low similarity":
            score += 0.15
        elif acoustic.seafloor_similarity == "Moderate":
            score += 0.08
    else:
        score += (anomaly_score or 0.5) * 0.45
        if acoustic.shadow_evidence_level in ["Strong", "Moderate"]:
            score += 0.35
        if acoustic.seafloor_similarity == "Low similarity":
            score += 0.20

    if score >= 0.75:
        return "HIGH" if score < 0.90 else "CRITICAL"
    elif score >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"
