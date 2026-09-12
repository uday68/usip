import pytest
import numpy as np
from ml.validation.validator import validate_acoustic_signature, compute_priority_rating

def test_validate_acoustic_signature_highlight_and_shadow():
    sonar_canvas = np.ones((500, 500), dtype=np.uint8) * 100
    # Specular highlight: bx=200, by=200, bw=50, bh=50
    sonar_canvas[200:250, 200:250] = 240
    # Down-range acoustic cast shadow: x from 250 to 320
    sonar_canvas[200:250, 250:320] = 15

    metrics = validate_acoustic_signature(sonar_canvas, 200, 200, 50, 50)
    assert metrics.highlight_mean > 200
    assert metrics.shadow_mean < 30
    assert metrics.shadow_contrast_ratio > 2.0
    assert metrics.shadow_evidence_level == "Strong"
    assert "Acoustic shadow geometry can support object-height estimation" in metrics.physics_explanation

def test_priority_rating_computation():
    # High confidence + strong shadow -> CRITICAL
    level, score = compute_priority_rating(
        model_conf=0.95,
        anomaly_score=0.1,
        shadow_contrast=2.5,
        context_sim=0.2,
        is_anomaly=False
    )
    assert level == "CRITICAL"
    assert score >= 0.82

    # Low confidence + no shadow -> LOW
    level_low, score_low = compute_priority_rating(
        model_conf=0.35,
        anomaly_score=0.1,
        shadow_contrast=1.05,
        context_sim=0.9,
        is_anomaly=False
    )
    assert level_low == "LOW"
    assert score_low < 0.45

