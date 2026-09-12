import pytest
import numpy as np
from ml.anomaly.anomaly_engine import AnomalyDiscoveryEngine, extract_acoustic_features

def test_acoustic_feature_extraction():
    dummy_crop = np.random.normal(120, 25, (128, 128)).astype(np.uint8)
    feat = extract_acoustic_features(dummy_crop)
    assert isinstance(feat, np.ndarray)
    assert feat.shape[0] == 11  # moments + entropy + gradients + gabor energies
    assert not np.isnan(feat).any()

def test_anomaly_discovery_scoring():
    engine = AnomalyDiscoveryEngine()
    # If not fitted, verify it provides safe defaults
    dummy_seabed = np.random.normal(100, 15, (128, 128)).astype(np.uint8)
    res = engine.score_patch(dummy_seabed)

    assert "anomaly_score" in res
    assert "nearest_cluster_id" in res
    assert "nearest_cluster_dist" in res
    assert "is_anomaly" in res
    assert 0.0 <= res["anomaly_score"] <= 1.0

