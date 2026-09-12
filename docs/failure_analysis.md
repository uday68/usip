# Acoustic Failure Analysis & Model Error Taxonomy

**System**: USIP — Underwater Sonar Intelligence Platform (SIH26057)

---

## 1. Acoustic Failure Taxonomy

Acoustic imagery presents unique challenges not encountered in optical vision. We analyze five major error modes:

### Mode 1: False Positive — Natural Rock Outcrop Interpreted as Target
- **Cause**: High-relief boulders or exposed bedrock produce sharp specular highlights and acoustic cast shadows identical to man-made block debris.
- **Mitigation in USIP**: The **Seafloor Context Similarity** metric detects whether the surrounding seabed contains dense natural ripple or gravel facies, down-weighting the priority of isolated rocks in rocky terrain.

### Mode 2: False Negative — Low-Contrast / Partially Buried Debris Missed
- **Cause**: Small plastic debris, tires, or containers partially covered by sediment silt yield weak backscatter contrast and negligible cast shadows.
- **Mitigation in USIP**: The **Acoustic CLAHE Preprocessing Pipeline** boosts subtle backscatter differences before passing tiles to the detector.

### Mode 3: Acoustic Ambiguity — Shadow Resembles Target / Multi-Path Reflection
- **Cause**: Towed vehicle altitude fluctuations cause grazing angle changes, elongating or compressing acoustic shadows unpredictably.
- **Mitigation in USIP**: Multi-factor priority ranking balances raw model confidence with measured shadow contrast ratio.

### Mode 4: Domain Shift — Sonar Frequency & Swath Variations
- **Cause**: Switching between 100 kHz (LF, penetrates sediment, lower resolution) and 900 kHz (HF, high surface detail) alters speckle statistics.
- **Mitigation in USIP**: The dataset compiler trains on both HF and LF records from SubPipeMini to enforce cross-frequency invariance.

### Mode 5: Unknown Anomaly — Target Does Not Belong to Trained Classes
- **Cause**: Unusual man-made debris or seabed features absent from training data cannot be classified by the supervised model.
- **Mitigation in USIP**: The **Unsupervised Anomaly Discovery Engine** detects departure from normal seabed clusters and flags candidates as `[EXPERIMENTAL] Acoustic Anomaly`.

