# Known Limitations & Honest Scientific Disclosures

**Platform**: USIP — Underwater Sonar Intelligence Platform (SIH26057)

---

## 1. Scientific Honesty & Safe Fallbacks

In compliance with strict research integrity:

### A. Geolocation Metadata
- **AquaScan-1K**: Does not embed navigation GPS coordinates in its public benchmark release.
- **Honest Handling**: The system refuses to fabricate latitude/longitude coordinates. It returns:
  `"Geolocation unavailable — no valid navigation metadata associated with candidate."`
  The dashboard map clearly displays:
  `"Spatial map unavailable — navigation metadata not provided."`
- **SubPipeMini**: Features vehicle-frame Cartesian coordinates $(x, y, z)$. These are mapped to local origin projections while preserving raw metric distances.
- **AI4Shipwrecks**: Historical shipwrecks are matched to NOAA Thunder Bay Sanctuary registry coordinates.

### B. Physical Height Estimation
- Target elevation cannot be claimed with certainty without knowing vehicle altitude, acoustic slant-range, and beam grazing angle.
- **Honest Handling**: USIP measures **Specular Highlight to Acoustic Shadow Contrast Ratio** and states:
  `"Acoustic shadow geometry can support object-height estimation when acquisition geometry permits."`

### C. Unsupervised Anomaly Discovery
- We explicitly reject the claim that *"Unknown target discovery is solved"*.
- **Honest Handling**: The system flags candidates as:
  `[EXPERIMENTAL] Acoustic Anomaly` with an empirical `Novelty Score`. The human analyst remains the final operational decision-maker.

### D. Sensor Invariance
- Sonar images from different frequencies (e.g. 100 kHz LF vs 900 kHz HF) exhibit different speckle statistics. Cross-frequency domain shifts require multi-frequency normalization.

