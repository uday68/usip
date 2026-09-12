# Unsupervised Acoustic Anomaly Discovery Engine

**Module**: `ml/anomaly/`  
**Platform**: USIP (SIH26057)

---

## 1. Scientific Philosophy & Terminology

Supervised object detectors can only identify targets belonging to classes present in their training taxonomy (e.g. shipwrecks, pipelines, standard debris). However, underwater operations frequently encounter novel, uncatalogued acoustic signatures:
- Unidentified submerged objects (USO)
- Geomorphological vents / seafloor pockmarks
- Unusual synthetic seabed installations
- Exotic debris geometry

### Operational Honesty Rule:
We explicitly avoid unscientific claims like *"Unknown target detection is solved"*.  
Instead, the system designates these detections as:
- **`[EXPERIMENTAL] Acoustic Anomaly`**
- **`Anomaly Candidate`**
- **`Novelty Score: [0.0 - 1.0]`**
- **`Candidate requiring expert review`**

---

## 2. Technical Methodology

```
Acoustic Sonar Patch (128x128)
           │
           ▼
[Multi-Scale Acoustic Feature Extraction]
  ├── Statistical Moments (Mean, Variance, Skewness, Kurtosis)
  ├── Acoustic Texture Entropy
  ├── Sobel Spatial Gradient Distribution
  └── Multi-Orientation Gabor Filter Bank (0°, 45°, 90°, 135°)
           │
           ▼
[StandardScaler Normalization]
           │
           ▼
[MiniBatchKMeans Seafloor Facies Clustering]
  (Fitted strictly on unlabelled normal seabed from sss_ssl_dataset_N713_384)
           │
           ▼
[Novelty Distance Calculation]
  - Euclidean / Mahalanobis distance to nearest normal seabed centroid
  - Calibrated against the 95th percentile baseline of normal seafloor
           │
           ▼
[Anomaly Score Output: 0.0 to 1.0]
```

---

## 3. Artifact Storage
- **Centroids**: `ml/anomaly/weights/seabed_clusters.joblib`
- **Scaler**: `ml/anomaly/weights/feature_scaler.joblib`
- **Baseline Statistics**: `ml/anomaly/weights/seabed_stats.json`
- **Trained on**: 295 acoustic normal seabed tiles from survey runs `N7`, `N8`, `N9`, `N13`.

