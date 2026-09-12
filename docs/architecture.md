# USIP Platform Architecture

**System**: Underwater Sonar Intelligence Platform  
**Problem Statement**: SIH26057  

---

## High-Level Architecture

The platform is designed around a strictly decoupled, modular pipeline where acoustic physics constraints govern data processing:

```
[Side-Scan Sonar Imagery] (TIFF / PNG / PBM / Waterfall)
           │
           ▼
[1. Preprocessing Pipeline] ──► Slant-range / Dynamic Range / Despeckle / CLAHE / Tiling
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[2. Supervised Target Detector]   [3. Unsupervised Anomaly Engine]
  - Real trained YOLOv8n            - Acoustic Feature Extractor
  - Shipwreck / Pipeline / Debris   - Normal Seabed Clusters
  - Model Confidence & BBoxes       - Novelty Distance Scoring
     │                              │
     └─────┬────────────────────────┘
           ▼
[4. Candidate Fusion Layer] ──► Unified Candidate Schema & Deduplication
           │
           ▼
[5. Sonar-Aware Validation] ──► Specular Highlight / Shadow Contrast / Priority Ranking
           │
     ┌─────┴────────────────────────┐
     ▼                              ▼
[6. Metadata Ingestion]           [7. Relational Database]
  - Telemetry CSV / Sanctuary Reg   - PostgreSQL / PostGIS / SQLite
  - Depth, Altitude, Heading        - Surveys, Images, Candidates, Reviews
     │                              │
     └─────┬────────────────────────┘
           ▼
[8. FastAPI Backend Server] (REST API: /api/v1/inference, /candidates, /reviews, /reports)
           │
           ▼
[9. Tactical Sonar Dashboard] (Viewer, Crosshairs, Candidate Panel, Map, Review Queue)
```

---

## Component Separation

1. **`ml/preprocessing/`**: Standalone computer vision and sonar acoustics routines. Has zero dependencies on web frameworks or databases.
2. **`ml/detection/`**: Deep learning model loading, inference, and bounding box coordinate transformation.
3. **`ml/anomaly/`**: Unsupervised feature extraction and distance scoring based on seafloor facies baseline.
4. **`ml/validation/`**: Physical validation logic evaluating highlight-to-shadow contrast and surrounding seafloor similarity.
5. **`ml/metadata/`**: Ingestion of external navigation tables, coordinate projections, and honest fallbacks.
6. **`backend/app/`**: REST API endpoints, database ORM sessions, asynchronous task handling, and static file serving.
7. **`backend/static/`**: Tactical dark dashboard supporting real-time visualization, human-in-the-loop candidate confirmation, and report generation.

