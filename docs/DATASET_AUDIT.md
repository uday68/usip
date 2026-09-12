# Side-Scan Sonar (SSS) Comprehensive Dataset Audit & Forensic Report
**Project**: USIP — Underwater Sonar Intelligence Platform  
**Problem Statement**: SIH26057 — AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Audit Date**: September 11, 2026  
**Auditor**: Lead ML + Backend + Systems Engineer  

---

## 1. Dataset Name & Identification
The platform utilizes four real-world benchmark Side-Scan Sonar (SSS) datasets discovered and verified in the environment:
1. **AI4Shipwrecks**: Benchmark side-scan sonar dataset targeting underwater archaeological shipwrecks collected across diverse Great Lakes surveys (Lake Huron / Thunder Bay Sanctuary).
2. **AquaScan-1K**: Benchmark side-scan sonar target dataset containing marine debris and anthropogenic seabed artifacts collected across 13 distinct dates.
3. **SubPipeMini**: Dual-frequency acoustic pipeline inspection survey dataset containing continuous side-scan sonar waterfall records (HF and LF) alongside synchronized vehicle telemetry.
4. **SSS SSL Dataset (`N713_384`)**: Deep multi-survey self-supervised acoustic dataset capturing natural seabed morphology across survey tracks `N7`, `N8`, `N9`, and `N13`.

---

## 2. Dataset Physical Sizes on Disk
- `D:\AI4Shipwrecks.zip`: 1,216,701,275 bytes (**1.13 GB**)
- `D:\AquaScan-1K.zip`: 1,132,588,834 bytes (**1.05 GB**)
- `D:\SubPipeMini.zip`: 6,078,303,615 bytes (**5.66 GB**) (and full `SubPipe.zip`: 28.01 GB)
- `D:\sss_ssl_dataset_N713_384.zip`: 9,302,423,575 bytes (**8.66 GB**)
- `D:\USIP_SSL_SAMPLE`: 295 uncompressed high-resolution acoustic TIFF tiles (**170 MB**)
- `D:\USIP_DEMO_DATA`: 15 curated high-fidelity evaluation samples (**26 MB**)

---

## 3. Directory Structures

### A. AI4Shipwrecks
```text
AI4Shipwrecks/
├── train/
│   ├── images/   # 141 PNG images (grayscale acoustic amplitude)
│   └── labels/   # 141 PNG binary segmentation masks (0=seabed, 1=wreck)
├── test/
│   ├── images/   # 120 PNG images
│   └── labels/   # 120 PNG binary segmentation masks
└── extras/
    └── terrain/
        ├── images/ # 25 PNG seabed terrain images
        └── labels/ # 25 PNG binary terrain masks
```

### B. AquaScan-1K
```text
AquaScan-1K/
├── classes.txt   # Class label: "Human" (denoting human-made debris)
├── images/       # 539 PNG images (1920x1080)
└── labels/       # 543 TXT files in YOLO format (class_idx xc yc w h)
```

### C. SubPipeMini
```text
SubPipeMini/
├── config.yaml
├── DATA/
│   ├── Acceleration.csv
│   ├── AngularVelocity.csv
│   ├── Depth.csv
│   ├── Altitude.csv
│   ├── ForwardDistance.csv
│   ├── EstimatedState.csv      # Synchronized AUV navigation (timestamp, x, y, z, roll, pitch, yaw)
│   ├── Pressure.csv
│   ├── Temperature.csv
│   ├── WaterVelocity.csv
│   ├── Cam0_images/            # Optical GoPro frames (16,170 images - excluded from sonar pipeline)
│   ├── Cam1_images/            # Optical camera frames (2,160 images)
│   ├── SSS_HF_images/          # High-Frequency SSS waterfall images
│   │   ├── Image/              # 539 PBM images (5000 x 500)
│   │   └── YOLO_Annotation/   # 435 TXT labels in YOLO format + classes.txt
│   └── SSS_LF_images/          # Low-Frequency SSS waterfall images
│       ├── Image/              # 539 PBM images (5000 x 500)
│       └── YOLO_Annotation/   # 455 TXT labels in YOLO format + classes.txt
```

### D. SSS SSL Sample (`USIP_SSL_SAMPLE`)
```text
USIP_SSL_SAMPLE/
└── N*.tiff                     # 295 tiles, 384x384 single-channel acoustic backscatter
```

---

## 4. Image Count Analysis
- **AI4Shipwrecks**: 286 SSS images (141 train + 120 test + 25 terrain extras).
- **AquaScan-1K**: 539 SSS images.
- **SubPipeMini**: 1,078 SSS acoustic images (539 HF + 539 LF). Note: The 16,170 `Cam0` frames are optical camera feeds and are NOT acoustic sonar.
- **SSS SSL Patches**: 295 benchmark tiles extracted locally from 50GB multi-part sonar acoustic surveys.
- **Total Combined SSS Images Audited**: **2,198 true acoustic side-scan sonar images**.

---

## 5. Annotation Count & Target Statistics
- **AI4Shipwrecks**:
  - 286 binary segmentation masks (pixel-level).
  - Train: 87 positive masks containing active shipwrecks, 54 negative masks (pure ambient seabed).
  - Test: 74 positive masks containing active shipwrecks, 46 negative masks.
  - Bounding boxes extracted: 161 tight rectangular bounding boxes.
- **AquaScan-1K**:
  - 543 YOLO annotation files containing **570 verified bounding boxes** (all class `0`).
- **SubPipeMini**:
  - SSS HF: 435 YOLO annotation files (**435 bounding boxes** for pipeline spans).
  - SSS LF: 455 YOLO annotation files (**455 bounding boxes** for pipeline spans).
  - Total: **890 pipeline bounding boxes**.
- **Combined Target Annotations**: **1,621 target annotations** (Shipwrecks: 161, Debris: 570, Pipeline: 890).

---

## 6. Classes
The target taxonomy supported by ground truth:
1. `Shipwreck` (Large elevated acoustic structure with multi-point specular highlights and extensive acoustic shadows).
2. `Pipeline / Subsea Cylinder` (Continuous linear cylindrical acoustic structure with continuous parallel cast shadow).
3. `Anthropogenic Debris / Man-Made Target` (Discrete localized high-backscatter target distinct from ambient sediment).
4. `[EXPERIMENTAL] Acoustic Anomaly` (Unsupervised out-of-distribution cluster candidate identified by embedding distance from normal seabed patches).

---

## 7. Image Dimensions & Aspect Ratios
- **AI4Shipwrecks**: Fixed swath width 1,728 pixels × along-track length 1,859 to 6,749 pixels. Single-channel grayscale (`mode=L`).
- **AquaScan-1K**: Fixed 1920 × 1080 pixels (`mode=RGB` colorized sonar waterfall displays).
- **SubPipeMini**: Fixed 5000 × 500 pixels (`mode=RGB` `.pbm` format, where 5000 is across-track range and 500 is along-track pings).
- **SSS SSL Patches**: Fixed 384 × 384 pixels (`mode=RGBA`/grayscale single-band amplitude).

---

## 8. Annotation Format Specification
- **AI4Shipwrecks**: PNG 8-bit single-channel masks (`0` = background/water column/seabed, `1` = shipwreck hull/structure).
- **AquaScan-1K**: Standard YOLO format: `<class_id> <x_center> <y_center> <width> <height>` (normalized `[0.0, 1.0]`).
- **SubPipeMini**: Standard YOLO format: `<class_id> <x_center> <y_center> <width> <height>`.

---

## 9. Metadata Availability & Forensics
- **SubPipeMini**: Rich AUV sensor telemetry is available in `EstimatedState.csv`:
  - Fields: `image`, `timestamp`, `x (m)`, `y (m)`, `z (m)`, `phi (rad)`, `theta (rad)`, `psi (rad)`, `u (m/s)`, `v (m/s)`, `w (m/s)`, `depth (m)`, `alt (m)`.
  - Enables local cartesian metric coordinate assignment, altitude-based slant-range calculation, and vehicle heading correlation.
- **AquaScan-1K**: Survey timestamps embedded in filename conventions:
  - Example: `0002b00e-Screenshot_2025-08-10_23.00.36.png` contains survey date `2025-08-10` and time `23:00:36`. Navigation lat/lon coordinates are not embedded.
  - In accordance with the Honesty Rule: Geolocation is flagged as unavailable (`"Geolocation unavailable — no valid navigation metadata associated with candidate"`).
- **AI4Shipwrecks**: Contains historic shipwreck site names (`EB_Allen`, `DR_Hanna`, `Montana`, `Pewabic`, `Lucinda_van_Valkenburg`, etc.) located within NOAA Thunder Bay National Marine Sanctuary (Lake Huron). Coordinates are indexed via the sanctuary shipwreck registry.

---

## 10. Existing Splits Analysis
- **AI4Shipwrecks**: Explicit author-defined partition into `train/` (141 images) and `test/` (120 images) plus `extras/terrain/` (25 images).
- **AquaScan-1K**: Flat directory structure (`images/`, `labels/`). No predefined split.
- **SubPipeMini**: Continuous survey sequence across chunks. No predefined split.

---

## 11. Duplicate and Near-Duplicate Analysis
- An MD5 hash audit of `AquaScan-1K` detected **1 exact duplicate image** (filtered out during dataset compilation).
- Consecutive along-track ping waterfall records in `SubPipeMini` have significant spatial overlap between adjacent frames (up to 70% overlap between ping increments).
- Adjacent frames in `AI4Shipwrecks` represent contiguous along-track survey passes over the same shipwreck hull.

---

## 12. Potential Data Leakage Risks (CRITICAL)
1. **Site Leakage**: Randomly shuffling images of the same shipwreck (e.g. putting `EB_Allen_01` in train and `EB_Allen_02` in test) is severe data leakage. The model would memorize the unique acoustic texture of that specific wreck rather than learning generalized shipwreck morphology.
2. **Temporal / Track Leakage**: In `SubPipeMini`, adjacent waterfall frames occur seconds apart. Random splitting would place contiguous pings in both train and test.
3. **Survey Date Leakage**: In `AquaScan-1K`, images from the same survey date/run share identical water column conditions, sonar gain, and sediment ripple frequencies.

---

## 13. Recommended Leakage-Resistant Split Strategy

### A. AI4Shipwrecks: Strict Site-Disjoint Split (Verified)
- **Train Sites (12 Wrecks)**: `DM_Wilson`, `DR_Hanna`, `EB_Allen`, `Egyptian`, `Grecian`, `Heart_Failure`, `Isaac_M_Scott`, `Montana`, `Near_Shore`, `Oscar_T_Flint`, `Pewabic`, `WP_Rend`.
- **Validation / Held-out Test Sites (13 Wrecks)**: `Artificial_Reef`, `Barge_No_1`, `Corsair`, `Corsican`, `Haltiner_Barge`, `James_Davidson`, `Lucinda_van_Valkenburg`, `Monohansett`, `Monrovia`, `Shamrock`, `Viator`, `WH_Gilbert`, `WP_Thew`.
- **Overlap**: **0% (strictly disjoint)**.

### B. AquaScan-1K: Date-Disjoint Partition
- **Train Set**: Surveys from `2025-06-09` through `2025-08-03` (~70%).
- **Validation Set**: Surveys from `2025-08-10` through `2025-08-18` (~15%).
- **Held-out Test Set**: Surveys from `2025-08-28`, `2025-08-29`, and `2025-11-14` (~15%).

### C. SubPipeMini: Sequence Chunk-Disjoint Partition
- Split along temporal index chunks with an acoustic exclusion buffer between splits to prevent overlap leakage.

---

## 14. Recommended Detector Model Choice
- **Architecture**: **YOLOv8 / YOLO11 Nano (`yolov8n`)** with custom anchor/feature tuning for sonar acoustic characteristics.
- **Justification**:
  1. Marine survey operations require low-latency edge deployment on AUVs/towed sonars with restricted compute budgets.
  2. Local GPU is an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). Lightweight architecture trains reliably without OOM errors.
  3. Seamlessly unifies multi-source datasets into a single multi-class target detector.
  4. Fully documented in `docs/model_selection.md`.

---

## 15. Recommended Acoustic Preprocessing Pipeline
Standard RGB computer vision operations corrupt acoustic physics. The preprocessing module enforces:
1. **Channel Unification & Dynamic Range Normalization**: Convert colorized displays back to true acoustic intensity or apply CLAHE on luminance.
2. **Slant-Range & Water Column Muting**: Identify acoustic nadir/altitude blind zones.
3. **Adaptive Contrast Enhancement**: Bilateral despeckle filtering that preserves high-frequency highlight edges while smoothing acoustic speckle.
4. **Adaptive Tiling with 20% Overlap**: Needed for high-aspect ratio sonar strips (such as 5000×500 waterfall pings) to avoid downsampling small debris to sub-pixel noise.

---

## 16. Recommended Training & Evaluation Strategy
1. **Supervised Target Detector**: Train a multi-class acoustic detector on unified train splits across the 3 target classes: `Shipwreck`, `Pipeline`, and `Debris`.
2. **Evaluated on Held-Out Test Sites Only**: Compute true mAP@50, mAP@50:95, Precision, Recall, and Confusion Matrix.
3. **Unsupervised Anomaly Discovery**: Feature extraction via acoustic patch embeddings trained strictly on normal seabed patches (`USIP_SSL_SAMPLE`), evaluated by anomaly novelty score against targets and unseen seabed anomalies.

---

## 17. Known Limitations
1. Navigation GPS metadata is absent in `AquaScan-1K`. The system will explicitly flag: `"Geolocation unavailable — no valid navigation metadata associated with candidate"`.
2. Exact physical target elevation cannot be claimed without knowing towfish altitude and slant-range angle; the system will report highlight-to-shadow contrast and estimated shadow length ratio rather than fabricating physical height.
3. Optical camera feeds (`Cam0`, `Cam1`) in `SubPipeMini` are excluded from the sonar pipeline to ensure pure acoustic modeling integrity.

