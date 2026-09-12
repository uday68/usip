# Dataset Integration & Leakage-Resistant Partitioning

**Platform**: USIP — Underwater Sonar Intelligence Platform (SIH26057)

---

## 1. Primary Sonar Datasets Utilized

### A. AI4Shipwrecks
- **Modality**: Grayscale Side-Scan Sonar amplitude (swath width 1,728 px).
- **Target**: Anthropogenic historical shipwrecks in Lake Huron / Thunder Bay Sanctuary.
- **Annotations**: Pixel-level binary segmentation masks.
- **Partitioning**: Strictly **site-disjoint**.
  - Train: 12 shipwreck sites (`DM_Wilson`, `DR_Hanna`, `EB_Allen`, `Egyptian`, `Grecian`, `Heart_Failure`, `Isaac_M_Scott`, `Montana`, `Near_Shore`, `Oscar_T_Flint`).
  - Validation: 2 sites (`Pewabic`, `WP_Rend`).
  - Held-out Test: 13 sites (`Artificial_Reef`, `Barge_No_1`, `Corsair`, `Corsican`, `Haltiner_Barge`, `James_Davidson`, `Lucinda_van_Valkenburg`, `Monohansett`, `Monrovia`, `Shamrock`, `Viator`, `WH_Gilbert`, `WP_Thew`).
  - Site overlap between train and test: **0%**.

### B. AquaScan-1K
- **Modality**: RGB/colorized side-scan sonar waterfall images (1920×1080 px).
- **Target**: Anthropogenic marine debris and human-made artifacts on the seafloor.
- **Annotations**: YOLO format bounding boxes (Class 2: `Debris`).
- **Partitioning**: **Survey-date disjoint**.
  - Train: Dates `2025-06-09` through `2025-08-03`.
  - Val: Dates `2025-08-10` through `2025-08-18`.
  - Test: Dates `2025-08-28` through `2025-11-14`.

### C. SubPipeMini
- **Modality**: Dual-frequency High-Frequency (HF) and Low-Frequency (LF) PBM sonar waterfall records (5000×500 px).
- **Target**: Subsea pipelines, exposed umbilical lines, and cylindrical infrastructure.
- **Annotations**: YOLO format bounding boxes (Class 1: `Pipeline`) + synchronized vehicle telemetry (`EstimatedState.csv`).
- **Partitioning**: **Chronological sequence disjoint** with 20-frame acoustic buffer gap to eliminate along-track temporal overlap.

### D. SSS SSL Dataset (`sss_ssl_dataset_N713_384`)
- **Modality**: High-resolution acoustic TIFF patches (384×384 px) across survey lines `N7`, `N8`, `N9`, and `N13`.
- **Target**: Unlabelled normal seabed morphology.
- **Role**: Baseline training distribution for the Unsupervised Anomaly Discovery Engine.

---

## 2. Compilation Summary
- **Total Compiled Samples**: 4,453 unified sonar image tiles (640×640 px).
- **Train Split**: 3,036 samples (1,154 shipwreck, 386 debris, 1,496 pipeline).
- **Validation Split**: 403 samples.
- **Held-out Test Split**: 1,014 samples.

