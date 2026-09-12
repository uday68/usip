# USIP — Underwater Sonar Intelligence Platform

[![SIH Problem](https://img.shields.io/badge/SIH%202026-Problem%20SIH26057-00f2fe.svg)](https://www.sih.gov.in)
[![Acoustic Engine](https://img.shields.io/badge/Dual%20AI%20Engine-YOLOv8n%20%2B%20GMM%20Anomaly-10b981.svg)](#architecture)
[![Dataset](https://img.shields.io/badge/Unified%20SSS%20Dataset-4%2C453%20Acoustic%20Samples-3b82f6.svg)](#dataset-compilation)
[![License](https://img.shields.io/badge/License-MIT-slate.svg)](#)

> **SIH 2026 Problem SIH26057**: AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery.  
> **USIP** is an end-to-end, runnable mission intelligence system engineered for real Side-Scan Sonar (SSS) hydrographic surveys. It unifies deep object detection, unsupervised acoustic anomaly discovery, physics-aware highlight/shadow acoustic validation, geospatial telemetry association, and an interactive tactical marine dashboard.

---

## 🌟 Key Capabilities & Architectural Innovations

1. **Dual-Branch AI Inference Engine**:
   - **Branch 1 (Supervised Known Target Detector)**: Custom-trained **YOLOv8n** on 4,453 unified side-scan sonar samples across three distinct marine classes: `Shipwreck`, `Pipeline`, and `Debris`.
   - **Branch 2 (Unsupervised Acoustic Anomaly Engine)**: Lightweight **GMM / Centroid Distance Model** trained on acoustic embeddings of verified normal seabed tiles (`sss_ssl_dataset_N713_384`). Discovers novel acoustic anomalies and unmodeled geomorphologies with zero false certainty.
2. **Physics-Aware Acoustic Validation**:
   - Computes **Specular Highlight to Acoustic Shadow Contrast Ratio** ($C_s = \mu_h / (\mu_s + \epsilon)$) from adjacent acoustic shadow zones.
   - Evaluates **Seafloor Context Similarity** ($S_c$) to suppress false alarms caused by natural sand ripples and reverberation.
   - Generates a **Composite Priority Rating** (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) factoring detector confidence, anomaly score, and acoustic physics validity.
3. **Strict Leakage-Free Dataset Compilation**:
   - Multi-source compilation of **AI4Shipwrecks**, **AquaScan-1K**, **SubPipeMini**, and **sss_ssl_dataset_N713_384**.
   - Strict site-disjoint, shipwreck-disjoint, survey sequence-disjoint, and survey date-disjoint splits: **3,036 Train / 403 Val / 1,014 Held-Out Test**.
4. **Geospatial & Marine Telemetry Association**:
   - Ingests survey metadata and maps detections to geographic coordinates (WGS-84).
   - Pre-populated with official NOAA Lake Huron National Marine Sanctuary coordinates for verified shipwrecks.
   - Fallback indicators when coordinates are not available, preventing fabricated geolocations.
5. **Interactive Tactical Marine Dashboard**:
   - Built with Tailwind CSS, HTML5 Canvas, Leaflet.js, and Lucide icons.
   - Real-time crosshair coordinate tracking, Raw vs. Enhanced CLAHE toggle, layer visibility filters.
   - **Human-in-the-Loop Analyst Review Queue**: Operators can tag candidates as `Confirmed`, `Rejected`, or `Uncertain` with operational notes saved directly to the database.
   - **Mission Intelligence Export Center**: Instant 1-click export of complete mission dossiers to structured **JSON** and **CSV**.

---

## 📊 Dataset Compilation & Forensic Audit

| Dataset | Modality / Sensor | Original Size | Train Samples | Val Samples | Held-Out Test Samples | Target Classes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AI4Shipwrecks** | Klein 3000 / Edgetech SSS | ~2.5 GB | 1,154 | 158 | 388 | Shipwreck (Hull, Keel, Debris Field) |
| **AquaScan-1K** | High-Res Towfish SSS | ~1.8 GB | 386 | 54 | 130 | Marine Debris, Man-made Objects |
| **SubPipeMini** | Autonomous ROV SSS | ~1.2 GB | 1,496 | 191 | 496 | Subsea Hydrocarbon Pipeline |
| **sss_ssl_dataset**| Multi-beam / SSS Seabed | ~700 MB | 295 Normal Seabed Patches | — | — | Unsupervised Normal Acoustic Seabed Baseline |
| **Unified Total** | **640x640 Unified Format** | **~6.2 GB** | **3,036** | **403** | **1,014** | **3 Classes + Acoustic Anomalies** |

*All data pipelines and splits are managed at `ml/data/compiler.py` and `configs/settings.py`.*

---

## 📈 Empirical Evaluation & Model Metrics

The supervised detector was evaluated on **1,014 completely unseen, held-out test sonar samples**:

- **Overall mAP@50**: `0.1316` (0.132)
- **Overall Precision**: `0.1783` (17.8%)
- **Overall Recall**: `0.2443` (24.4%)
- **Class Breakdown (mAP@50)**:
  - **Marine Debris**: `0.2658` (26.6%)
  - **Shipwreck**: `0.1135` (11.4%)
  - **Subsea Pipeline**: `0.0156` (1.6%)
- **Inference Latency**: `12.4 ms` per tile on NVIDIA GeForce RTX 3050 Laptop GPU (over 80 FPS capability).
- Evaluation artifacts, confusion matrices, and precision-recall curves are preserved under `reports/evaluation/`.

---

## 🚀 Quickstart & Launch Instructions

### Option 1: 1-Click Windows Launcher (Recommended)
Simply double-click the batch file in the repository root:
```cmd
run_usip.bat
```
This automatically activates the dedicated environment on `D:\usip_env` and launches the Uvicorn ASGI server on port `8000`.

### Option 2: Command Line Launch
```powershell
# Activate dedicated virtual environment
D:\usip_env\Scripts\Activate.ps1

# Launch backend server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at **[http://localhost:8000](http://localhost:8000)**.

### Option 3: Docker Deployment
```bash
docker-compose up --build
```

---

## 🧪 Verification & Automated Test Suite

USIP includes a full pytest regression suite covering all preprocessing, anomaly detection, metadata extraction, physics validation, and API endpoints:

```powershell
# Run the complete test suite
pytest tests/ -v

# Run the end-to-end integration test
python scripts/test_end_to_end.py
```
**Results**: All 13 tests pass with 100% integrity.

---

## 🎯 SIH Judge Demonstration Walkthrough

Follow these steps for an interactive demonstration for the evaluation panel:

1. **Launch the Dashboard**: Double-click `run_usip.bat` and navigate to `http://localhost:8000`.
2. **Inspect Initial Detection**:
   - The dashboard opens by default with **Wreck: Montana (NOAA Lake Huron)**.
   - Observe the 3 high-confidence detections bounded in green.
   - Note the Leaflet tactical map rendering the official NOAA sanctuary location (45.0315°N, -83.1878°W) with survey trackline.
3. **Demonstrate Sonar Physics Validation**:
   - Click on any detection candidate in the Target Candidates table.
   - View the **Acoustic Physics & Explainability Card**: observe the **Highlight-to-Shadow Contrast Ratio** and **Seafloor Context Similarity**.
4. **Test Real-Time SSS Presets**:
   - Click **Wreck: Pewabic** to view a historic 1865 steamer detection.
   - Click **Pipeline** to see subsea pipe corridor tracking from SubPipeMini.
   - Click **Debris Cluster** to see high-density anthropogenic debris from AquaScan-1K.
   - Click **SSL Seabed Tile** to demonstrate background normalization.
   - Click **Acoustic Anomaly [Exp]** to trigger the unsupervised anomaly detector.
5. **Interactive Controls**:
   - Toggle **Raw vs. Enhanced (CLAHE)** to inspect how contrast equalization enhances sonar shadow boundaries.
   - Hover over the sonar canvas to observe real-time cursor coordinate telemetry.
6. **Human-in-the-Loop Workflow**:
   - In the **Analyst Verification Queue**, select `Confirmed` or `Rejected` and enter operational notes.
   - Click Confirm — the status updates instantly and records to SQLite (`usip_sonar.db`).
7. **Mission Dossier Export**:
   - Click **Export JSON** or **Export CSV** in the top navigation bar to download the structured mission intelligence report.

---

## 📁 Repository Structure

```text
sih26057/
├── backend/                        # FastAPI Backend & Dashboard
│   ├── app/
│   │   ├── api/endpoints.py        # REST endpoints (/inference, /candidates, /reviews, /reports)
│   │   ├── db/session.py           # SQLAlchemy database session
│   │   ├── models/entities.py      # ORM models (Surveys, Images, Candidates, Reviews)
│   │   ├── schemas/api_models.py   # Pydantic validation schemas
│   │   ├── services/               # Inference & Reporting Services
│   │   └── main.py                 # FastAPI application factory
│   └── static/
│       ├── index.html              # Tactical marine dashboard
│       └── results/                # Output analysis visualizations
│
├── configs/                        # Unified project configuration
│   └── settings.py
│
├── demo_samples/                   # Curated benchmark SSS samples for judge demonstration
├── docs/                           # Architecture, audit, and engineering documentation
│   ├── DATASET_AUDIT.md            # Forensic 17-point audit of all 4 datasets
│   ├── IMPLEMENTATION_STATUS.md    # 20-point milestone checklist
│   ├── architecture.md             # System architecture specification
│   ├── model_selection.md          # Architectural justification for detector
│   └── ...                         # Detailed module documentation
│
├── ml/                             # Machine Learning & Sonar Physics Modules
│   ├── anomaly/                    # GMM Unsupervised Anomaly Engine
│   ├── data/                       # Dataset Compiler & Leakage-Free Splitter
│   ├── detection/weights/best.pt   # Custom-trained YOLOv8n detector weights
│   ├── metadata/                   # NOAA Geolocation & ROV Telemetry Parsers
│   ├── preprocessing/              # CLAHE, Tiling, Quality Control, Augmentation
│   ├── training/                   # Detector training scripts
│   └── validation/                 # Acoustic Shadow & Context Physics Validators
│
├── reports/evaluation/             # Metrics, confusion matrices, diagnostic curves
├── tests/                          # 13-point automated test suite
├── run_usip.bat                    # 1-click Windows launcher
└── README.md                       # Master project documentation
```

---

## ⚖️ Limitations & Operational Guidance
- While the system operates at low latency (<15 ms per frame), true side-scan sonar image quality depends heavily on altitude above seafloor and towfish speed.
- In low-contrast seafloors (such as uniform mud or silt), shadow definition decreases; the physics validator automatically flags such candidates as having lower shadow confidence.
- For complete field deployment instructions, see `docs/deployment.md`.

*Built for Smart India Hackathon 2026.*
