# SIH Judge Demonstration Guide (1-Click Workflow)

**Project**: USIP — Underwater Sonar Intelligence Platform  
**Problem Statement**: SIH26057  

---

## 1. Objective of Demonstration
Demonstrate an end-to-end, runnable AI system operating on real benchmark side-scan sonar (SSS) imagery:
- Detect elevated seabed targets (`Shipwreck`, `Pipeline`, `Debris`) with actual model confidence.
- Identify novel acoustic anomalies (`[EXPERIMENTAL] Acoustic Anomaly`) using normal seabed cluster distance.
- Validate acoustic physics (specular highlight-to-shadow contrast ratio).
- Compute composite priority ranking (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
- Associate navigation coordinates when available, and provide honest fallback when absent.
- Provide human-in-the-loop analyst review and export mission reports to JSON and CSV.

---

## 2. 10-Step Judge Demonstration Script

1. **Launch Platform**:
   Double click `run_usip.bat` or run:
   ```powershell
   D:\usip_env\Scripts\python.exe -m uvicorn backend.app.main:app --port 8000
   ```
2. **Open Tactical Dashboard**:
   Navigate to **`http://localhost:8000`** in any web browser.
3. **Select Preset 1: Shipwreck**:
   Click the **Shipwreck (AI4Shipwrecks)** demo button.
   - Note the multi-point specular highlight and long cast shadow.
   - Observe Model Confidence (e.g. ~92%) and Priority (`CRITICAL`).
   - Notice verified Lake Huron / Thunder Bay Sanctuary coordinates plotted on the map.
4. **Select Preset 2: Subsea Pipeline**:
   Click the **Pipeline (SubPipeMini)** demo button.
   - Observe elongated linear bounding box with parallel continuous acoustic shadow.
   - Notice synchronized vehicle telemetry (AUV depth, altitude, heading).
5. **Select Preset 3: Anthropogenic Debris**:
   Click the **Marine Debris (AquaScan-1K)** demo button.
   - Observe detection of compact human-made seabed target.
   - Note the honest metadata status: *"Geolocation unavailable — no valid navigation metadata associated with candidate"*.
6. **Select Preset 4: Novel Acoustic Anomaly**:
   Click the **Acoustic Anomaly (Experimental)** demo button.
   - Note the distinct golden badge: `[EXPERIMENTAL] Acoustic Anomaly`.
   - Observe Anomaly Score (e.g. 0.85) and Nearest Cluster Distance indicating departure from normal seafloor.
7. **Inspect Acoustic Evidence**:
   Click any target in the candidate list to inspect:
   - Highlight Mean vs Shadow Mean.
   - Shadow Contrast Ratio.
   - Physical narrative explaining the acoustic return.
8. **Toggle Sonar Visualizations**:
   Use the **Raw vs Enhanced (CLAHE)** toggle to show the effect of the acoustic preprocessing pipeline.
9. **Execute Human-in-the-Loop Review**:
   Click **Confirm Detection** or **Reject False Alarm**, enter analyst notes, and submit. The decision is recorded in the relational database.
10. **Export Mission Reports**:
    Click **Export JSON Report** and **Export CSV Log** to demonstrate mission intelligence data transfer.

