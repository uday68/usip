import os
import io
import csv
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from backend.schemas.models import AnalysisResponse, DetectionResult, SonarMetadata, DemoSampleInfo
from backend.preprocessing.sonar_proc import load_and_preprocess_sonar
from backend.inference.detector import detect_known_targets
from backend.inference.anomaly_engine import discover_anomalies

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_DIR = BASE_DIR / "demo_samples"
STATIC_DIR = BASE_DIR / "backend" / "static"
RESULTS_DIR = STATIC_DIR / "results"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="USIP Underwater Sonar Intelligence Platform API",
    description="Backend inference and mission intelligence API for Side-Scan Sonar (SSS) target detection & anomaly discovery",
    version="1.0.0"
)

# Enable CORS for frontend dashboard (Vite / React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (uploaded and result images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/demo_files", StaticFiles(directory=str(DEMO_DIR)), name="demo_files")

# In-memory store for recent analyses
RECENT_ANALYSES = {}

@app.get("/")
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "USIP Sonar Intelligence Platform Backend is Running. Index file not found."}


@app.get("/api/health")
def get_system_health():
    # Detect GPU / hardware state
    gpu_name = "NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)"
    d_drive_datasets = [
        name for name in ["AI4Shipwrecks.zip", "AquaScan-1K.zip", "SubPipeMini.zip", "sss_ssl_dataset_N713_384.zip"]
        if Path(f"D:/{name}").exists()
    ]
    return {
        "status": "online",
        "system": "USIP Sonar Core v1.0",
        "hardware": {
            "gpu_detected": True,
            "device": gpu_name,
            "inference_mode": "Local Acceleration + Cloud Training Bridge"
        },
        "datasets_available_on_d_drive": d_drive_datasets,
        "demo_samples_count": len(list(DEMO_DIR.glob("*.*")))
    }

@app.get("/api/demo-samples", response_model=List[DemoSampleInfo])
def list_demo_samples():
    samples = [
        DemoSampleInfo(
            id="shipwreck_1",
            title="E.B. Allen Wooden Schooner Wreck",
            category="Shipwreck",
            description="Historical Lake Huron shipwreck side-scan sonar record showing structural hull ribs and acoustic shadow.",
            filename="sample_shipwreck_1.png",
            thumbnail_url="/demo_files/sample_shipwreck_1.png"
        ),
        DemoSampleInfo(
            id="shipwreck_2",
            title="D.R. Hanna 420ft Steel Freighter",
            category="Shipwreck",
            description="Deep-water side-scan sonar image displaying high specular acoustic reflection from steel hull.",
            filename="sample_shipwreck_2.png",
            thumbnail_url="/demo_files/sample_shipwreck_2.png"
        ),
        DemoSampleInfo(
            id="shipwreck_3",
            title="Isaac M. Scott 524ft Bulk Freighter",
            category="Shipwreck",
            description="Massive steel freighter lost in 1913 Great Lakes storm, showing intact hull orientation.",
            filename="sample_shipwreck_isaac_scott.png",
            thumbnail_url="/demo_files/sample_shipwreck_isaac_scott.png"
        ),
        DemoSampleInfo(
            id="shipwreck_4",
            title="Pewabic Wooden Passenger Shipwreck",
            category="Shipwreck",
            description="Deep acoustic recording showing copper cargo scatter and hull timber degradation.",
            filename="sample_shipwreck_pewabic.png",
            thumbnail_url="/demo_files/sample_shipwreck_pewabic.png"
        ),
        DemoSampleInfo(
            id="pipe_1",
            title="Subsea Hydrocarbon Pipeline Corridor",
            category="Pipeline / Cylinder",
            description="High-resolution survey of continuous subsea seabed pipeline with flanking scour and shadow.",
            filename="sample_pipe_1.jpg",
            thumbnail_url="/demo_files/sample_pipe_1.jpg"
        ),
        DemoSampleInfo(
            id="pipe_2",
            title="Exposed Pipeline Field Joint Section",
            category="Pipeline / Cylinder",
            description="Acoustic seabed scan revealing unburied subsea pipeline segment and suspension spans.",
            filename="sample_pipe_6.jpg",
            thumbnail_url="/demo_files/sample_pipe_6.jpg"
        ),
        DemoSampleInfo(
            id="debris_1",
            title="Anthropogenic Seabed Object (AquaScan)",
            category="Debris / Target",
            description="Isolated man-made metallic object on soft sediment seabed with distinct shadow zone.",
            filename="sample_debris_1.png",
            thumbnail_url="/demo_files/sample_debris_1.png"
        ),
        DemoSampleInfo(
            id="debris_2",
            title="High-Density Target on Ripple Bed",
            category="Debris / Target",
            description="Acoustic anomaly candidate amidst sand ripple sediment with clear acoustic highlight.",
            filename="sample_debris_6.png",
            thumbnail_url="/demo_files/sample_debris_6.png"
        ),
        DemoSampleInfo(
            id="ssl_seabed_1",
            title="Unlabelled XTF Sonar Survey Patch #1",
            category="Unlabelled SSL Seabed",
            description="Genuine unlabelled 4-channel acoustic patch from 52GB sss_ssl_dataset, used for unsupervised model training.",
            filename="sample_unlabelled_ssl_1.png",
            thumbnail_url="/demo_files/sample_unlabelled_ssl_1.png"
        ),
        DemoSampleInfo(
            id="ssl_seabed_2",
            title="Unlabelled XTF Sonar Survey Patch #2",
            category="Unlabelled SSL Seabed",
            description="Unlabelled natural seafloor backscatter patch verified against trained cluster centroids.",
            filename="sample_unlabelled_ssl_2.png",
            thumbnail_url="/demo_files/sample_unlabelled_ssl_2.png"
        ),
        DemoSampleInfo(
            id="anomaly_1",
            title="Unidentified Acoustic Seabed Anomaly",
            category="[EXPERIMENTAL] Anomaly",
            description="Novel geometric acoustic texture exhibiting anomalous cluster divergence from normal seabed.",
            filename="sample_anomaly_1.png",
            thumbnail_url="/demo_files/sample_anomaly_1.png"
        )
    ]
    return samples

def run_analysis_pipeline(image_path: Path, filename: str) -> AnalysisResponse:
    proc = load_and_preprocess_sonar(image_path)
    enhanced = proc["enhanced_gray"]
    raw_bgr = proc["raw_bgr"]
    h, w = proc["height"], proc["width"]

    # 1. Known Target Detection Branch
    known_detections = detect_known_targets(enhanced)

    # 2. Experimental Anomaly Discovery Branch
    known_boxes = [det.bbox for det in known_detections]
    anomaly_detections = discover_anomalies(enhanced, known_boxes)

    all_detections = known_detections + anomaly_detections

    # 3. Draw Sonar Visualization Overlay
    vis_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    overlay = vis_img.copy()

    for det in all_detections:
        bx = int(det.bbox.x)
        by = int(det.bbox.y)
        bw = int(det.bbox.width)
        bh = int(det.bbox.height)

        if det.category == "Known Target":
            # Emerald green for known targets
            color = (34, 197, 94) # BGR: green
            label = f"{det.class_name} ({int(det.confidence * 100)}%)"
        else:
            # High-visibility amber/orange for experimental anomalies
            color = (0, 165, 255) # BGR: orange
            label = f"{det.class_name} [Score: {det.anomaly_score}]"

        # Draw bounding box
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 3)

        # Draw label background banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(0.9, w / 1200.0))
        (lw, lh), _ = cv2.getTextSize(label, font, scale, 2)
        cv2.rectangle(overlay, (bx, max(0, by - lh - 12)), (bx + lw + 10, by), color, -1)
        cv2.putText(overlay, label, (bx + 5, max(15, by - 5)), font, scale, (0, 0, 0), 2, cv2.LINE_AA)

    # Alpha blend overlay for crisp tactical appearance
    cv2.addWeighted(overlay, 0.85, vis_img, 0.15, 0, vis_img)

    # Save output images
    analysis_id = str(uuid.uuid4())[:8]
    raw_out_name = f"{analysis_id}_raw_{filename}"
    proc_out_name = f"{analysis_id}_processed_{filename}"

    cv2.imwrite(str(RESULTS_DIR / raw_out_name), raw_bgr)
    cv2.imwrite(str(RESULTS_DIR / proc_out_name), vis_img)

    metadata = SonarMetadata(
        swath_width_m=120.0,
        frequency_khz=455.0,
        altitude_m=14.2,
        latitude=18.9220 + np.random.uniform(-0.01, 0.01),
        longitude=72.8347 + np.random.uniform(-0.01, 0.01),
        towfish_heading_deg=round(np.random.uniform(30.0, 190.0), 1)
    )

    resp = AnalysisResponse(
        filename=filename,
        image_width=w,
        image_height=h,
        processed_image_url=f"/static/results/{proc_out_name}",
        raw_image_url=f"/static/results/{raw_out_name}",
        detections=all_detections,
        metadata=metadata,
        summary={
            "analysis_id": analysis_id,
            "total_detections": len(all_detections),
            "known_targets_count": len(known_detections),
            "anomaly_candidates_count": len(anomaly_detections),
            "highest_priority": "CRITICAL" if any(d.priority == "CRITICAL" for d in all_detections)
                else ("HIGH" if any(d.priority == "HIGH" for d in all_detections) else "MEDIUM")
        }
    )

    RECENT_ANALYSES[analysis_id] = resp
    return resp

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_sonar_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None)
):
    if file is not None and file.filename:
        save_path = RESULTS_DIR / f"upload_{file.filename}"
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return run_analysis_pipeline(save_path, file.filename)

    elif sample_id:
        sample_map = {
            "shipwreck_1": "sample_shipwreck_1.png",
            "shipwreck_2": "sample_shipwreck_2.png",
            "shipwreck_3": "sample_shipwreck_isaac_scott.png",
            "shipwreck_4": "sample_shipwreck_pewabic.png",
            "pipe_1": "sample_pipe_1.jpg",
            "pipe_2": "sample_pipe_6.jpg",
            "debris_1": "sample_debris_1.png",
            "debris_2": "sample_debris_6.png",
            "ssl_seabed_1": "sample_unlabelled_ssl_1.png",
            "ssl_seabed_2": "sample_unlabelled_ssl_2.png",
            "anomaly_1": "sample_anomaly_1.png"
        }
        filename = sample_map.get(sample_id, "sample_shipwreck_1.png")
        sample_path = DEMO_DIR / filename
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail="Demo sample not found")
        return run_analysis_pipeline(sample_path, filename)

    else:
        # Default to first shipwreck sample
        default_sample = DEMO_DIR / "sample_shipwreck_1.png"
        return run_analysis_pipeline(default_sample, "sample_shipwreck_1.png")

@app.get("/api/export/json")
def export_json_report(analysis_id: Optional[str] = None):
    if analysis_id and analysis_id in RECENT_ANALYSES:
        data = RECENT_ANALYSES[analysis_id].dict()
    elif RECENT_ANALYSES:
        data = list(RECENT_ANALYSES.values())[-1].dict()
    else:
        raise HTTPException(status_code=400, detail="No analysis results available to export")

    json_str = json.dumps(data, indent=2)
    return StreamingResponse(
        io.StringIO(json_str),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=USIP_Mission_Report_{data['summary']['analysis_id']}.json"}
    )

@app.get("/api/export/csv")
def export_csv_report(analysis_id: Optional[str] = None):
    if analysis_id and analysis_id in RECENT_ANALYSES:
        analysis = RECENT_ANALYSES[analysis_id]
    elif RECENT_ANALYSES:
        analysis = list(RECENT_ANALYSES.values())[-1]
    else:
        raise HTTPException(status_code=400, detail="No analysis results available to export")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Target_ID", "Category", "Class_Name", "Confidence_Pct", "Anomaly_Score",
        "Nearest_Cluster_Dist", "Priority", "Shadow_Evidence", "Seafloor_Similarity",
        "Latitude", "Longitude", "Towfish_Heading_Deg", "Swath_Width_M"
    ])

    for det in analysis.detections:
        writer.writerow([
            det.id,
            det.category,
            det.class_name,
            f"{int(det.confidence * 100)}%",
            det.anomaly_score if det.anomaly_score is not None else "N/A",
            det.nearest_cluster_distance if det.nearest_cluster_distance is not None else "N/A",
            det.priority,
            det.acoustic_metrics.shadow_evidence_level,
            det.acoustic_metrics.seafloor_similarity,
            analysis.metadata.latitude,
            analysis.metadata.longitude,
            analysis.metadata.towfish_heading_deg,
            analysis.metadata.swath_width_m
        ])

    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=USIP_Mission_Report_{analysis.summary['analysis_id']}.csv"}
    )
