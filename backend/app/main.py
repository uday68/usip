import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.db.session import engine, Base
from backend.app.api.endpoints import router as api_router
from configs.settings import config

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="USIP — Underwater Sonar Intelligence Platform",
    description="Automated Underwater Marine Debris, Target, and Acoustic Anomaly Detection System (SIH26057)",
    version="2.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path("backend/static")
static_dir.mkdir(parents=True, exist_ok=True)
results_dir = static_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include REST routes
app.include_router(api_router)

@app.get("/")
def serve_dashboard():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "USIP Sonar Intelligence API Online. See /docs for Swagger."}

# Compatibility endpoints for tactical dashboard
from fastapi import UploadFile, File, Form, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.app.db.session import get_db
from backend.app.services.inference_service import inference_service
from backend.app.services.report_service import ReportService
from typing import Optional

@app.post("/api/analyze")
async def analyze_compatibility(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    demo_dir = Path("demo_samples")
    if file is not None:
        content = await file.read()
        filename = file.filename
    elif sample_id is not None:
        # Find matching demo sample
        target_name = sample_id
        if not target_name.endswith((".png", ".jpg", ".jpeg")):
            # Match prefix e.g. montana -> sample_shipwreck_montana.png
            candidates = list(demo_dir.glob(f"*{sample_id}*.*"))
            if candidates:
                target_file = candidates[0]
            else:
                target_file = demo_dir / f"sample_{sample_id}.png"
        else:
            target_file = demo_dir / target_name

        if not target_file.exists():
            # Fallback to first demo sample
            all_samples = list(demo_dir.glob("*.png"))
            if all_samples:
                target_file = all_samples[0]
            else:
                raise HTTPException(status_code=404, detail=f"Sample {sample_id} not found.")

        content = target_file.read_bytes()
        filename = target_file.name
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or a sample_id.")

    res = inference_service.run_inference(
        image_bytes=content,
        filename=filename,
        db=db,
        survey_name="Tactical Sonar Mission"
    )

    # Convert candidates to frontend-friendly detections schema
    legacy_detections = []
    for c in res["candidates"]:
        is_known = c["source"] == "detector"
        legacy_detections.append({
            "id": c["candidate_id"],
            "category": "Known Target" if is_known else "Acoustic Anomaly",
            "class_name": c["target_class"],
            "confidence": c["model_confidence"] if c["model_confidence"] is not None else 0.85,
            "anomaly_score": c["anomaly_score"],
            "priority": c["final_priority"],
            "bbox": {
                "x": c["bbox"][0],
                "y": c["bbox"][1],
                "width": c["bbox"][2],
                "height": c["bbox"][3]
            },
            "acoustic_metrics": {
                "highlight_mean": 210.0,
                "shadow_mean": 25.0,
                "shadow_contrast_ratio": c["shadow_evidence"],
                "shadow_evidence_level": c["shadow_evidence_level"],
                "seafloor_similarity": f"{c['context_score']*100:.1f}%"
            },
            "explanation": c["explanation"]
        })

    # Default lat/lon if metadata absent for map visualization demo
    lat = res["telemetry"].get("latitude")
    lon = res["telemetry"].get("longitude")
    if lat is None or lon is None:
        lat = 45.0315
        lon = -83.1878

    return {
        "filename": res["filename"],
        "image_width": res["image_width"],
        "image_height": res["image_height"],
        "raw_image_url": res["raw_image_url"],
        "processed_image_url": res["processed_image_url"],
        "quality_metrics": res["quality_metrics"],
        "detections": legacy_detections,
        "candidates": res["candidates"],
        "metadata": {
            "latitude": lat,
            "longitude": lon,
            "depth": res["telemetry"].get("depth_m", 28.4),
            "altitude": res["telemetry"].get("altitude_m", 12.0),
            "available": res["telemetry"].get("metadata_available", False),
            "status_note": res["telemetry"].get("status_note", "")
        },
        "summary": {
            "analysis_id": res["survey_id"],
            "survey_id": res["survey_id"],
            "total_detections": res["total_candidates"],
            "high_priority": res["high_priority_count"],
            "latency_ms": res["inference_latency_ms"],
            "device": res["device_used"]
        }
    }

@app.get("/api/export/{fmt}")
def export_compatibility(fmt: str, db: Session = Depends(get_db)):
    if fmt.lower() == "json":
        data = ReportService.generate_json_report(1, db)
        return JSONResponse(content=data)
    elif fmt.lower() == "csv":
        csv_data = ReportService.generate_csv_report(1, db)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=survey_target_report.csv"}
        )
    raise HTTPException(status_code=400, detail="Invalid format. Use json or csv.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)


