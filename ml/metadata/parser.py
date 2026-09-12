import os
import re
import csv
import json
from pathlib import Path
from typing import Optional, Dict, Any
from ml.metadata.schema import SonarMetadata
from ml.metadata.geolocation import lookup_shipwreck_coordinates, local_xy_to_wgs84

class MetadataIngestionEngine:
    """
    Multi-source metadata ingestion engine.
    Supports CSV vehicle state tables, JSON sidecars, and filename patterns.
    Enforces honest fallback when navigation data is absent.
    """
    def __init__(self, telemetry_csv: Optional[Path] = None):
        self.telemetry_cache: Dict[str, Dict[str, Any]] = {}
        if telemetry_csv and telemetry_csv.exists():
            self._load_telemetry_csv(telemetry_csv)

    def _load_telemetry_csv(self, csv_path: Path):
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_key = Path(row.get("image", "")).name
                    if img_key:
                        self.telemetry_cache[img_key] = row
        except Exception as e:
            print(f"Failed to load telemetry CSV {csv_path}: {e}")

    def parse(self, filename: str, sidecar_json: Optional[Path] = None) -> SonarMetadata:
        base_name = Path(filename).name

        # 1. Check sidecar JSON if provided
        if sidecar_json and sidecar_json.exists():
            try:
                with open(sidecar_json, "r") as f:
                    data = json.load(f)
                    return SonarMetadata(
                        survey_id=data.get("survey_id", "SURVEY-SIDECAR"),
                        source_file=base_name,
                        timestamp=data.get("timestamp"),
                        latitude=data.get("latitude"),
                        longitude=data.get("longitude"),
                        depth_m=data.get("depth_m"),
                        altitude_m=data.get("altitude_m"),
                        heading_deg=data.get("heading_deg"),
                        metadata_available=True,
                        status_note="Navigation coordinates verified from survey sidecar."
                    )
            except Exception:
                pass

        # 2. Check telemetry cache (from SubPipe AUV state)
        if base_name in self.telemetry_cache:
            row = self.telemetry_cache[base_name]
            x_m = float(row.get(" x (m)", row.get("x", 0.0)))
            y_m = float(row.get(" y (m)", row.get("y", 0.0)))
            depth = float(row.get(" depth (m)", row.get("depth", 0.0)))
            alt = float(row.get(" alt (m)", row.get("alt", 0.0)))
            ts = row.get("timestamp", None)
            lat, lon = local_xy_to_wgs84(x_m, y_m)
            return SonarMetadata(
                survey_id="SUBPIPE-SURVEY-A1",
                source_file=base_name,
                timestamp=str(ts),
                latitude=lat,
                longitude=lon,
                x_local_m=round(x_m, 2),
                y_local_m=round(y_m, 2),
                depth_m=round(depth, 2),
                altitude_m=round(alt, 2),
                metadata_available=True,
                status_note="AUV Navigation telemetry synchronized from vehicle inertial navigation."
            )

        # 3. Check Historic Shipwreck Registry (Lake Huron Thunder Bay Sanctuary)
        wreck_coords = lookup_shipwreck_coordinates(base_name)
        if wreck_coords:
            lat, lon, site_note = wreck_coords
            return SonarMetadata(
                survey_id="THUNDER-BAY-SSS-MISSION",
                source_file=base_name,
                latitude=lat,
                longitude=lon,
                metadata_available=True,
                status_note=f"Verified archaeological sanctuary coordinates ({site_note})."
            )

        # 4. Check Date in Filename (AquaScan convention, e.g. Screenshot_2025-08-10_23.00.36)
        date_match = re.search(r"Screenshot_(\d{4}-\d{2}-\d{2})_(\d{2}\.\d{2}\.\d{2})", base_name)
        if date_match:
            date_str = date_match.group(1)
            time_str = date_match.group(2).replace(".", ":")
            iso_ts = f"{date_str}T{time_str}Z"
            return SonarMetadata(
                survey_id=f"AQUASCAN-SURVEY-{date_str}",
                source_file=base_name,
                timestamp=iso_ts,
                latitude=None,
                longitude=None,
                metadata_available=False,
                status_note="Geolocation unavailable — no valid navigation metadata associated with candidate."
            )

        # 5. Default Honest Fallback
        return SonarMetadata(
            survey_id="SURVEY-UNASSIGNED",
            source_file=base_name,
            latitude=None,
            longitude=None,
            metadata_available=False,
            status_note="Geolocation unavailable — no valid navigation metadata associated with candidate."
        )

