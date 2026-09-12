from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class SonarMetadata(BaseModel):
    survey_id: str = Field(description="Survey run identifier or vessel track")
    source_file: str = Field(description="Raw sonar image or data file name")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of acquisition")
    ping_id: Optional[int] = Field(None, description="Acoustic ping index")
    latitude: Optional[float] = Field(None, description="WGS84 Latitude")
    longitude: Optional[float] = Field(None, description="WGS84 Longitude")
    x_local_m: Optional[float] = Field(None, description="Local Cartesian Easting (meters)")
    y_local_m: Optional[float] = Field(None, description="Local Cartesian Northing (meters)")
    depth_m: Optional[float] = Field(None, description="Vehicle depth below sea surface (meters)")
    altitude_m: Optional[float] = Field(None, description="Vehicle altitude above seabed (meters)")
    heading_deg: Optional[float] = Field(None, description="Vehicle heading in degrees")
    metadata_available: bool = Field(False, description="True if valid navigation metadata is present")
    status_note: str = Field("Geolocation unavailable — no valid navigation metadata associated with candidate.")

