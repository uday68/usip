import pytest
from pathlib import Path
from ml.metadata.parser import MetadataIngestionEngine
from ml.metadata.geolocation import lookup_shipwreck_coordinates, local_xy_to_wgs84

def test_historic_shipwreck_lookup():
    coords = lookup_shipwreck_coordinates("EB_Allen_01.png")
    assert coords is not None
    lat, lon, sanctuary = coords
    assert round(lat, 2) == 45.03
    assert round(lon, 2) == -83.19
    assert "Thunder Bay" in sanctuary

def test_honest_fallback_missing_metadata():
    engine = MetadataIngestionEngine()
    meta = engine.parse("unknown_random_sonar_scan.png")
    assert meta.latitude is None
    assert meta.longitude is None
    assert meta.metadata_available is False
    assert "Geolocation unavailable" in meta.status_note

def test_local_xy_projection():
    lat, lon = local_xy_to_wgs84(100.0, 200.0, origin_lat=45.0, origin_lon=-83.0)
    assert round(lat, 3) == 45.002
    assert round(lon, 3) == -82.999

